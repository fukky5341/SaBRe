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
execution time: IAR + RelationalAnalysis = 2.12 + 2.93 = 5.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7995453

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7979994
time: 1.74 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.51 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.51
Output dim: 0, lower bound: -0.7995453, upper bound: 0.7979994
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.51
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

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.43 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.51 seconds

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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.42 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
time: 1.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.02 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.02
Output dim: 0, lower bound: -0.7979994, upper bound: 0.7979994

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

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7933224, upper bound: 0.7913886
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7927763, upper bound: 0.7913886
time: 1.38 seconds

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

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7933224, upper bound: 0.7913886
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7927763, upper bound: 0.7913886
time: 1.40 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0554657, 1.1135265, 0.2186534, 1.0904168, -1.0349511, 0.8948731
1: -0.5040710, 0.4977783, -0.3835300, 0.3779540, -0.8820250, 0.8813083
2: -0.3895540, 0.5955865, -0.2930107, 0.4877503, -0.8773042, 0.8885972
3: -0.3476263, 0.5043057, -0.2813488, 0.3669437, -0.7145700, 0.7856544
4: -0.5146475, 0.5465473, -0.4004387, 0.3811752, -0.8958226, 0.9469860
5: -0.5675435, 0.6660394, -0.4304177, 0.5497525, -1.1172960, 1.0964570
6: -0.4086787, 0.5632119, -0.3121916, 0.4397539, -0.8484325, 0.8754035
7: -0.4989775, 0.5481607, -0.4026238, 0.4266526, -0.9256301, 0.9507844
8: -0.5505396, 0.6379345, -0.4137142, 0.5166412, -1.0671808, 1.0516487
9: -0.5090230, 0.6205251, -0.3955224, 0.4956400, -1.0046629, 1.0160475

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7916756, upper bound: 0.7913886
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7913886, upper bound: 0.7913886
time: 1.39 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0554657, 1.1135265, 0.0554657, 1.1135265, -1.0580608, 1.0580608
1: -0.5040710, 0.4977783, -0.5040710, 0.4977783, -1.0018493, 1.0018493
2: -0.3895540, 0.5955865, -0.3895540, 0.5955865, -0.9851405, 0.9851405
3: -0.3476263, 0.5043057, -0.3476263, 0.5043057, -0.8519319, 0.8519319
4: -0.5146475, 0.5465473, -0.5146475, 0.5465473, -1.0611948, 1.0611948
5: -0.5675435, 0.6660394, -0.5675435, 0.6660394, -1.2335830, 1.2335830
6: -0.4086787, 0.5632119, -0.4086787, 0.5632119, -0.9718905, 0.9718905
7: -0.4989775, 0.5481607, -0.4989775, 0.5481607, -1.0471382, 1.0471382
8: -0.5505396, 0.6379345, -0.5505396, 0.6379345, -1.1884741, 1.1884741
9: -0.5090230, 0.6205251, -0.5090230, 0.6205251, -1.1295481, 1.1295481

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7916756, upper bound: 0.7913886
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7913886, upper bound: 0.7913886
time: 1.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.86 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7933224, upper bound: 0.7913886
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7927763, upper bound: 0.7913886
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7933224, upper bound: 0.7913886
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7927763, upper bound: 0.7913886
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7916756, upper bound: 0.7913886
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7913886, upper bound: 0.7913886
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7916756, upper bound: 0.7913886
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.86
Output dim: 0, lower bound: -0.7913886, upper bound: 0.7913886

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.3666971, 1.0696577, 0.2186534, 1.0904168, -0.7237197, 0.8510043
1: -0.2998649, 0.2921253, -0.3835300, 0.3779540, -0.6778189, 0.6756553
2: -0.2137002, 0.3913441, -0.2930107, 0.4877503, -0.7014505, 0.6843548
3: -0.2196099, 0.2941632, -0.2813488, 0.3669437, -0.5865536, 0.5755120
4: -0.3118195, 0.2750432, -0.4004387, 0.3811752, -0.6929946, 0.6754819
5: -0.3377491, 0.4404216, -0.4304177, 0.5497525, -0.8875016, 0.8708392
6: -0.2387033, 0.3330199, -0.3121916, 0.4397539, -0.6784571, 0.6452116
7: -0.3240428, 0.3335895, -0.4026238, 0.4266526, -0.7506954, 0.7362133
8: -0.3133390, 0.4063403, -0.4137142, 0.5166412, -0.8299802, 0.8200545
9: -0.3043399, 0.3918386, -0.3955224, 0.4956400, -0.7999799, 0.7873610

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7925361
time: 2.08 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7927763
time: 2.10 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.3364010, 1.0748954, 0.2552564, 1.0852740, -0.7488730, 0.8196390
1: -0.3213227, 0.3128231, -0.3632197, 0.3552937, -0.6766163, 0.6760427
2: -0.2324356, 0.4158757, -0.2729273, 0.4639402, -0.6963758, 0.6888031
3: -0.2342138, 0.3133531, -0.2661830, 0.3491349, -0.5833488, 0.5795360
4: -0.3316072, 0.2974870, -0.3768937, 0.3467138, -0.6783209, 0.6743807
5: -0.3581833, 0.4681399, -0.4068483, 0.5233102, -0.8814936, 0.8749882
6: -0.2579479, 0.3488463, -0.2917694, 0.4108506, -0.6687984, 0.6406156
7: -0.3445697, 0.3568338, -0.3821481, 0.4038460, -0.7484157, 0.7389819
8: -0.3330806, 0.4326400, -0.3839063, 0.4892633, -0.8223439, 0.8165463
9: -0.3264823, 0.4200152, -0.3715027, 0.4709474, -0.7974296, 0.7915179

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7820158
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7874407
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.3666971, 1.0696577, 0.0554657, 1.1135265, -0.7468294, 1.0141920
1: -0.2998649, 0.2921253, -0.5040710, 0.4977783, -0.7976432, 0.7961963
2: -0.2137002, 0.3913441, -0.3895540, 0.5955865, -0.8092867, 0.7808980
3: -0.2196099, 0.2941632, -0.3476263, 0.5043057, -0.7239155, 0.6417895
4: -0.3118195, 0.2750432, -0.5146475, 0.5465473, -0.8583667, 0.7896907
5: -0.3377491, 0.4404216, -0.5675435, 0.6660394, -1.0037885, 1.0079651
6: -0.2387033, 0.3330199, -0.4086787, 0.5632119, -0.8019151, 0.7416986
7: -0.3240428, 0.3335895, -0.4989775, 0.5481607, -0.8722035, 0.8325670
8: -0.3133390, 0.4063403, -0.5505396, 0.6379345, -0.9512735, 0.9568799
9: -0.3043399, 0.3918386, -0.5090230, 0.6205251, -0.9248651, 0.9008616

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7909758
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7913886
time: 2.19 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.3364010, 1.0748954, 0.0972574, 1.1081629, -0.7717619, 0.9776380
1: -0.3213227, 0.3128231, -0.4746581, 0.4691257, -0.7904484, 0.7874812
2: -0.2324356, 0.4158757, -0.3661393, 0.5703748, -0.8028105, 0.7820150
3: -0.2342138, 0.3133531, -0.3314993, 0.4699053, -0.7041191, 0.6448523
4: -0.3316072, 0.2974870, -0.4876888, 0.5072688, -0.8388760, 0.7851759
5: -0.3581833, 0.4681399, -0.5330017, 0.6382977, -0.9964811, 1.0011417
6: -0.2579479, 0.3488463, -0.3860683, 0.5308915, -0.7888393, 0.7349146
7: -0.3445697, 0.3568338, -0.4765078, 0.5182509, -0.8628206, 0.8333416
8: -0.3330806, 0.4326400, -0.5182214, 0.6088630, -0.9419436, 0.9508614
9: -0.3264823, 0.4200152, -0.4822029, 0.5902983, -0.9167806, 0.9022180

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7808752
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7860014
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.2275838, 1.0902638, 0.2186534, 1.0904168, -0.8628330, 0.8716105
1: -0.3823259, 0.3769295, -0.3835300, 0.3779540, -0.7602800, 0.7604595
2: -0.2916646, 0.4866098, -0.2930107, 0.4877503, -0.7794149, 0.7796205
3: -0.2795692, 0.3658088, -0.2813488, 0.3669437, -0.6465130, 0.6471576
4: -0.3991448, 0.3798461, -0.4004387, 0.3811752, -0.7803199, 0.7802848
5: -0.4274701, 0.5471249, -0.4304177, 0.5497525, -0.9772226, 0.9775425
6: -0.3115780, 0.4312532, -0.3121916, 0.4397539, -0.7513319, 0.7434449
7: -0.4017522, 0.4251831, -0.4026238, 0.4266526, -0.8284048, 0.8278068
8: -0.4113429, 0.5137860, -0.4137142, 0.5166412, -0.9279841, 0.9275002
9: -0.3945765, 0.4944862, -0.3955224, 0.4956400, -0.8902165, 0.8900086

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7925361
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7927763
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.1923790, 1.0966504, 0.2552564, 1.0852740, -0.8928950, 0.8413939
1: -0.4111070, 0.4073653, -0.3632197, 0.3552937, -0.7664006, 0.7705849
2: -0.3153429, 0.5159366, -0.2729273, 0.4639402, -0.7792830, 0.7888639
3: -0.2959042, 0.3956319, -0.2661830, 0.3491349, -0.6450391, 0.6618149
4: -0.4293916, 0.4228130, -0.3768937, 0.3467138, -0.7761054, 0.7997067
5: -0.4572200, 0.5774447, -0.4068483, 0.5233102, -0.9805303, 0.9842930
6: -0.3375357, 0.4563557, -0.2917694, 0.4108506, -0.7483863, 0.7481251
7: -0.4280876, 0.4534785, -0.3821481, 0.4038460, -0.8319336, 0.8356267
8: -0.4476635, 0.5450330, -0.3839063, 0.4892633, -0.9369267, 0.9289393
9: -0.4244551, 0.5251222, -0.3715027, 0.4709474, -0.8954025, 0.8966249

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7820158
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7874407
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.2275838, 1.0902638, 0.0554657, 1.1135265, -0.8859427, 1.0347981
1: -0.3823259, 0.3769295, -0.5040710, 0.4977783, -0.8801042, 0.8810005
2: -0.2916646, 0.4866098, -0.3895540, 0.5955865, -0.8872511, 0.8761637
3: -0.2795692, 0.3658088, -0.3476263, 0.5043057, -0.7838749, 0.7134351
4: -0.3991448, 0.3798461, -0.5146475, 0.5465473, -0.9456921, 0.8944936
5: -0.4274701, 0.5471249, -0.5675435, 0.6660394, -1.0935096, 1.1146684
6: -0.3115780, 0.4312532, -0.4086787, 0.5632119, -0.8747899, 0.8399318
7: -0.4017522, 0.4251831, -0.4989775, 0.5481607, -0.9499128, 0.9241605
8: -0.4113429, 0.5137860, -0.5505396, 0.6379345, -1.0492773, 1.0643256
9: -0.3945765, 0.4944862, -0.5090230, 0.6205251, -1.0151017, 1.0035092

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7909758
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7913886
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.1923790, 1.0966504, 0.0972574, 1.1081629, -0.9157839, 0.9993930
1: -0.4111070, 0.4073653, -0.4746581, 0.4691257, -0.8802327, 0.8820234
2: -0.3153429, 0.5159366, -0.3661393, 0.5703748, -0.8857177, 0.8820759
3: -0.2959042, 0.3956319, -0.3314993, 0.4699053, -0.7658094, 0.7271312
4: -0.4293916, 0.4228130, -0.4876888, 0.5072688, -0.9366604, 0.9105018
5: -0.4572200, 0.5774447, -0.5330017, 0.6382977, -1.0955178, 1.1104465
6: -0.3375357, 0.4563557, -0.3860683, 0.5308915, -0.8684272, 0.8424240
7: -0.4280876, 0.4534785, -0.4765078, 0.5182509, -0.9463385, 0.9299864
8: -0.4476635, 0.5450330, -0.5182214, 0.6088630, -1.0565264, 1.0632544
9: -0.4244551, 0.5251222, -0.4822029, 0.5902983, -1.0147535, 1.0073252

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7808752
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7860014
time: 1.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.81 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7925361
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7927763
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7820158
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7874407
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7909758
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7925361, upper bound: 0.7913886
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7808752
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7874407, upper bound: 0.7860014
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7925361
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7927763
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7820158
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7874407
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7909758
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7909758, upper bound: 0.7913886
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7808752
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 0, lower bound: -0.7860014, upper bound: 0.7860014

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.3666971, 1.0696577, 0.3666971, 1.0696577, -0.7029606, 0.7029606
1: -0.2998649, 0.2921253, -0.2998649, 0.2921253, -0.5919902, 0.5919902
2: -0.2137002, 0.3913441, -0.2137002, 0.3913441, -0.6050442, 0.6050442
3: -0.2196099, 0.2941632, -0.2196099, 0.2941632, -0.5137731, 0.5137731
4: -0.3118195, 0.2750432, -0.3118195, 0.2750432, -0.5868627, 0.5868627
5: -0.3377491, 0.4404216, -0.3377491, 0.4404216, -0.7781707, 0.7781707
6: -0.2387033, 0.3330199, -0.2387033, 0.3330199, -0.5717232, 0.5717232
7: -0.3240428, 0.3335895, -0.3240428, 0.3335895, -0.6576322, 0.6576322
8: -0.3133390, 0.4063403, -0.3133390, 0.4063403, -0.7196794, 0.7196794
9: -0.3043399, 0.3918386, -0.3043399, 0.3918386, -0.6961786, 0.6961786

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7872482
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7872482
time: 1.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.3666971, 1.0696577, 0.3364010, 1.0748954, -0.7081983, 0.7332567
1: -0.2998649, 0.2921253, -0.3213227, 0.3128231, -0.6126879, 0.6134480
2: -0.2137002, 0.3913441, -0.2324356, 0.4158757, -0.6295758, 0.6237797
3: -0.2196099, 0.2941632, -0.2342138, 0.3133531, -0.5329629, 0.5283771
4: -0.3118195, 0.2750432, -0.3316072, 0.2974870, -0.6093065, 0.6066504
5: -0.3377491, 0.4404216, -0.3581833, 0.4681399, -0.8058890, 0.7986049
6: -0.2387033, 0.3330199, -0.2579479, 0.3488463, -0.5875495, 0.5909678
7: -0.3240428, 0.3335895, -0.3445697, 0.3568338, -0.6808765, 0.6781591
8: -0.3133390, 0.4063403, -0.3330806, 0.4326400, -0.7459790, 0.7394210
9: -0.3043399, 0.3918386, -0.3264823, 0.4200152, -0.7243551, 0.7183208

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7874407
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7874407
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.3686733, 1.0706941, 0.4821835, 1.0549498, -0.6862764, 0.5885105
1: -0.3035026, 0.2958409, -0.2372869, 0.2325563, -0.5360588, 0.5331278
2: -0.2166118, 0.3957256, -0.1580658, 0.3206545, -0.5372663, 0.5537914
3: -0.2209829, 0.2973216, -0.1728131, 0.2378457, -0.4588286, 0.4701347
4: -0.3151055, 0.2788742, -0.2538525, 0.2096823, -0.5247878, 0.5327268
5: -0.3392833, 0.4439522, -0.2708040, 0.3551852, -0.6944684, 0.7147561
6: -0.2425120, 0.3286508, -0.1846612, 0.2599395, -0.5024514, 0.5133120
7: -0.3278295, 0.3373355, -0.2653494, 0.2650805, -0.5929100, 0.6026850
8: -0.3153197, 0.4094285, -0.2505568, 0.3244537, -0.6397734, 0.6599853
9: -0.3084567, 0.3969768, -0.2411534, 0.3110517, -0.6195084, 0.6381302

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7816601, upper bound: 0.7725707
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7725707
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.3364010, 1.0748954, 0.3717514, 1.0702673, -0.7338663, 0.7031440
1: -0.3213227, 0.3128231, -0.3017069, 0.2941247, -0.6154473, 0.6145299
2: -0.2324356, 0.4158757, -0.2150241, 0.3936897, -0.6261253, 0.6308998
3: -0.2342138, 0.3133531, -0.2196766, 0.2957086, -0.5299225, 0.5330297
4: -0.3316072, 0.2974870, -0.3134444, 0.2769979, -0.6086050, 0.6109314
5: -0.3581833, 0.4681399, -0.3374265, 0.4415451, -0.7997284, 0.8055664
6: -0.2579479, 0.3488463, -0.2409432, 0.3267875, -0.5847354, 0.5897895
7: -0.3445697, 0.3568338, -0.3261352, 0.3353762, -0.6799458, 0.6829690
8: -0.3330806, 0.4326400, -0.3135634, 0.4071241, -0.7402048, 0.7462033
9: -0.3264823, 0.4200152, -0.3066315, 0.3946468, -0.7211291, 0.7266467

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7874407
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7874407
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.3666971, 1.0696577, 0.2275838, 1.0902638, -0.7235668, 0.8420739
1: -0.2998649, 0.2921253, -0.3823259, 0.3769295, -0.6767944, 0.6744512
2: -0.2137002, 0.3913441, -0.2916646, 0.4866098, -0.7003100, 0.6830087
3: -0.2196099, 0.2941632, -0.2795692, 0.3658088, -0.5854187, 0.5737325
4: -0.3118195, 0.2750432, -0.3991448, 0.3798461, -0.6916655, 0.6741880
5: -0.3377491, 0.4404216, -0.4274701, 0.5471249, -0.8848740, 0.8678917
6: -0.2387033, 0.3330199, -0.3115780, 0.4312532, -0.6699564, 0.6445979
7: -0.3240428, 0.3335895, -0.4017522, 0.4251831, -0.7492259, 0.7353417
8: -0.3133390, 0.4063403, -0.4113429, 0.5137860, -0.8271250, 0.8176832
9: -0.3043399, 0.3918386, -0.3945765, 0.4944862, -0.7988262, 0.7864151

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7856536
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7856536
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.3666971, 1.0696577, 0.1923790, 1.0966504, -0.7299533, 0.8772787
1: -0.2998649, 0.2921253, -0.4111070, 0.4073653, -0.7072302, 0.7032323
2: -0.2137002, 0.3913441, -0.3153429, 0.5159366, -0.7296367, 0.7066870
3: -0.2196099, 0.2941632, -0.2959042, 0.3956319, -0.6152418, 0.5900674
4: -0.3118195, 0.2750432, -0.4293916, 0.4228130, -0.7346324, 0.7044348
5: -0.3377491, 0.4404216, -0.4572200, 0.5774447, -0.9151938, 0.8976417
6: -0.2387033, 0.3330199, -0.3375357, 0.4563557, -0.6950589, 0.6705556
7: -0.3240428, 0.3335895, -0.4280876, 0.4534785, -0.7775213, 0.7616771
8: -0.3133390, 0.4063403, -0.4476635, 0.5450330, -0.8583720, 0.8540038
9: -0.3043399, 0.3918386, -0.4244551, 0.5251222, -0.8294622, 0.8162937

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7860014
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7860014
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.3686733, 1.0706941, 0.3509675, 1.0731500, -0.7044767, 0.7197266
1: -0.3035026, 0.2958409, -0.3138312, 0.3057122, -0.6092147, 0.6096720
2: -0.2166118, 0.3957256, -0.2257439, 0.4074356, -0.6240473, 0.6214695
3: -0.2209829, 0.2973216, -0.2284966, 0.3065989, -0.5275818, 0.5258182
4: -0.3151055, 0.2788742, -0.3246599, 0.2896672, -0.6047727, 0.6035342
5: -0.3392833, 0.4439522, -0.3499651, 0.4577985, -0.7970818, 0.7939173
6: -0.2425120, 0.3286508, -0.2515358, 0.3393685, -0.5818805, 0.5801866
7: -0.3278295, 0.3373355, -0.3375754, 0.3486063, -0.6764358, 0.6749109
8: -0.3153197, 0.4094285, -0.3254217, 0.4226831, -0.7380028, 0.7348502
9: -0.3084567, 0.3969768, -0.3189550, 0.4103798, -0.7188365, 0.7159318

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7815083, upper bound: 0.7718369
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7718369
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3364010, 1.0748954, 0.2386133, 1.0902946, -0.7538936, 0.8362820
1: -0.3213227, 0.3128231, -0.3815904, 0.3765238, -0.6978465, 0.6944135
2: -0.2324356, 0.4158757, -0.2906587, 0.4860782, -0.7185138, 0.7065344
3: -0.2342138, 0.3133531, -0.2777127, 0.3650096, -0.5992234, 0.5910658
4: -0.3316072, 0.2974870, -0.3983926, 0.3795892, -0.7111964, 0.6958796
5: -0.3581833, 0.4681399, -0.4243879, 0.5445215, -0.9027048, 0.8925278
6: -0.2579479, 0.3488463, -0.3117016, 0.4205734, -0.6785213, 0.6605479
7: -0.3445697, 0.3568338, -0.4014842, 0.4241085, -0.7686782, 0.7583179
8: -0.3330806, 0.4326400, -0.4093528, 0.5109652, -0.8440458, 0.8419927
9: -0.3264823, 0.4200152, -0.3943411, 0.4940148, -0.8204971, 0.8143562

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7860014
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7860014
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.2275838, 1.0902638, 0.3666971, 1.0696577, -0.8420739, 0.7235668
1: -0.3823259, 0.3769295, -0.2998649, 0.2921253, -0.6744512, 0.6767944
2: -0.2916646, 0.4866098, -0.2137002, 0.3913441, -0.6830087, 0.7003100
3: -0.2795692, 0.3658088, -0.2196099, 0.2941632, -0.5737325, 0.5854187
4: -0.3991448, 0.3798461, -0.3118195, 0.2750432, -0.6741880, 0.6916655
5: -0.4274701, 0.5471249, -0.3377491, 0.4404216, -0.8678917, 0.8848740
6: -0.3115780, 0.4312532, -0.2387033, 0.3330199, -0.6445979, 0.6699564
7: -0.4017522, 0.4251831, -0.3240428, 0.3335895, -0.7353417, 0.7492259
8: -0.4113429, 0.5137860, -0.3133390, 0.4063403, -0.8176832, 0.8271250
9: -0.3945765, 0.4944862, -0.3043399, 0.3918386, -0.7864151, 0.7988262

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7872482
time: 1.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7872482
time: 1.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.2275838, 1.0902638, 0.3364010, 1.0748954, -0.8473116, 0.7538629
1: -0.3823259, 0.3769295, -0.3213227, 0.3128231, -0.6951489, 0.6982521
2: -0.2916646, 0.4866098, -0.2324356, 0.4158757, -0.7075403, 0.7190454
3: -0.2795692, 0.3658088, -0.2342138, 0.3133531, -0.5929223, 0.6000227
4: -0.3991448, 0.3798461, -0.3316072, 0.2974870, -0.6966318, 0.7114533
5: -0.4274701, 0.5471249, -0.3581833, 0.4681399, -0.8956101, 0.9053082
6: -0.3115780, 0.4312532, -0.2579479, 0.3488463, -0.6604243, 0.6892011
7: -0.4017522, 0.4251831, -0.3445697, 0.3568338, -0.7585859, 0.7697527
8: -0.4113429, 0.5137860, -0.3330806, 0.4326400, -0.8439828, 0.8468666
9: -0.3945765, 0.4944862, -0.3264823, 0.4200152, -0.8145916, 0.8209685

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7874407
time: 2.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7874407
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.2322812, 1.0912958, 0.4821835, 1.0549498, -0.8226686, 0.6091123
1: -0.3855472, 0.3812227, -0.2372869, 0.2325563, -0.6181035, 0.6185096
2: -0.2943870, 0.4907170, -0.1580658, 0.3206545, -0.6150415, 0.6487828
3: -0.2804350, 0.3685404, -0.1728131, 0.2378457, -0.5182807, 0.5413536
4: -0.4030806, 0.3862872, -0.2538525, 0.2096823, -0.6127629, 0.6401398
5: -0.4288222, 0.5495177, -0.2708040, 0.3551852, -0.7840074, 0.8203217
6: -0.3157437, 0.4252924, -0.1846612, 0.2599395, -0.5756831, 0.6099536
7: -0.4056272, 0.4282870, -0.2653494, 0.2650805, -0.6707077, 0.6936364
8: -0.4151788, 0.5161407, -0.2505568, 0.3244537, -0.7396325, 0.7666975
9: -0.3988589, 0.4987577, -0.2411534, 0.3110517, -0.7099106, 0.7399111

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7800854, upper bound: 0.7725707
time: 1.71 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7777442, upper bound: 0.7725707
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.1923790, 1.0966504, 0.3717514, 1.0702673, -0.8778883, 0.7248990
1: -0.4111070, 0.4073653, -0.3017069, 0.2941247, -0.7052317, 0.7090721
2: -0.3153429, 0.5159366, -0.2150241, 0.3936897, -0.7090326, 0.7309606
3: -0.2959042, 0.3956319, -0.2196766, 0.2957086, -0.5916128, 0.6153085
4: -0.4293916, 0.4228130, -0.3134444, 0.2769979, -0.7063894, 0.7362574
5: -0.4572200, 0.5774447, -0.3374265, 0.4415451, -0.8987651, 0.9148711
6: -0.3375357, 0.4563557, -0.2409432, 0.3267875, -0.6643232, 0.6972989
7: -0.4280876, 0.4534785, -0.3261352, 0.3353762, -0.7634639, 0.7796137
8: -0.4476635, 0.5450330, -0.3135634, 0.4071241, -0.8547876, 0.8585963
9: -0.4244551, 0.5251222, -0.3066315, 0.3946468, -0.8191019, 0.8317537

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7874407
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7874407
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.2275838, 1.0902638, 0.2275838, 1.0902638, -0.8626801, 0.8626801
1: -0.3823259, 0.3769295, -0.3823259, 0.3769295, -0.7592554, 0.7592554
2: -0.2916646, 0.4866098, -0.2916646, 0.4866098, -0.7782744, 0.7782744
3: -0.2795692, 0.3658088, -0.2795692, 0.3658088, -0.6453781, 0.6453781
4: -0.3991448, 0.3798461, -0.3991448, 0.3798461, -0.7789909, 0.7789909
5: -0.4274701, 0.5471249, -0.4274701, 0.5471249, -0.9745950, 0.9745950
6: -0.3115780, 0.4312532, -0.3115780, 0.4312532, -0.7428312, 0.7428312
7: -0.4017522, 0.4251831, -0.4017522, 0.4251831, -0.8269352, 0.8269352
8: -0.4113429, 0.5137860, -0.4113429, 0.5137860, -0.9251288, 0.9251288
9: -0.3945765, 0.4944862, -0.3945765, 0.4944862, -0.8890628, 0.8890628

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7856536
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7856536
time: 1.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.2275838, 1.0902638, 0.1923790, 1.0966504, -0.8690666, 0.8978848
1: -0.3823259, 0.3769295, -0.4111070, 0.4073653, -0.7896912, 0.7880365
2: -0.2916646, 0.4866098, -0.3153429, 0.5159366, -0.8076012, 0.8019527
3: -0.2795692, 0.3658088, -0.2959042, 0.3956319, -0.6752012, 0.6617130
4: -0.3991448, 0.3798461, -0.4293916, 0.4228130, -0.8219577, 0.8092377
5: -0.4274701, 0.5471249, -0.4572200, 0.5774447, -1.0049148, 1.0043449
6: -0.3115780, 0.4312532, -0.3375357, 0.4563557, -0.7679337, 0.7687889
7: -0.4017522, 0.4251831, -0.4280876, 0.4534785, -0.8552307, 0.8532706
8: -0.4113429, 0.5137860, -0.4476635, 0.5450330, -0.9563758, 0.9614494
9: -0.3945765, 0.4944862, -0.4244551, 0.5251222, -0.9196987, 0.9189414

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7860014
time: 1.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7860014
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.2322812, 1.0912958, 0.3509675, 1.0731500, -0.8408688, 0.7403284
1: -0.3855472, 0.3812227, -0.3138312, 0.3057122, -0.6912594, 0.6950538
2: -0.2943870, 0.4907170, -0.2257439, 0.4074356, -0.7018226, 0.7164608
3: -0.2804350, 0.3685404, -0.2284966, 0.3065989, -0.5870339, 0.5970371
4: -0.4030806, 0.3862872, -0.3246599, 0.2896672, -0.6927479, 0.7109472
5: -0.4288222, 0.5495177, -0.3499651, 0.4577985, -0.8866208, 0.8994828
6: -0.3157437, 0.4252924, -0.2515358, 0.3393685, -0.6551122, 0.6768281
7: -0.4056272, 0.4282870, -0.3375754, 0.3486063, -0.7542336, 0.7658623
8: -0.4151788, 0.5161407, -0.3254217, 0.4226831, -0.8378619, 0.8415624
9: -0.3988589, 0.4987577, -0.3189550, 0.4103798, -0.8092388, 0.8177127

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7800816, upper bound: 0.7718369
time: 1.57 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7777442, upper bound: 0.7718369
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.1923790, 1.0966504, 0.2386133, 1.0902946, -0.8979156, 0.8580370
1: -0.4111070, 0.4073653, -0.3815904, 0.3765238, -0.7876308, 0.7889557
2: -0.3153429, 0.5159366, -0.2906587, 0.4860782, -0.8014211, 0.8065952
3: -0.2959042, 0.3956319, -0.2777127, 0.3650096, -0.6609138, 0.6733446
4: -0.4293916, 0.4228130, -0.3983926, 0.3795892, -0.8089808, 0.8212055
5: -0.4572200, 0.5774447, -0.4243879, 0.5445215, -1.0017415, 1.0018326
6: -0.3375357, 0.4563557, -0.3117016, 0.4205734, -0.7581091, 0.7680573
7: -0.4280876, 0.4534785, -0.4014842, 0.4241085, -0.8521961, 0.8549627
8: -0.4476635, 0.5450330, -0.4093528, 0.5109652, -0.9586287, 0.9543858
9: -0.4244551, 0.5251222, -0.3943411, 0.4940148, -0.9184700, 0.9194633

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7860014
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7860014
time: 1.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.22 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7872482
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7872482
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7874407
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7874407
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7816601, upper bound: 0.7725707
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7725707
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7874407
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7874407
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7856536
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7856536
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7823158, upper bound: 0.7860014
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7860014
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7815083, upper bound: 0.7718369
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7786506, upper bound: 0.7718369
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7860014
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7820158, upper bound: 0.7860014
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7872482
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7872482
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7874407
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7874407
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7800854, upper bound: 0.7725707
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7777442, upper bound: 0.7725707
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7874407
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7874407
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7856536
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7856536
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808319, upper bound: 0.7860014
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7860014
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7800816, upper bound: 0.7718369
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7777442, upper bound: 0.7718369
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7860014
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.22
Output dim: 0, lower bound: -0.7808752, upper bound: 0.7860014

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.5738506, 1.0415723, 0.4057236, 1.0655552, -0.4917046, 0.6358488
1: -0.1839559, 0.1786978, -0.2818894, 0.2751844, -0.4591403, 0.4605872
2: -0.1145039, 0.2592611, -0.1975021, 0.3712220, -0.4857259, 0.4567632
3: -0.1317621, 0.1913594, -0.2052600, 0.2779084, -0.4096706, 0.3966193
4: -0.2027689, 0.1581057, -0.2951121, 0.2562897, -0.4590585, 0.4532178
5: -0.2106911, 0.2920812, -0.3169314, 0.4149783, -0.6256694, 0.6090125
6: -0.1434024, 0.1996230, -0.2236292, 0.3062230, -0.4496255, 0.4232523
7: -0.2162840, 0.2041539, -0.3074358, 0.3137512, -0.5300353, 0.5115898
8: -0.1964648, 0.2498609, -0.2941808, 0.3816922, -0.5781570, 0.5440417
9: -0.1855037, 0.2474656, -0.2864886, 0.3689305, -0.5544342, 0.5339541

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7823295
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7782580
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.3666971, 1.0696577, -0.5974402, 0.6896350
1: -0.2431005, 0.2381126, -0.2998649, 0.2921253, -0.5352258, 0.5379775
2: -0.1632061, 0.3272455, -0.2137002, 0.3913441, -0.5545502, 0.5409456
3: -0.1770424, 0.2430676, -0.2196099, 0.2941632, -0.4712057, 0.4626775
4: -0.2592304, 0.2157572, -0.3118195, 0.2750432, -0.5342736, 0.5275766
5: -0.2768164, 0.3629788, -0.3377491, 0.4404216, -0.7172380, 0.7007279
6: -0.1897404, 0.2659722, -0.2387033, 0.3330199, -0.5227603, 0.5046754
7: -0.2708351, 0.2714246, -0.3240428, 0.3335895, -0.6044246, 0.5954673
8: -0.2562429, 0.3319142, -0.3133390, 0.4063403, -0.6625832, 0.6452532
9: -0.2470626, 0.3185959, -0.3043399, 0.3918386, -0.6389012, 0.6229358

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7823158
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7880314
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.5738506, 1.0415723, 0.3686733, 1.0706941, -0.4968435, 0.6728990
1: -0.1839559, 0.1786978, -0.3035026, 0.2958409, -0.4797968, 0.4822003
2: -0.1145039, 0.2592611, -0.2166118, 0.3957256, -0.5102295, 0.4758729
3: -0.1317621, 0.1913594, -0.2209829, 0.2973216, -0.4290837, 0.4123423
4: -0.2027689, 0.1581057, -0.3151055, 0.2788742, -0.4816431, 0.4732112
5: -0.2106911, 0.2920812, -0.3392833, 0.4439522, -0.6546433, 0.6313645
6: -0.1434024, 0.1996230, -0.2425120, 0.3286508, -0.4720532, 0.4421350
7: -0.2162840, 0.2041539, -0.3278295, 0.3373355, -0.5536196, 0.5319834
8: -0.1964648, 0.2498609, -0.3153197, 0.4094285, -0.6058933, 0.5651807
9: -0.1855037, 0.2474656, -0.3084567, 0.3969768, -0.5824805, 0.5559223

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7816753
time: 2.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7786506
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.3364010, 1.0748954, -0.6026779, 0.7199311
1: -0.2431005, 0.2381126, -0.3213227, 0.3128231, -0.5559236, 0.5594352
2: -0.1632061, 0.3272455, -0.2324356, 0.4158757, -0.5790818, 0.5596812
3: -0.1770424, 0.2430676, -0.2342138, 0.3133531, -0.4903955, 0.4772815
4: -0.2592304, 0.2157572, -0.3316072, 0.2974870, -0.5567175, 0.5473644
5: -0.2768164, 0.3629788, -0.3581833, 0.4681399, -0.7449564, 0.7211622
6: -0.1897404, 0.2659722, -0.2579479, 0.3488463, -0.5385866, 0.5239201
7: -0.2708351, 0.2714246, -0.3445697, 0.3568338, -0.6276689, 0.6159942
8: -0.2562429, 0.3319142, -0.3330806, 0.4326400, -0.6888828, 0.6649949
9: -0.2470626, 0.3185959, -0.3264823, 0.4200152, -0.6670778, 0.6450782

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7820158
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7874407
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.5443780, 1.0459632, 0.3717514, 1.0702673, -0.5258893, 0.6742118
1: -0.2010765, 0.1963980, -0.3017069, 0.2941247, -0.4952012, 0.4981049
2: -0.1278276, 0.2788188, -0.2150241, 0.3936897, -0.5215173, 0.4938429
3: -0.1453555, 0.2056128, -0.2196766, 0.2957086, -0.4410641, 0.4252894
4: -0.2192978, 0.1738993, -0.3134444, 0.2769979, -0.4962956, 0.4873437
5: -0.2309013, 0.3106825, -0.3374265, 0.4415451, -0.6724464, 0.6481090
6: -0.1553648, 0.2199016, -0.2409432, 0.3267875, -0.4821523, 0.4608448
7: -0.2318041, 0.2240368, -0.3261352, 0.3353762, -0.5671803, 0.5501720
8: -0.2140101, 0.2749390, -0.3135634, 0.4071241, -0.6211342, 0.5885024
9: -0.2033888, 0.2668447, -0.3066315, 0.3946468, -0.5980356, 0.5734762

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7816601
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7786506
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.4362575, 1.0613201, 0.3717514, 1.0702673, -0.6340098, 0.6895687
1: -0.2640777, 0.2581610, -0.3017069, 0.2941247, -0.5582023, 0.5598679
2: -0.1817534, 0.3510281, -0.2150241, 0.3936897, -0.5754430, 0.5660522
3: -0.1923025, 0.2619095, -0.2196766, 0.2957086, -0.4880111, 0.4815861
4: -0.2786354, 0.2376771, -0.3134444, 0.2769979, -0.5556332, 0.5511215
5: -0.2985106, 0.3911000, -0.3374265, 0.4415451, -0.7400557, 0.7285265
6: -0.2080675, 0.2877399, -0.2409432, 0.3267875, -0.5348550, 0.5286831
7: -0.2906288, 0.2943149, -0.3261352, 0.3353762, -0.6260049, 0.6204501
8: -0.2767597, 0.3588341, -0.3135634, 0.4071241, -0.6838838, 0.6723975
9: -0.2683841, 0.3458169, -0.3066315, 0.3946468, -0.6630309, 0.6524484

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7819102
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7786506
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.5738506, 1.0415723, 0.2711614, 1.0852319, -0.5113813, 0.7704109
1: -0.1839559, 0.1786978, -0.3618644, 0.3543747, -0.5383306, 0.5405622
2: -0.1145039, 0.2592611, -0.2712142, 0.4628203, -0.5773242, 0.5304754
3: -0.1317621, 0.1913594, -0.2633085, 0.3477496, -0.4795118, 0.4546679
4: -0.2027689, 0.1581057, -0.3754696, 0.3457245, -0.5484934, 0.5335752
5: -0.2106911, 0.2920812, -0.4020914, 0.5193099, -0.7300011, 0.6941726
6: -0.1434024, 0.1996230, -0.2916344, 0.3956037, -0.5390061, 0.4912575
7: -0.2162840, 0.2041539, -0.3814593, 0.4019997, -0.6182837, 0.5856132
8: -0.1964648, 0.2498609, -0.3804902, 0.4849254, -0.6813902, 0.6303512
9: -0.1855037, 0.2474656, -0.3708000, 0.4698921, -0.6553957, 0.6182656

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7805713
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7771950
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.2275838, 1.0902638, -0.6180463, 0.8287483
1: -0.2431005, 0.2381126, -0.3823259, 0.3769295, -0.6200300, 0.6204385
2: -0.1632061, 0.3272455, -0.2916646, 0.4866098, -0.6498159, 0.6189101
3: -0.1770424, 0.2430676, -0.2795692, 0.3658088, -0.5428513, 0.5226369
4: -0.2592304, 0.2157572, -0.3991448, 0.3798461, -0.6390765, 0.6149020
5: -0.2768164, 0.3629788, -0.4274701, 0.5471249, -0.8239413, 0.7904490
6: -0.1897404, 0.2659722, -0.3115780, 0.4312532, -0.6209936, 0.5775502
7: -0.2708351, 0.2714246, -0.4017522, 0.4251831, -0.6960182, 0.6731768
8: -0.2562429, 0.3319142, -0.4113429, 0.5137860, -0.7700288, 0.7432571
9: -0.2470626, 0.3185959, -0.3945765, 0.4944862, -0.7415488, 0.7131724

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7808319
time: 1.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7862869
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.5738506, 1.0415723, 0.2322812, 1.0912958, -0.5174452, 0.8092911
1: -0.1839559, 0.1786978, -0.3855472, 0.3812227, -0.5651786, 0.5642450
2: -0.1145039, 0.2592611, -0.2943870, 0.4907170, -0.6052209, 0.5536481
3: -0.1317621, 0.1913594, -0.2804350, 0.3685404, -0.5003026, 0.4717943
4: -0.2027689, 0.1581057, -0.4030806, 0.3862872, -0.5890561, 0.5611863
5: -0.2106911, 0.2920812, -0.4288222, 0.5495177, -0.7602088, 0.7209034
6: -0.1434024, 0.1996230, -0.3157437, 0.4252924, -0.5686948, 0.5153667
7: -0.2162840, 0.2041539, -0.4056272, 0.4282870, -0.6445710, 0.6097811
8: -0.1964648, 0.2498609, -0.4151788, 0.5161407, -0.7126055, 0.6650398
9: -0.1855037, 0.2474656, -0.3988589, 0.4987577, -0.6842613, 0.6463245

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7800982
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7777442
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.1923790, 1.0966504, -0.6244329, 0.8639531
1: -0.2431005, 0.2381126, -0.4111070, 0.4073653, -0.6504658, 0.6492196
2: -0.1632061, 0.3272455, -0.3153429, 0.5159366, -0.6791427, 0.6425884
3: -0.1770424, 0.2430676, -0.2959042, 0.3956319, -0.5726743, 0.5389718
4: -0.2592304, 0.2157572, -0.4293916, 0.4228130, -0.6820434, 0.6451488
5: -0.2768164, 0.3629788, -0.4572200, 0.5774447, -0.8542611, 0.8201989
6: -0.1897404, 0.2659722, -0.3375357, 0.4563557, -0.6460961, 0.6035079
7: -0.2708351, 0.2714246, -0.4280876, 0.4534785, -0.7243137, 0.6995122
8: -0.2562429, 0.3319142, -0.4476635, 0.5450330, -0.8012758, 0.7795777
9: -0.2470626, 0.3185959, -0.4244551, 0.5251222, -0.7721848, 0.7430511

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7808752
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7860014
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.5443780, 1.0459632, 0.2386133, 1.0902946, -0.5459166, 0.8073498
1: -0.2010765, 0.1963980, -0.3815904, 0.3765238, -0.5776003, 0.5779884
2: -0.1278276, 0.2788188, -0.2906587, 0.4860782, -0.6139058, 0.5694775
3: -0.1453555, 0.2056128, -0.2777127, 0.3650096, -0.5103651, 0.4833255
4: -0.2192978, 0.1738993, -0.3983926, 0.3795892, -0.5988870, 0.5722919
5: -0.2309013, 0.3106825, -0.4243879, 0.5445215, -0.7754228, 0.7350705
6: -0.1553648, 0.2199016, -0.3117016, 0.4205734, -0.5759382, 0.5316032
7: -0.2318041, 0.2240368, -0.4014842, 0.4241085, -0.6559126, 0.6255210
8: -0.2140101, 0.2749390, -0.4093528, 0.5109652, -0.7249752, 0.6842918
9: -0.2033888, 0.2668447, -0.3943411, 0.4940148, -0.6974036, 0.6611857

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7800854
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7777442
time: 1.54 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.4362575, 1.0613201, 0.2386133, 1.0902946, -0.6540371, 0.8227067
1: -0.2640777, 0.2581610, -0.3815904, 0.3765238, -0.6406015, 0.6397514
2: -0.1817534, 0.3510281, -0.2906587, 0.4860782, -0.6678315, 0.6416867
3: -0.1923025, 0.2619095, -0.2777127, 0.3650096, -0.5573121, 0.5396222
4: -0.2786354, 0.2376771, -0.3983926, 0.3795892, -0.6582246, 0.6360697
5: -0.2985106, 0.3911000, -0.4243879, 0.5445215, -0.8430321, 0.8154880
6: -0.2080675, 0.2877399, -0.3117016, 0.4205734, -0.6286409, 0.5994415
7: -0.2906288, 0.2943149, -0.4014842, 0.4241085, -0.7147373, 0.6957991
8: -0.2767597, 0.3588341, -0.4093528, 0.5109652, -0.7877249, 0.7681869
9: -0.2683841, 0.3458169, -0.3943411, 0.4940148, -0.7623990, 0.7401580

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7802552
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7777442
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.4494660, 1.0594878, 0.4057236, 1.0655552, -0.6160892, 0.6537642
1: -0.2563724, 0.2507969, -0.2818894, 0.2751844, -0.5315567, 0.5326864
2: -0.1749407, 0.3422925, -0.1975021, 0.3712220, -0.5461627, 0.5397946
3: -0.1866973, 0.2549886, -0.2052600, 0.2779084, -0.4646057, 0.4602486
4: -0.2715075, 0.2296257, -0.2951121, 0.2562897, -0.5277972, 0.5247378
5: -0.2905419, 0.3807709, -0.3169314, 0.4149783, -0.7055202, 0.6977023
6: -0.2013357, 0.2797443, -0.2236292, 0.3062230, -0.5075588, 0.5033735
7: -0.2833583, 0.2859068, -0.3074358, 0.3137512, -0.5971096, 0.5933426
8: -0.2692236, 0.3489462, -0.2941808, 0.3816922, -0.6509157, 0.6431270
9: -0.2605525, 0.3358182, -0.2864886, 0.3689305, -0.6294830, 0.6223068

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7821149
time: 1.41 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7782580
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.3666971, 1.0696577, -0.7257169, 0.7074277
1: -0.3179300, 0.3096298, -0.2998649, 0.2921253, -0.6100553, 0.6094947
2: -0.2293682, 0.4120828, -0.2137002, 0.3913441, -0.6207123, 0.6257830
3: -0.2314784, 0.3102807, -0.2196099, 0.2941632, -0.5256416, 0.5298905
4: -0.3284518, 0.2939504, -0.3118195, 0.2750432, -0.6034950, 0.6057699
5: -0.3542041, 0.4632936, -0.3377491, 0.4404216, -0.7946256, 0.8010427
6: -0.2551171, 0.3436220, -0.2387033, 0.3330199, -0.5881370, 0.5823252
7: -0.3414433, 0.3530792, -0.3240428, 0.3335895, -0.6750328, 0.6771220
8: -0.3294307, 0.4279434, -0.3133390, 0.4063403, -0.7357711, 0.7412824
9: -0.3231213, 0.4156990, -0.3043399, 0.3918386, -0.7149599, 0.7200390

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7823158
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7880314
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4494660, 1.0594878, 0.3686733, 1.0706941, -0.6212281, 0.6908145
1: -0.2563724, 0.2507969, -0.3035026, 0.2958409, -0.5522133, 0.5542995
2: -0.1749407, 0.3422925, -0.2166118, 0.3957256, -0.5706663, 0.5589042
3: -0.1866973, 0.2549886, -0.2209829, 0.2973216, -0.4840189, 0.4759715
4: -0.2715075, 0.2296257, -0.3151055, 0.2788742, -0.5503818, 0.5447312
5: -0.2905419, 0.3807709, -0.3392833, 0.4439522, -0.7344941, 0.7200541
6: -0.2013357, 0.2797443, -0.2425120, 0.3286508, -0.5299865, 0.5222563
7: -0.2833583, 0.2859068, -0.3278295, 0.3373355, -0.6206939, 0.6137363
8: -0.2692236, 0.3489462, -0.3153197, 0.4094285, -0.6786520, 0.6642660
9: -0.2605525, 0.3358182, -0.3084567, 0.3969768, -0.6575292, 0.6442749

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7815235
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7786506
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.3364010, 1.0748954, -0.7309546, 0.7377238
1: -0.3179300, 0.3096298, -0.3213227, 0.3128231, -0.6307531, 0.6309525
2: -0.2293682, 0.4120828, -0.2324356, 0.4158757, -0.6452439, 0.6445185
3: -0.2314784, 0.3102807, -0.2342138, 0.3133531, -0.5448315, 0.5444945
4: -0.3284518, 0.2939504, -0.3316072, 0.2974870, -0.6259388, 0.6255575
5: -0.3542041, 0.4632936, -0.3581833, 0.4681399, -0.8223439, 0.8214769
6: -0.2551171, 0.3436220, -0.2579479, 0.3488463, -0.6039633, 0.6015698
7: -0.3414433, 0.3530792, -0.3445697, 0.3568338, -0.6982771, 0.6976489
8: -0.3294307, 0.4279434, -0.3330806, 0.4326400, -0.7620707, 0.7610241
9: -0.3231213, 0.4156990, -0.3264823, 0.4200152, -0.7431364, 0.7421813

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7820158
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7874407
time: 1.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.4185251, 1.0637795, 0.3717514, 1.0702673, -0.6517422, 0.6920281
1: -0.2744217, 0.2680472, -0.3017069, 0.2941247, -0.5685464, 0.5697541
2: -0.1908993, 0.3627554, -0.2150241, 0.3936897, -0.5845889, 0.5777795
3: -0.1998275, 0.2712008, -0.2196766, 0.2957086, -0.4955361, 0.4908774
4: -0.2882042, 0.2484862, -0.3134444, 0.2769979, -0.5652021, 0.5619307
5: -0.3092084, 0.4049672, -0.3374265, 0.4415451, -0.7507535, 0.7423937
6: -0.2171049, 0.2984740, -0.2409432, 0.3267875, -0.5438924, 0.5394171
7: -0.3003893, 0.3056024, -0.3261352, 0.3353762, -0.6357656, 0.6317376
8: -0.2868768, 0.3721091, -0.3135634, 0.4071241, -0.6940009, 0.6856725
9: -0.2788982, 0.3592398, -0.3066315, 0.3946468, -0.6735450, 0.6658713

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7815083
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7786506
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.3116328, 1.0789609, 0.3717514, 1.0702673, -0.7586346, 0.7072095
1: -0.3374979, 0.3281388, -0.3017069, 0.2941247, -0.6316226, 0.6298457
2: -0.2469946, 0.4338901, -0.2150241, 0.3936897, -0.6406842, 0.6489142
3: -0.2453206, 0.3272926, -0.2196766, 0.2957086, -0.5410292, 0.5469692
4: -0.3469042, 0.3156349, -0.3134444, 0.2769979, -0.6239021, 0.6290793
5: -0.3745051, 0.4885432, -0.3374265, 0.4415451, -0.8160502, 0.8259697
6: -0.2715800, 0.3644212, -0.2409432, 0.3267875, -0.5983675, 0.6053644
7: -0.3592793, 0.3743900, -0.3261352, 0.3353762, -0.6946555, 0.7005253
8: -0.3498084, 0.4527032, -0.3135634, 0.4071241, -0.7569325, 0.7662666
9: -0.3431737, 0.4399822, -0.3066315, 0.3946468, -0.7378206, 0.7466137

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7819102
time: 1.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7786506
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.4494660, 1.0594878, 0.2711614, 1.0852319, -0.6357659, 0.7883264
1: -0.2563724, 0.2507969, -0.3618644, 0.3543747, -0.6107471, 0.6126614
2: -0.1749407, 0.3422925, -0.2712142, 0.4628203, -0.6377610, 0.6135067
3: -0.1866973, 0.2549886, -0.2633085, 0.3477496, -0.5344469, 0.5182971
4: -0.2715075, 0.2296257, -0.3754696, 0.3457245, -0.6172320, 0.6050953
5: -0.2905419, 0.3807709, -0.4020914, 0.5193099, -0.8098519, 0.7828623
6: -0.2013357, 0.2797443, -0.2916344, 0.3956037, -0.5969394, 0.5713787
7: -0.2833583, 0.2859068, -0.3814593, 0.4019997, -0.6853580, 0.6673660
8: -0.2692236, 0.3489462, -0.3804902, 0.4849254, -0.7541490, 0.7294365
9: -0.2605525, 0.3358182, -0.3708000, 0.4698921, -0.7304446, 0.7066182

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7805623
time: 1.42 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7771950
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.2275838, 1.0902638, -0.7463231, 0.8465410
1: -0.3179300, 0.3096298, -0.3823259, 0.3769295, -0.6948595, 0.6919557
2: -0.2293682, 0.4120828, -0.2916646, 0.4866098, -0.7159780, 0.7037474
3: -0.2314784, 0.3102807, -0.2795692, 0.3658088, -0.5972872, 0.5898499
4: -0.3284518, 0.2939504, -0.3991448, 0.3798461, -0.7082978, 0.6930952
5: -0.3542041, 0.4632936, -0.4274701, 0.5471249, -0.9013289, 0.8907638
6: -0.2551171, 0.3436220, -0.3115780, 0.4312532, -0.6863703, 0.6552000
7: -0.3414433, 0.3530792, -0.4017522, 0.4251831, -0.7666264, 0.7548314
8: -0.3294307, 0.4279434, -0.4113429, 0.5137860, -0.8432167, 0.8392863
9: -0.3231213, 0.4156990, -0.3945765, 0.4944862, -0.8176075, 0.8102756

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7808319
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7862869
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4494660, 1.0594878, 0.2322812, 1.0912958, -0.6418298, 0.8272066
1: -0.2563724, 0.2507969, -0.3855472, 0.3812227, -0.6375951, 0.6363441
2: -0.1749407, 0.3422925, -0.2943870, 0.4907170, -0.6656577, 0.6366795
3: -0.1866973, 0.2549886, -0.2804350, 0.3685404, -0.5552377, 0.5354236
4: -0.2715075, 0.2296257, -0.4030806, 0.3862872, -0.6577947, 0.6327063
5: -0.2905419, 0.3807709, -0.4288222, 0.5495177, -0.8400596, 0.8095932
6: -0.2013357, 0.2797443, -0.3157437, 0.4252924, -0.6266281, 0.5954880
7: -0.2833583, 0.2859068, -0.4056272, 0.4282870, -0.7116453, 0.6915340
8: -0.2692236, 0.3489462, -0.4151788, 0.5161407, -0.7853643, 0.7641250
9: -0.2605525, 0.3358182, -0.3988589, 0.4987577, -0.7593101, 0.7346771

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7800935
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7777442
time: 1.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.1923790, 1.0966504, -0.7527096, 0.8817458
1: -0.3179300, 0.3096298, -0.4111070, 0.4073653, -0.7252953, 0.7207367
2: -0.2293682, 0.4120828, -0.3153429, 0.5159366, -0.7453047, 0.7274257
3: -0.2314784, 0.3102807, -0.2959042, 0.3956319, -0.6271103, 0.6061848
4: -0.3284518, 0.2939504, -0.4293916, 0.4228130, -0.7512647, 0.7233420
5: -0.3542041, 0.4632936, -0.4572200, 0.5774447, -0.9316487, 0.9205136
6: -0.2551171, 0.3436220, -0.3375357, 0.4563557, -0.7114727, 0.6811576
7: -0.3414433, 0.3530792, -0.4280876, 0.4534785, -0.7949219, 0.7811668
8: -0.3294307, 0.4279434, -0.4476635, 0.5450330, -0.8744637, 0.8756069
9: -0.3231213, 0.4156990, -0.4244551, 0.5251222, -0.8482435, 0.8401542

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7808752
time: 2.27 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7860014
time: 1.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.4185251, 1.0637795, 0.2386133, 1.0902946, -0.6717695, 0.8251661
1: -0.2744217, 0.2680472, -0.3815904, 0.3765238, -0.6509455, 0.6496376
2: -0.1908993, 0.3627554, -0.2906587, 0.4860782, -0.6769775, 0.6534141
3: -0.1998275, 0.2712008, -0.2777127, 0.3650096, -0.5648371, 0.5489135
4: -0.2882042, 0.2484862, -0.3983926, 0.3795892, -0.6677934, 0.6468788
5: -0.3092084, 0.4049672, -0.4243879, 0.5445215, -0.8537298, 0.8293551
6: -0.2171049, 0.2984740, -0.3117016, 0.4205734, -0.6376783, 0.6101755
7: -0.3003893, 0.3056024, -0.4014842, 0.4241085, -0.7244979, 0.7070866
8: -0.2868768, 0.3721091, -0.4093528, 0.5109652, -0.7978420, 0.7814619
9: -0.2788982, 0.3592398, -0.3943411, 0.4940148, -0.7729130, 0.7535809

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7800816
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7777442
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.3116328, 1.0789609, 0.2386133, 1.0902946, -0.7786618, 0.8403475
1: -0.3374979, 0.3281388, -0.3815904, 0.3765238, -0.7140217, 0.7097293
2: -0.2469946, 0.4338901, -0.2906587, 0.4860782, -0.7330728, 0.7245488
3: -0.2453206, 0.3272926, -0.2777127, 0.3650096, -0.6103302, 0.6050053
4: -0.3469042, 0.3156349, -0.3983926, 0.3795892, -0.7264934, 0.7140275
5: -0.3745051, 0.4885432, -0.4243879, 0.5445215, -0.9190265, 0.9129311
6: -0.2715800, 0.3644212, -0.3117016, 0.4205734, -0.6921533, 0.6761228
7: -0.3592793, 0.3743900, -0.4014842, 0.4241085, -0.7833878, 0.7758743
8: -0.3498084, 0.4527032, -0.4093528, 0.5109652, -0.8607736, 0.8620560
9: -0.3431737, 0.4399822, -0.3943411, 0.4940148, -0.8371886, 0.8343233

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7802552
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7777442
time: 1.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.76 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7823295
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7782580
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7823158
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7880314
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7816753
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7786506
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7820158
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7874407
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7816601
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7786506
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7819102
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7786506
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7805713
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7771950
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7808319
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7862869
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7800982
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7722676, upper bound: 0.7777442
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7808752
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7880314, upper bound: 0.7860014
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7800854
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7777442
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7802552
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7725707, upper bound: 0.7777442
NS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7821149
NS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7782580
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7823158
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7880314
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7815235
NS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7786506
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7820158
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7874407
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7815083
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7786506
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7819102
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7786506
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7805623
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7771950
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7808319
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7862869
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7800935
NS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7711996, upper bound: 0.7777442
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7808752
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7862869, upper bound: 0.7860014
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7800816
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7777442
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7802552
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -0.7718369, upper bound: 0.7777442

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.5738506, 1.0415723, -0.5693548, 0.4824815
1: -0.2431005, 0.2381126, -0.1839559, 0.1786978, -0.4217983, 0.4220685
2: -0.1632061, 0.3272455, -0.1145039, 0.2592611, -0.4224672, 0.4417494
3: -0.1770424, 0.2430676, -0.1317621, 0.1913594, -0.3684018, 0.3748298
4: -0.2592304, 0.2157572, -0.2027689, 0.1581057, -0.4173361, 0.4185261
5: -0.2768164, 0.3629788, -0.2106911, 0.2920812, -0.5688976, 0.5736700
6: -0.1897404, 0.2659722, -0.1434024, 0.1996230, -0.3893634, 0.4093746
7: -0.2708351, 0.2714246, -0.2162840, 0.2041539, -0.4749891, 0.4877086
8: -0.2562429, 0.3319142, -0.1964648, 0.2498609, -0.5061038, 0.5283791
9: -0.2470626, 0.3185959, -0.1855037, 0.2474656, -0.4945282, 0.5040996

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7823295, upper bound: 0.7722676
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7722676
time: 1.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.4722175, 1.0563321, -0.5841146, 0.5841146
1: -0.2431005, 0.2381126, -0.2431005, 0.2381126, -0.4812131, 0.4812131
2: -0.1632061, 0.3272455, -0.1632061, 0.3272455, -0.4904516, 0.4904516
3: -0.1770424, 0.2430676, -0.1770424, 0.2430676, -0.4201100, 0.4201100
4: -0.2592304, 0.2157572, -0.2592304, 0.2157572, -0.4749876, 0.4749876
5: -0.2768164, 0.3629788, -0.2768164, 0.3629788, -0.6397953, 0.6397953
6: -0.1897404, 0.2659722, -0.1897404, 0.2659722, -0.4557126, 0.4557126
7: -0.2708351, 0.2714246, -0.2708351, 0.2714246, -0.5422597, 0.5422597
8: -0.2562429, 0.3319142, -0.2562429, 0.3319142, -0.5881571, 0.5881571
9: -0.2470626, 0.3185959, -0.2470626, 0.3185959, -0.5656585, 0.5656585

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7823295, upper bound: 0.7782580
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7782580
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.5443780, 1.0459632, -0.5737457, 0.5119541
1: -0.2431005, 0.2381126, -0.2010765, 0.1963980, -0.4394985, 0.4391891
2: -0.1632061, 0.3272455, -0.1278276, 0.2788188, -0.4420249, 0.4550731
3: -0.1770424, 0.2430676, -0.1453555, 0.2056128, -0.3826552, 0.3884231
4: -0.2592304, 0.2157572, -0.2192978, 0.1738993, -0.4331298, 0.4350550
5: -0.2768164, 0.3629788, -0.2309013, 0.3106825, -0.5874989, 0.5938802
6: -0.1897404, 0.2659722, -0.1553648, 0.2199016, -0.4096420, 0.4213370
7: -0.2708351, 0.2714246, -0.2318041, 0.2240368, -0.4948719, 0.5032287
8: -0.2562429, 0.3319142, -0.2140101, 0.2749390, -0.5311819, 0.5459243
9: -0.2470626, 0.3185959, -0.2033888, 0.2668447, -0.5139073, 0.5219846

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7820599, upper bound: 0.7725707
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7725707
time: 1.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.4362575, 1.0613201, -0.5891026, 0.6200746
1: -0.2431005, 0.2381126, -0.2640777, 0.2581610, -0.5012615, 0.5021902
2: -0.1632061, 0.3272455, -0.1817534, 0.3510281, -0.5142342, 0.5089989
3: -0.1770424, 0.2430676, -0.1923025, 0.2619095, -0.4389519, 0.4353701
4: -0.2592304, 0.2157572, -0.2786354, 0.2376771, -0.4969076, 0.4943925
5: -0.2768164, 0.3629788, -0.2985106, 0.3911000, -0.6679165, 0.6614895
6: -0.1897404, 0.2659722, -0.2080675, 0.2877399, -0.4774803, 0.4740397
7: -0.2708351, 0.2714246, -0.2906288, 0.2943149, -0.5651500, 0.5620533
8: -0.2562429, 0.3319142, -0.2767597, 0.3588341, -0.6150770, 0.6086739
9: -0.2470626, 0.3185959, -0.2683841, 0.3458169, -0.5928795, 0.5869800

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7820599, upper bound: 0.7786506
time: 1.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7786506
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.4494660, 1.0594878, -0.5872703, 0.6068661
1: -0.2431005, 0.2381126, -0.2563724, 0.2507969, -0.4938974, 0.4944850
2: -0.1632061, 0.3272455, -0.1749407, 0.3422925, -0.5054986, 0.5021862
3: -0.1770424, 0.2430676, -0.1866973, 0.2549886, -0.4320310, 0.4297649
4: -0.2592304, 0.2157572, -0.2715075, 0.2296257, -0.4888561, 0.4872647
5: -0.2768164, 0.3629788, -0.2905419, 0.3807709, -0.6575873, 0.6535208
6: -0.1897404, 0.2659722, -0.2013357, 0.2797443, -0.4694847, 0.4673079
7: -0.2708351, 0.2714246, -0.2833583, 0.2859068, -0.5567420, 0.5547829
8: -0.2562429, 0.3319142, -0.2692236, 0.3489462, -0.6051891, 0.6011378
9: -0.2470626, 0.3185959, -0.2605525, 0.3358182, -0.5828808, 0.5791484

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7821149, upper bound: 0.7711996
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7711996
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.3439408, 1.0741248, -0.6019073, 0.7123914
1: -0.2431005, 0.2381126, -0.3179300, 0.3096298, -0.5527303, 0.5560426
2: -0.1632061, 0.3272455, -0.2293682, 0.4120828, -0.5752889, 0.5566137
3: -0.1770424, 0.2430676, -0.2314784, 0.3102807, -0.4873230, 0.4745460
4: -0.2592304, 0.2157572, -0.3284518, 0.2939504, -0.5531808, 0.5442089
5: -0.2768164, 0.3629788, -0.3542041, 0.4632936, -0.7401100, 0.7171829
6: -0.1897404, 0.2659722, -0.2551171, 0.3436220, -0.5333624, 0.5210893
7: -0.2708351, 0.2714246, -0.3414433, 0.3530792, -0.6239144, 0.6128678
8: -0.2562429, 0.3319142, -0.3294307, 0.4279434, -0.6841863, 0.6613449
9: -0.2470626, 0.3185959, -0.3231213, 0.4156990, -0.6627616, 0.6417172

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7821149, upper bound: 0.7771950
time: 1.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7771950
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.4185251, 1.0637795, -0.5915620, 0.6378070
1: -0.2431005, 0.2381126, -0.2744217, 0.2680472, -0.5111477, 0.5125343
2: -0.1632061, 0.3272455, -0.1908993, 0.3627554, -0.5259615, 0.5181447
3: -0.1770424, 0.2430676, -0.1998275, 0.2712008, -0.4482433, 0.4428951
4: -0.2592304, 0.2157572, -0.2882042, 0.2484862, -0.5077167, 0.5039614
5: -0.2768164, 0.3629788, -0.3092084, 0.4049672, -0.6817837, 0.6721872
6: -0.1897404, 0.2659722, -0.2171049, 0.2984740, -0.4882143, 0.4830771
7: -0.2708351, 0.2714246, -0.3003893, 0.3056024, -0.5764375, 0.5718139
8: -0.2562429, 0.3319142, -0.2868768, 0.3721091, -0.6283519, 0.6187910
9: -0.2470626, 0.3185959, -0.2788982, 0.3592398, -0.6063024, 0.5974941

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7818287, upper bound: 0.7718369
time: 2.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7718369
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.4722175, 1.0563321, 0.3116328, 1.0789609, -0.6067434, 0.7446994
1: -0.2431005, 0.2381126, -0.3374979, 0.3281388, -0.5712394, 0.5756105
2: -0.1632061, 0.3272455, -0.2469946, 0.4338901, -0.5970962, 0.5742401
3: -0.1770424, 0.2430676, -0.2453206, 0.3272926, -0.5043350, 0.4883882
4: -0.2592304, 0.2157572, -0.3469042, 0.3156349, -0.5748653, 0.5626614
5: -0.2768164, 0.3629788, -0.3745051, 0.4885432, -0.7653596, 0.7374839
6: -0.1897404, 0.2659722, -0.2715800, 0.3644212, -0.5541615, 0.5375521
7: -0.2708351, 0.2714246, -0.3592793, 0.3743900, -0.6452252, 0.6307039
8: -0.2562429, 0.3319142, -0.3498084, 0.4527032, -0.7089461, 0.6817226
9: -0.2470626, 0.3185959, -0.3431737, 0.4399822, -0.6870449, 0.6617696

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7818287, upper bound: 0.7718369
time: 1.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7777442
time: 2.27 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.5738506, 1.0415723, -0.6976316, 0.5002742
1: -0.3179300, 0.3096298, -0.1839559, 0.1786978, -0.4966278, 0.4935857
2: -0.2293682, 0.4120828, -0.1145039, 0.2592611, -0.4886293, 0.5265867
3: -0.2314784, 0.3102807, -0.1317621, 0.1913594, -0.4228378, 0.4420428
4: -0.3284518, 0.2939504, -0.2027689, 0.1581057, -0.4865574, 0.4967193
5: -0.3542041, 0.4632936, -0.2106911, 0.2920812, -0.6462852, 0.6739848
6: -0.2551171, 0.3436220, -0.1434024, 0.1996230, -0.4547401, 0.4870244
7: -0.3414433, 0.3530792, -0.2162840, 0.2041539, -0.5455973, 0.5693632
8: -0.3294307, 0.4279434, -0.1964648, 0.2498609, -0.5792916, 0.6244082
9: -0.3231213, 0.4156990, -0.1855037, 0.2474656, -0.5705869, 0.6012027

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7805713, upper bound: 0.7722676
time: 1.90 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7722676
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.4722175, 1.0563321, -0.7123914, 0.6019073
1: -0.3179300, 0.3096298, -0.2431005, 0.2381126, -0.5560426, 0.5527303
2: -0.2293682, 0.4120828, -0.1632061, 0.3272455, -0.5566137, 0.5752889
3: -0.2314784, 0.3102807, -0.1770424, 0.2430676, -0.4745460, 0.4873230
4: -0.3284518, 0.2939504, -0.2592304, 0.2157572, -0.5442089, 0.5531808
5: -0.3542041, 0.4632936, -0.2768164, 0.3629788, -0.7171829, 0.7401100
6: -0.2551171, 0.3436220, -0.1897404, 0.2659722, -0.5210893, 0.5333624
7: -0.3414433, 0.3530792, -0.2708351, 0.2714246, -0.6128678, 0.6239144
8: -0.3294307, 0.4279434, -0.2562429, 0.3319142, -0.6613449, 0.6841863
9: -0.3231213, 0.4156990, -0.2470626, 0.3185959, -0.6417172, 0.6627616

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7805713, upper bound: 0.7782580
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7782580
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.5443780, 1.0459632, -0.7020224, 0.5297468
1: -0.3179300, 0.3096298, -0.2010765, 0.1963980, -0.5143281, 0.5107063
2: -0.2293682, 0.4120828, -0.1278276, 0.2788188, -0.5081871, 0.5399104
3: -0.2314784, 0.3102807, -0.1453555, 0.2056128, -0.4370912, 0.4556361
4: -0.3284518, 0.2939504, -0.2192978, 0.1738993, -0.5023510, 0.5132482
5: -0.3542041, 0.4632936, -0.2309013, 0.3106825, -0.6648866, 0.6941949
6: -0.2551171, 0.3436220, -0.1553648, 0.2199016, -0.4750186, 0.4989868
7: -0.3414433, 0.3530792, -0.2318041, 0.2240368, -0.5654801, 0.5848833
8: -0.3294307, 0.4279434, -0.2140101, 0.2749390, -0.6043697, 0.6419535
9: -0.3231213, 0.4156990, -0.2033888, 0.2668447, -0.5899659, 0.6190878

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7803281, upper bound: 0.7725707
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7725707
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.4362575, 1.0613201, -0.7173793, 0.6378673
1: -0.3179300, 0.3096298, -0.2640777, 0.2581610, -0.5760911, 0.5737075
2: -0.2293682, 0.4120828, -0.1817534, 0.3510281, -0.5803963, 0.5938362
3: -0.2314784, 0.3102807, -0.1923025, 0.2619095, -0.4933879, 0.5025831
4: -0.3284518, 0.2939504, -0.2786354, 0.2376771, -0.5661289, 0.5725858
5: -0.3542041, 0.4632936, -0.2985106, 0.3911000, -0.7453041, 0.7618042
6: -0.2551171, 0.3436220, -0.2080675, 0.2877399, -0.5428569, 0.5516894
7: -0.3414433, 0.3530792, -0.2906288, 0.2943149, -0.6357582, 0.6437080
8: -0.3294307, 0.4279434, -0.2767597, 0.3588341, -0.6882648, 0.7047031
9: -0.3231213, 0.4156990, -0.2683841, 0.3458169, -0.6689382, 0.6840831

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7803281, upper bound: 0.7786506
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7786506
time: 1.95 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.4494660, 1.0594878, -0.7155471, 0.6246588
1: -0.3179300, 0.3096298, -0.2563724, 0.2507969, -0.5687270, 0.5660021
2: -0.2293682, 0.4120828, -0.1749407, 0.3422925, -0.5716606, 0.5870236
3: -0.2314784, 0.3102807, -0.1866973, 0.2549886, -0.4864670, 0.4969779
4: -0.3284518, 0.2939504, -0.2715075, 0.2296257, -0.5580775, 0.5654579
5: -0.3542041, 0.4632936, -0.2905419, 0.3807709, -0.7349750, 0.7538356
6: -0.2551171, 0.3436220, -0.2013357, 0.2797443, -0.5348613, 0.5449577
7: -0.3414433, 0.3530792, -0.2833583, 0.2859068, -0.6273501, 0.6364375
8: -0.3294307, 0.4279434, -0.2692236, 0.3489462, -0.6783769, 0.6971670
9: -0.3231213, 0.4156990, -0.2605525, 0.3358182, -0.6589395, 0.6762515

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7805623, upper bound: 0.7711996
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7711996
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.3439408, 1.0741248, -0.7301841, 0.7301841
1: -0.3179300, 0.3096298, -0.3179300, 0.3096298, -0.6275598, 0.6275598
2: -0.2293682, 0.4120828, -0.2293682, 0.4120828, -0.6414510, 0.6414510
3: -0.2314784, 0.3102807, -0.2314784, 0.3102807, -0.5417590, 0.5417590
4: -0.3284518, 0.2939504, -0.3284518, 0.2939504, -0.6224022, 0.6224022
5: -0.3542041, 0.4632936, -0.3542041, 0.4632936, -0.8174977, 0.8174977
6: -0.2551171, 0.3436220, -0.2551171, 0.3436220, -0.5987390, 0.5987390
7: -0.3414433, 0.3530792, -0.3414433, 0.3530792, -0.6945225, 0.6945225
8: -0.3294307, 0.4279434, -0.3294307, 0.4279434, -0.7573741, 0.7573741
9: -0.3231213, 0.4156990, -0.3231213, 0.4156990, -0.7388203, 0.7388203

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7805623, upper bound: 0.7771950
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7771950
time: 1.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.4185251, 1.0637795, -0.7198387, 0.6555997
1: -0.3179300, 0.3096298, -0.2744217, 0.2680472, -0.5859773, 0.5840515
2: -0.2293682, 0.4120828, -0.1908993, 0.3627554, -0.5921236, 0.6029820
3: -0.2314784, 0.3102807, -0.1998275, 0.2712008, -0.5026792, 0.5101081
4: -0.3284518, 0.2939504, -0.2882042, 0.2484862, -0.5769380, 0.5821546
5: -0.3542041, 0.4632936, -0.3092084, 0.4049672, -0.7591712, 0.7725020
6: -0.2551171, 0.3436220, -0.2171049, 0.2984740, -0.5535910, 0.5607269
7: -0.3414433, 0.3530792, -0.3003893, 0.3056024, -0.6470457, 0.6534685
8: -0.3294307, 0.4279434, -0.2868768, 0.3721091, -0.7015398, 0.7148202
9: -0.3231213, 0.4156990, -0.2788982, 0.3592398, -0.6823611, 0.6945972

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7803280, upper bound: 0.7718369
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7718369
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3439408, 1.0741248, 0.3116328, 1.0789609, -0.7350202, 0.7624921
1: -0.3179300, 0.3096298, -0.3374979, 0.3281388, -0.6460689, 0.6471277
2: -0.2293682, 0.4120828, -0.2469946, 0.4338901, -0.6632583, 0.6590774
3: -0.2314784, 0.3102807, -0.2453206, 0.3272926, -0.5587710, 0.5556012
4: -0.3284518, 0.2939504, -0.3469042, 0.3156349, -0.6440867, 0.6408546
5: -0.3542041, 0.4632936, -0.3745051, 0.4885432, -0.8427473, 0.8377987
6: -0.2551171, 0.3436220, -0.2715800, 0.3644212, -0.6195382, 0.6152020
7: -0.3414433, 0.3530792, -0.3592793, 0.3743900, -0.7158333, 0.7123585
8: -0.3294307, 0.4279434, -0.3498084, 0.4527032, -0.7821339, 0.7777518
9: -0.3231213, 0.4156990, -0.3431737, 0.4399822, -0.7631035, 0.7588727

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 122

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 171

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7803280, upper bound: 0.7777442
time: 1.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7777442
time: 1.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.92 seconds
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7823295, upper bound: 0.7722676
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7722676
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7823295, upper bound: 0.7782580
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7782580
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7820599, upper bound: 0.7725707
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7725707
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7820599, upper bound: 0.7786506
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7786506
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7821149, upper bound: 0.7711996
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7711996
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7821149, upper bound: 0.7771950
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7771950
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7818287, upper bound: 0.7718369
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7718369
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7818287, upper bound: 0.7718369
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7782580, upper bound: 0.7777442
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7805713, upper bound: 0.7722676
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7722676
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7805713, upper bound: 0.7782580
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7782580
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7803281, upper bound: 0.7725707
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7725707
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7803281, upper bound: 0.7786506
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7786506
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7805623, upper bound: 0.7711996
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7711996
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7805623, upper bound: 0.7771950
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7771950
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7803280, upper bound: 0.7718369
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7718369
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7803280, upper bound: 0.7777442
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.92
Output dim: 0, lower bound: -0.7771950, upper bound: 0.7777442

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.05 + 385.42 = 390.47 seconds
