## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.030823649999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0156055, 0.0176595, -0.0156055, 0.0176595, -0.0332650, 0.0332650)
1: (-0.0115796, -0.0001929, -0.0115796, -0.0001929, -0.0113866, 0.0113866)
2: (-0.0029904, 0.0292235, -0.0029904, 0.0292235, -0.0322139, 0.0322139)
3: (-0.0152654, 0.0159234, -0.0152654, 0.0159234, -0.0311888, 0.0311888)
4: (-0.0142183, 0.0128834, -0.0142183, 0.0128834, -0.0271017, 0.0271017)
5: (0.9782868, 1.0145735, 0.9782868, 1.0145735, -0.0362867, 0.0362867)
6: (-0.0145406, 0.0158245, -0.0145406, 0.0158245, -0.0303650, 0.0303650)
7: (-0.0295835, -0.0013468, -0.0295835, -0.0013468, -0.0282367, 0.0282367)
8: (-0.0124212, 0.0285642, -0.0124212, 0.0285642, -0.0409854, 0.0409854)
9: (-0.0108079, 0.0115272, -0.0108079, 0.0115272, -0.0223351, 0.0223351)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.85 + 2.22 = 4.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.18 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.53 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.53
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.53
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0090002, 0.0133130, -0.0147333, 0.0165688, -0.0255690, 0.0280462
1: -0.0101351, -0.0011694, -0.0114028, -0.0003409, -0.0097942, 0.0102334
2: -0.0018444, 0.0248405, -0.0026284, 0.0284740, -0.0303185, 0.0274689
3: -0.0119507, 0.0107289, -0.0148597, 0.0151162, -0.0270669, 0.0255886
4: -0.0118689, 0.0095278, -0.0139308, 0.0124343, -0.0243032, 0.0234586
5: 0.9874392, 1.0104873, 0.9806033, 1.0140734, -0.0266342, 0.0298840
6: -0.0100976, 0.0135746, -0.0138829, 0.0155491, -0.0256467, 0.0274575
7: -0.0274265, -0.0021527, -0.0293195, -0.0016014, -0.0258251, 0.0271668
8: -0.0095674, 0.0221032, -0.0120719, 0.0275894, -0.0371568, 0.0341751
9: -0.0087735, 0.0079918, -0.0105589, 0.0110792, -0.0198527, 0.0185508

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.48 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.42 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0140331, 0.0156932, -0.0152641, 0.0172326, -0.0312657, 0.0309573
1: -0.0112609, -0.0004597, -0.0115104, -0.0002509, -0.0110100, 0.0110507
2: -0.0025816, 0.0278724, -0.0028982, 0.0289302, -0.0315117, 0.0307706
3: -0.0145340, 0.0144683, -0.0151066, 0.0156075, -0.0301415, 0.0295749
4: -0.0137000, 0.0120739, -0.0141058, 0.0127076, -0.0264076, 0.0261797
5: 0.9824629, 1.0136719, 0.9791935, 1.0143777, -0.0319148, 0.0344784
6: -0.0133550, 0.0153281, -0.0142832, 0.0157167, -0.0290717, 0.0296113
7: -0.0291076, -0.0016343, -0.0294802, -0.0014116, -0.0276959, 0.0278459
8: -0.0117915, 0.0268068, -0.0122845, 0.0281827, -0.0399742, 0.0390913
9: -0.0103590, 0.0107195, -0.0107105, 0.0113518, -0.0217108, 0.0214300

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.57 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 1.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.07 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.07
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090002, 0.0133130, -0.0128565, 0.0148389, -0.0238391, 0.0261694
1: -0.0101351, -0.0011694, -0.0110224, -0.0006279, -0.0095072, 0.0098530
2: -0.0018444, 0.0248405, -0.0022613, 0.0270966, -0.0289411, 0.0271017
3: -0.0119507, 0.0107289, -0.0139867, 0.0135687, -0.0255193, 0.0247156
4: -0.0118689, 0.0095278, -0.0133120, 0.0115105, -0.0233794, 0.0228398
5: 0.9874392, 1.0104873, 0.9842666, 1.0129973, -0.0255581, 0.0262207
6: -0.0100976, 0.0135746, -0.0125936, 0.0149566, -0.0250541, 0.0261682
7: -0.0274265, -0.0021527, -0.0287514, -0.0018596, -0.0255669, 0.0265987
8: -0.0095674, 0.0221032, -0.0113203, 0.0256950, -0.0352625, 0.0334235
9: -0.0087735, 0.0079918, -0.0100231, 0.0101320, -0.0189055, 0.0180149

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.45 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0088757, 0.0127839, -0.0456196, 0.0551943, -0.0640699, 0.0584035
1: -0.0098275, -0.0013571, -0.0176635, 0.0048997, -0.0147272, 0.0163064
2: -0.0015853, 0.0240583, -0.0018146, 0.0550143, -0.0565996, 0.0258730
3: -0.0112449, 0.0097444, -0.0292265, 0.0436999, -0.0549448, 0.0389709
4: -0.0113686, 0.0088405, -0.0241140, 0.0283345, -0.0397031, 0.0329544
5: 0.9885388, 1.0096172, 0.8985686, 1.0317837, -0.0432449, 0.1110486
6: -0.0092323, 0.0130955, -0.0371721, 0.0253006, -0.0345329, 0.0502676
7: -0.0269671, -0.0023350, -0.0386689, 0.0230431, -0.0500103, 0.0363338
8: -0.0089597, 0.0208580, -0.0244410, 0.0621099, -0.0710697, 0.0452990
9: -0.0083403, 0.0072499, -0.0193765, 0.0269444, -0.0352847, 0.0266264

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.52 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.45 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0140331, 0.0156932, -0.0133849, 0.0150232, -0.0290563, 0.0290781
1: -0.0112609, -0.0004597, -0.0111295, -0.0005626, -0.0106983, 0.0106697
2: -0.0025816, 0.0278724, -0.0025301, 0.0273690, -0.0299506, 0.0304026
3: -0.0145340, 0.0144683, -0.0142325, 0.0139115, -0.0284455, 0.0287007
4: -0.0137000, 0.0120739, -0.0134862, 0.0117498, -0.0254498, 0.0255601
5: 0.9824629, 1.0136719, 0.9838836, 1.0133002, -0.0308372, 0.0297883
6: -0.0133550, 0.0153281, -0.0128948, 0.0151234, -0.0284784, 0.0282229
7: -0.0291076, -0.0016343, -0.0289113, -0.0016705, -0.0274371, 0.0272770
8: -0.0117915, 0.0268068, -0.0115319, 0.0261286, -0.0379202, 0.0383388
9: -0.0103590, 0.0107195, -0.0101739, 0.0103904, -0.0207494, 0.0208934

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.39 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.27 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0125026, 0.0147156, -0.0461979, 0.0559175, -0.0684201, 0.0609135
1: -0.0109506, -0.0006717, -0.0177808, 0.0049978, -0.0159484, 0.0171091
2: -0.0023180, 0.0269142, -0.0020735, 0.0555113, -0.0578292, 0.0289876
3: -0.0138220, 0.0133390, -0.0294955, 0.0442351, -0.0580572, 0.0428345
4: -0.0131953, 0.0113502, -0.0243046, 0.0286322, -0.0418275, 0.0356548
5: 0.9845230, 1.0127941, 0.8970325, 1.0321153, -0.0475923, 0.1157616
6: -0.0123917, 0.0148448, -0.0376082, 0.0254832, -0.0378749, 0.0524530
7: -0.0286443, -0.0018197, -0.0388440, 0.0235951, -0.0522394, 0.0370242
8: -0.0111786, 0.0254046, -0.0246726, 0.0627563, -0.0739349, 0.0500772
9: -0.0099221, 0.0099590, -0.0195416, 0.0272415, -0.0371635, 0.0295006

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 2.55 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 1.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.26 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.26
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0088254, 0.0126686, -0.0128565, 0.0148389, -0.0236643, 0.0255251
1: -0.0097605, -0.0013980, -0.0110224, -0.0006279, -0.0091326, 0.0096243
2: -0.0014807, 0.0238878, -0.0022613, 0.0270966, -0.0285774, 0.0261491
3: -0.0110910, 0.0095298, -0.0139867, 0.0135687, -0.0246597, 0.0235165
4: -0.0112595, 0.0086906, -0.0133120, 0.0115105, -0.0227700, 0.0220026
5: 0.9887787, 1.0094277, 0.9842666, 1.0129973, -0.0242186, 0.0251610
6: -0.0090437, 0.0129911, -0.0125936, 0.0149566, -0.0240003, 0.0255847
7: -0.0268670, -0.0024086, -0.0287514, -0.0018596, -0.0250074, 0.0263428
8: -0.0088273, 0.0205866, -0.0113203, 0.0256950, -0.0345223, 0.0319069
9: -0.0082459, 0.0070882, -0.0100231, 0.0101320, -0.0183779, 0.0171113

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0389911, 0.0239504, -0.0128565, 0.0148389, -0.0538301, 0.0368069
1: -0.0163199, 0.0026051, -0.0110224, -0.0006279, -0.0156920, 0.0136274
2: -0.0010463, 0.0405677, -0.0022613, 0.0270966, -0.0281429, 0.0428289
3: -0.0261432, 0.0305244, -0.0139867, 0.0135687, -0.0397119, 0.0445111
4: -0.0219286, 0.0233487, -0.0133120, 0.0115105, -0.0334391, 0.0366607
5: 0.9653244, 1.0279828, 0.9842666, 1.0129973, -0.0476729, 0.0437162
6: -0.0274964, 0.0232079, -0.0125936, 0.0149566, -0.0424530, 0.0358014
7: -0.0366624, 0.0054587, -0.0287514, -0.0018596, -0.0348029, 0.0342101
8: -0.0217865, 0.0471410, -0.0113203, 0.0256950, -0.0474815, 0.0584614
9: -0.0174842, 0.0229104, -0.0100231, 0.0101320, -0.0276162, 0.0329335

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309960, upper bound: 0.0308222
time: 2.03 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309784, upper bound: 0.0305470
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0088757, 0.0127839, -0.0390712, 0.0470051, -0.0558807, 0.0518551
1: -0.0098275, -0.0013571, -0.0163362, 0.0037886, -0.0136161, 0.0149791
2: -0.0015853, 0.0240583, -0.0010462, 0.0493873, -0.0509726, 0.0251046
3: -0.0112449, 0.0097444, -0.0261805, 0.0376397, -0.0488846, 0.0359249
4: -0.0113686, 0.0088405, -0.0219549, 0.0249634, -0.0363320, 0.0307954
5: 0.9885388, 1.0096172, 0.9159613, 1.0280287, -0.0394899, 0.0936559
6: -0.0092323, 0.0130955, -0.0322344, 0.0232331, -0.0324655, 0.0453300
7: -0.0269671, -0.0023350, -0.0366867, 0.0167928, -0.0437599, 0.0343516
8: -0.0089597, 0.0208580, -0.0218186, 0.0547910, -0.0637508, 0.0426766
9: -0.0083403, 0.0072499, -0.0175070, 0.0235807, -0.0319210, 0.0247570

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307841, upper bound: 0.0305470
time: 1.77 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305470
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0088757, 0.0127839, -0.0448981, 0.0542921, -0.0631678, 0.0576821
1: -0.0098275, -0.0013571, -0.0175173, 0.0047773, -0.0146048, 0.0161602
2: -0.0015853, 0.0240583, -0.0017526, 0.0543944, -0.0559797, 0.0258109
3: -0.0112449, 0.0097444, -0.0288909, 0.0430322, -0.0542771, 0.0386353
4: -0.0113686, 0.0088405, -0.0238761, 0.0279631, -0.0393317, 0.0327166
5: 0.9885388, 1.0096172, 0.9004847, 1.0313700, -0.0428312, 0.1091325
6: -0.0092323, 0.0130955, -0.0366282, 0.0250728, -0.0343052, 0.0497237
7: -0.0269671, -0.0023350, -0.0384505, 0.0223546, -0.0493217, 0.0361155
8: -0.0089597, 0.0208580, -0.0241521, 0.0613036, -0.0702633, 0.0450101
9: -0.0083403, 0.0072499, -0.0191705, 0.0265738, -0.0349141, 0.0264205

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307841, upper bound: 0.0305470
time: 1.90 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305470
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0140331, 0.0156932, -0.0088254, 0.0126686, -0.0267017, 0.0245186
1: -0.0112609, -0.0004597, -0.0097605, -0.0013980, -0.0098628, 0.0093007
2: -0.0025816, 0.0278724, -0.0014807, 0.0238878, -0.0264694, 0.0293531
3: -0.0145340, 0.0144683, -0.0110910, 0.0095298, -0.0240638, 0.0255593
4: -0.0137000, 0.0120739, -0.0112595, 0.0086906, -0.0223906, 0.0233334
5: 0.9824629, 1.0136719, 0.9887787, 1.0094277, -0.0269647, 0.0248932
6: -0.0133550, 0.0153281, -0.0090437, 0.0129911, -0.0263461, 0.0243717
7: -0.0291076, -0.0016343, -0.0268670, -0.0024086, -0.0266990, 0.0252327
8: -0.0117915, 0.0268068, -0.0088273, 0.0205866, -0.0323781, 0.0356341
9: -0.0103590, 0.0107195, -0.0082459, 0.0070882, -0.0174472, 0.0189654

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.26 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0140331, 0.0156932, -0.0121589, 0.0145958, -0.0286289, 0.0278521
1: -0.0112609, -0.0004597, -0.0108810, -0.0007142, -0.0105466, 0.0104212
2: -0.0025816, 0.0278724, -0.0022171, 0.0267371, -0.0293186, 0.0300895
3: -0.0145340, 0.0144683, -0.0136622, 0.0131161, -0.0276501, 0.0281305
4: -0.0137000, 0.0120739, -0.0130820, 0.0111945, -0.0248945, 0.0251559
5: 0.9824629, 1.0136719, 0.9847723, 1.0125971, -0.0301341, 0.0288996
6: -0.0133550, 0.0153281, -0.0121958, 0.0147363, -0.0280913, 0.0275238
7: -0.0291076, -0.0016343, -0.0285403, -0.0018907, -0.0272169, 0.0269059
8: -0.0117915, 0.0268068, -0.0110410, 0.0251226, -0.0369142, 0.0378478
9: -0.0103590, 0.0107195, -0.0098239, 0.0097909, -0.0201500, 0.0205434

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.29 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0125026, 0.0147156, -0.0390712, 0.0470051, -0.0595076, 0.0537868
1: -0.0109506, -0.0006717, -0.0163362, 0.0037886, -0.0147392, 0.0156645
2: -0.0023180, 0.0269142, -0.0010462, 0.0493873, -0.0517053, 0.0279604
3: -0.0138220, 0.0133390, -0.0261805, 0.0376397, -0.0514617, 0.0395195
4: -0.0131953, 0.0113502, -0.0219549, 0.0249634, -0.0381587, 0.0333051
5: 0.9845230, 1.0127941, 0.9159613, 1.0280287, -0.0435057, 0.0968328
6: -0.0123917, 0.0148448, -0.0322344, 0.0232331, -0.0356249, 0.0470792
7: -0.0286443, -0.0018197, -0.0366867, 0.0167928, -0.0454371, 0.0348670
8: -0.0111786, 0.0254046, -0.0218186, 0.0547910, -0.0659696, 0.0472231
9: -0.0099221, 0.0099590, -0.0175070, 0.0235807, -0.0335028, 0.0274660

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307859, upper bound: 0.0305134
time: 1.87 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305134
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0125026, 0.0147156, -0.0448981, 0.0542921, -0.0667947, 0.0596137
1: -0.0109506, -0.0006717, -0.0175173, 0.0047773, -0.0157279, 0.0168456
2: -0.0023180, 0.0269142, -0.0017526, 0.0543944, -0.0567124, 0.0286668
3: -0.0138220, 0.0133390, -0.0288909, 0.0430322, -0.0568543, 0.0422299
4: -0.0131953, 0.0113502, -0.0238761, 0.0279631, -0.0411584, 0.0352263
5: 0.9845230, 1.0127941, 0.9004847, 1.0313700, -0.0468470, 0.1123095
6: -0.0123917, 0.0148448, -0.0366282, 0.0250728, -0.0374646, 0.0514730
7: -0.0286443, -0.0018197, -0.0384505, 0.0223546, -0.0509988, 0.0366308
8: -0.0111786, 0.0254046, -0.0241521, 0.0613036, -0.0724822, 0.0495566
9: -0.0099221, 0.0099590, -0.0191705, 0.0265738, -0.0364959, 0.0291295

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307859, upper bound: 0.0305134
time: 1.99 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305134
time: 1.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.48 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0309960, upper bound: 0.0308222
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0309784, upper bound: 0.0305470
NS_A1_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0307841, upper bound: 0.0305470
NS_A1_B2_B1_B2, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305470
NS_A1_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0307841, upper bound: 0.0305470
NS_A1_B2_B2_B2, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305470
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0307859, upper bound: 0.0305134
NS_A2_B2_B1_B2, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305134
NS_A2_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0307859, upper bound: 0.0305134
NS_A2_B2_B2_B2, status: Status.VERIFIED, split count: 4, time: 5.48
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0305134

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0088254, 0.0126686, -0.0088254, 0.0126686, -0.0214940, 0.0214940
1: -0.0097605, -0.0013980, -0.0097605, -0.0013980, -0.0083625, 0.0083625
2: -0.0014807, 0.0238878, -0.0014807, 0.0238878, -0.0253686, 0.0253686
3: -0.0110910, 0.0095298, -0.0110910, 0.0095298, -0.0206208, 0.0206208
4: -0.0112595, 0.0086906, -0.0112595, 0.0086906, -0.0199502, 0.0199502
5: 0.9887787, 1.0094277, 0.9887787, 1.0094277, -0.0206490, 0.0206490
6: -0.0090437, 0.0129911, -0.0090437, 0.0129911, -0.0220348, 0.0220348
7: -0.0268670, -0.0024086, -0.0268670, -0.0024086, -0.0244584, 0.0244584
8: -0.0088273, 0.0205866, -0.0088273, 0.0205866, -0.0294139, 0.0294139
9: -0.0082459, 0.0070882, -0.0082459, 0.0070882, -0.0153341, 0.0153341

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0088254, 0.0126686, -0.0121589, 0.0145958, -0.0234211, 0.0248275
1: -0.0097605, -0.0013980, -0.0108810, -0.0007142, -0.0090463, 0.0094829
2: -0.0014807, 0.0238878, -0.0022171, 0.0267371, -0.0282178, 0.0261049
3: -0.0110910, 0.0095298, -0.0136622, 0.0131161, -0.0242071, 0.0231920
4: -0.0112595, 0.0086906, -0.0130820, 0.0111945, -0.0224541, 0.0217726
5: 0.9887787, 1.0094277, 0.9847723, 1.0125971, -0.0238184, 0.0246554
6: -0.0090437, 0.0129911, -0.0121958, 0.0147363, -0.0237800, 0.0251869
7: -0.0268670, -0.0024086, -0.0285403, -0.0018907, -0.0249763, 0.0261317
8: -0.0088273, 0.0205866, -0.0110410, 0.0251226, -0.0339499, 0.0316275
9: -0.0082459, 0.0070882, -0.0098239, 0.0097909, -0.0180368, 0.0169122

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0319152, 0.0214835, -0.0114656, 0.0143540, -0.0462692, 0.0329491
1: -0.0148856, 0.0017297, -0.0107404, -0.0008000, -0.0140856, 0.0124701
2: -0.0009942, 0.0369204, -0.0022502, 0.0263797, -0.0273739, 0.0391706
3: -0.0228518, 0.0259336, -0.0133397, 0.0126663, -0.0355181, 0.0392733
4: -0.0195956, 0.0201435, -0.0128534, 0.0108805, -0.0304761, 0.0329969
5: 0.9704528, 1.0239255, 0.9852747, 1.0121996, -0.0417468, 0.0386509
6: -0.0234614, 0.0209738, -0.0118004, 0.0145174, -0.0379789, 0.0327742
7: -0.0345205, 0.0018348, -0.0283304, -0.0018674, -0.0326531, 0.0301651
8: -0.0189528, 0.0413345, -0.0107633, 0.0245537, -0.0435064, 0.0520978
9: -0.0154641, 0.0194506, -0.0096260, 0.0094520, -0.0249160, 0.0290766

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
time: 2.19 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309049, upper bound: 0.0306951
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0315280, 0.0213485, -0.0099736, 0.0138339, -0.0453619, 0.0313221
1: -0.0148071, 0.0016818, -0.0104380, -0.0009846, -0.0138226, 0.0121198
2: -0.0010880, 0.0367208, -0.0022414, 0.0256107, -0.0266987, 0.0389621
3: -0.0226717, 0.0256824, -0.0126457, 0.0116983, -0.0343700, 0.0383281
4: -0.0194680, 0.0199681, -0.0123615, 0.0102047, -0.0296726, 0.0323296
5: 0.9707335, 1.0237035, 0.9863561, 1.0113440, -0.0406104, 0.0373474
6: -0.0232407, 0.0208516, -0.0109496, 0.0140464, -0.0372870, 0.0318012
7: -0.0344033, 0.0016365, -0.0278788, -0.0018736, -0.0325297, 0.0295152
8: -0.0187977, 0.0410168, -0.0101658, 0.0233294, -0.0421271, 0.0511826
9: -0.0153535, 0.0192613, -0.0092001, 0.0087225, -0.0240760, 0.0284614

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
time: 1.95 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308867, upper bound: 0.0304566
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0121589, 0.0145958, -0.0088254, 0.0126686, -0.0248275, 0.0234211
1: -0.0108810, -0.0007142, -0.0097605, -0.0013980, -0.0094829, 0.0090463
2: -0.0022171, 0.0267371, -0.0014807, 0.0238878, -0.0261049, 0.0282178
3: -0.0136622, 0.0131161, -0.0110910, 0.0095298, -0.0231920, 0.0242071
4: -0.0130820, 0.0111945, -0.0112595, 0.0086906, -0.0217726, 0.0224541
5: 0.9847723, 1.0125971, 0.9887787, 1.0094277, -0.0246554, 0.0238184
6: -0.0121958, 0.0147363, -0.0090437, 0.0129911, -0.0251869, 0.0237800
7: -0.0285403, -0.0018907, -0.0268670, -0.0024086, -0.0261317, 0.0249763
8: -0.0110410, 0.0251226, -0.0088273, 0.0205866, -0.0316275, 0.0339499
9: -0.0098239, 0.0097909, -0.0082459, 0.0070882, -0.0169122, 0.0180368

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.29 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.33 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0448981, 0.0542921, -0.0088254, 0.0126686, -0.0575668, 0.0631175
1: -0.0175173, 0.0047773, -0.0097605, -0.0013980, -0.0161193, 0.0145378
2: -0.0017526, 0.0543944, -0.0014807, 0.0238878, -0.0256404, 0.0558751
3: -0.0288909, 0.0430322, -0.0110910, 0.0095298, -0.0384207, 0.0541233
4: -0.0238761, 0.0279631, -0.0112595, 0.0086906, -0.0325668, 0.0392226
5: 0.9004847, 1.0313700, 0.9887787, 1.0094277, -0.1089430, 0.0425913
6: -0.0366282, 0.0250728, -0.0090437, 0.0129911, -0.0496193, 0.0341165
7: -0.0384505, 0.0223546, -0.0268670, -0.0024086, -0.0360420, 0.0492216
8: -0.0241521, 0.0613036, -0.0088273, 0.0205866, -0.0447386, 0.0701309
9: -0.0191705, 0.0265738, -0.0082459, 0.0070882, -0.0262588, 0.0348197

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310470, upper bound: 0.0308049
time: 2.03 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310289, upper bound: 0.0305134
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0121589, 0.0145958, -0.0121589, 0.0145958, -0.0267547, 0.0267547
1: -0.0108810, -0.0007142, -0.0108810, -0.0007142, -0.0101667, 0.0101667
2: -0.0022171, 0.0267371, -0.0022171, 0.0267371, -0.0289542, 0.0289542
3: -0.0136622, 0.0131161, -0.0136622, 0.0131161, -0.0267783, 0.0267783
4: -0.0130820, 0.0111945, -0.0130820, 0.0111945, -0.0242765, 0.0242765
5: 0.9847723, 1.0125971, 0.9847723, 1.0125971, -0.0278248, 0.0278248
6: -0.0121958, 0.0147363, -0.0121958, 0.0147363, -0.0269321, 0.0269321
7: -0.0285403, -0.0018907, -0.0285403, -0.0018907, -0.0266496, 0.0266496
8: -0.0110410, 0.0251226, -0.0110410, 0.0251226, -0.0361636, 0.0361636
9: -0.0098239, 0.0097909, -0.0098239, 0.0097909, -0.0196149, 0.0196149

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.36 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0448981, 0.0542921, -0.0121589, 0.0145958, -0.0594939, 0.0664510
1: -0.0175173, 0.0047773, -0.0108810, -0.0007142, -0.0168031, 0.0156582
2: -0.0017526, 0.0543944, -0.0022171, 0.0267371, -0.0284896, 0.0566115
3: -0.0288909, 0.0430322, -0.0136622, 0.0131161, -0.0420070, 0.0566945
4: -0.0238761, 0.0279631, -0.0130820, 0.0111945, -0.0350707, 0.0410451
5: 0.9004847, 1.0313700, 0.9847723, 1.0125971, -0.1121124, 0.0465978
6: -0.0366282, 0.0250728, -0.0121958, 0.0147363, -0.0513645, 0.0372686
7: -0.0384505, 0.0223546, -0.0285403, -0.0018907, -0.0365599, 0.0508948
8: -0.0241521, 0.0613036, -0.0110410, 0.0251226, -0.0492747, 0.0723446
9: -0.0191705, 0.0265738, -0.0098239, 0.0097909, -0.0289615, 0.0363978

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310087, upper bound: 0.0308049
time: 1.89 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309902, upper bound: 0.0305134
time: 2.08 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.88 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
NS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0309049, upper bound: 0.0306951
NS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
NS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0308867, upper bound: 0.0304566
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0310470, upper bound: 0.0308049
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0310289, upper bound: 0.0305134
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310085
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0310087, upper bound: 0.0308049
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.88
Output dim: 5, lower bound: -0.0309902, upper bound: 0.0305134

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0085480, 0.0109328, -0.0087684, 0.0122883, -0.0208363, 0.0197012
1: -0.0087512, -0.0020139, -0.0095394, -0.0015329, -0.0072183, 0.0075255
2: -0.0009037, 0.0213215, -0.0013623, 0.0233256, -0.0242293, 0.0226838
3: -0.0087751, 0.0062996, -0.0105836, 0.0088222, -0.0175973, 0.0168832
4: -0.0096180, 0.0067672, -0.0108999, 0.0081966, -0.0178146, 0.0176671
5: 0.9923873, 1.0065725, 0.9895692, 1.0088021, -0.0164148, 0.0170033
6: -0.0062045, 0.0114192, -0.0084217, 0.0126467, -0.0188513, 0.0198409
7: -0.0253599, -0.0028144, -0.0265368, -0.0024919, -0.0228680, 0.0237224
8: -0.0070024, 0.0165009, -0.0083905, 0.0196915, -0.0266939, 0.0248914
9: -0.0068245, 0.0046538, -0.0079345, 0.0065549, -0.0133794, 0.0125883

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0086494, 0.0130114, -0.0087622, 0.0121831, -0.0208325, 0.0217736
1: -0.0099598, -0.0012764, -0.0094782, -0.0015703, -0.0083895, 0.0082018
2: -0.0011146, 0.0243946, -0.0013494, 0.0231701, -0.0242847, 0.0257439
3: -0.0115483, 0.0101677, -0.0104433, 0.0086264, -0.0201747, 0.0206109
4: -0.0115837, 0.0091360, -0.0108004, 0.0080599, -0.0196435, 0.0199364
5: 0.9880661, 1.0099914, 0.9897879, 1.0086290, -0.0205628, 0.0202035
6: -0.0096043, 0.0133015, -0.0082496, 0.0125515, -0.0221558, 0.0215511
7: -0.0271646, -0.0026661, -0.0264455, -0.0025010, -0.0246636, 0.0237794
8: -0.0092210, 0.0213933, -0.0082696, 0.0194439, -0.0286649, 0.0296630
9: -0.0085265, 0.0075689, -0.0078483, 0.0064074, -0.0149339, 0.0154172

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0086859, 0.0119366, -0.0087471, 0.0129486, -0.0216345, 0.0206836
1: -0.0093348, -0.0016578, -0.0099233, -0.0012987, -0.0080362, 0.0082655
2: -0.0011906, 0.0228055, -0.0013179, 0.0243018, -0.0254923, 0.0241234
3: -0.0101143, 0.0081676, -0.0114645, 0.0100508, -0.0201651, 0.0196321
4: -0.0105672, 0.0077395, -0.0115243, 0.0090544, -0.0196216, 0.0192638
5: 0.9903004, 1.0082235, 0.9881966, 1.0098879, -0.0195875, 0.0200270
6: -0.0078464, 0.0123282, -0.0095016, 0.0132446, -0.0210910, 0.0218298
7: -0.0262314, -0.0026127, -0.0271101, -0.0025231, -0.0237083, 0.0244974
8: -0.0079864, 0.0188636, -0.0091489, 0.0212455, -0.0292319, 0.0280124
9: -0.0076464, 0.0060616, -0.0084751, 0.0074808, -0.0151273, 0.0145367

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0088193, 0.0126219, -0.0101426, 0.0138928, -0.0227121, 0.0227645
1: -0.0097333, -0.0014146, -0.0104723, -0.0009636, -0.0087697, 0.0090576
2: -0.0014681, 0.0238187, -0.0020202, 0.0256978, -0.0271659, 0.0258389
3: -0.0110286, 0.0094428, -0.0127244, 0.0118080, -0.0228367, 0.0221672
4: -0.0112153, 0.0086299, -0.0124173, 0.0102812, -0.0214965, 0.0210472
5: 0.9888757, 1.0093509, 0.9862335, 1.0114410, -0.0225652, 0.0231174
6: -0.0089672, 0.0129488, -0.0110461, 0.0140997, -0.0230670, 0.0239948
7: -0.0268264, -0.0024174, -0.0279299, -0.0020292, -0.0247972, 0.0255125
8: -0.0087736, 0.0204765, -0.0102335, 0.0234681, -0.0322417, 0.0307101
9: -0.0082076, 0.0070226, -0.0092484, 0.0088051, -0.0170127, 0.0162710

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0268589, 0.0197207, -0.0103604, 0.0139688, -0.0408277, 0.0300811
1: -0.0138607, 0.0011042, -0.0105164, -0.0009367, -0.0129240, 0.0116206
2: -0.0004051, 0.0343142, -0.0021322, 0.0258100, -0.0262151, 0.0364464
3: -0.0204999, 0.0226532, -0.0128256, 0.0119493, -0.0324492, 0.0354789
4: -0.0179286, 0.0178532, -0.0124890, 0.0103799, -0.0283085, 0.0303422
5: 0.9741176, 1.0210261, 0.9860757, 1.0115658, -0.0374482, 0.0349504
6: -0.0205782, 0.0193775, -0.0111702, 0.0141685, -0.0347467, 0.0305477
7: -0.0329900, -0.0007547, -0.0279959, -0.0019504, -0.0310396, 0.0272411
8: -0.0169279, 0.0371854, -0.0103207, 0.0236468, -0.0405747, 0.0475062
9: -0.0140206, 0.0169784, -0.0093105, 0.0089116, -0.0229322, 0.0262889

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309030, upper bound: 0.0308063
time: 1.89 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0328107, 0.0217957, -0.0100349, 0.0138553, -0.0466660, 0.0318306
1: -0.0150671, 0.0018405, -0.0104504, -0.0009770, -0.0140902, 0.0122909
2: -0.0005817, 0.0373820, -0.0021186, 0.0256423, -0.0262240, 0.0395006
3: -0.0232684, 0.0265146, -0.0126743, 0.0117381, -0.0350065, 0.0391889
4: -0.0198909, 0.0205491, -0.0123817, 0.0102324, -0.0301233, 0.0329309
5: 0.9698038, 1.0244390, 0.9863116, 1.0113792, -0.0415754, 0.0381274
6: -0.0239721, 0.0212566, -0.0109846, 0.0140657, -0.0380378, 0.0322412
7: -0.0347916, 0.0022934, -0.0278973, -0.0019599, -0.0328317, 0.0301907
8: -0.0193114, 0.0420694, -0.0101904, 0.0233797, -0.0426911, 0.0522597
9: -0.0157197, 0.0198885, -0.0092176, 0.0087525, -0.0244722, 0.0291061

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308891, upper bound: 0.0306951
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309049, upper bound: 0.0306951
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0264882, 0.0195914, -0.0091343, 0.0134475, -0.0399357, 0.0287257
1: -0.0137856, 0.0010584, -0.0102133, -0.0011216, -0.0126639, 0.0112717
2: -0.0005038, 0.0341231, -0.0021234, 0.0250394, -0.0255432, 0.0362464
3: -0.0203275, 0.0224127, -0.0121302, 0.0109793, -0.0313068, 0.0345429
4: -0.0178064, 0.0176852, -0.0119962, 0.0097026, -0.0275090, 0.0296814
5: 0.9743863, 1.0208138, 0.9871593, 1.0107086, -0.0363222, 0.0336545
6: -0.0203668, 0.0192604, -0.0103177, 0.0136965, -0.0340633, 0.0295781
7: -0.0328778, -0.0009446, -0.0275433, -0.0019566, -0.0309212, 0.0265987
8: -0.0167794, 0.0368812, -0.0097220, 0.0224199, -0.0391993, 0.0466032
9: -0.0139148, 0.0167971, -0.0088837, 0.0081806, -0.0220954, 0.0256808

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
time: 2.20 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0320665, 0.0215363, -0.0091278, 0.0133377, -0.0454043, 0.0306641
1: -0.0149163, 0.0017484, -0.0101495, -0.0011606, -0.0137557, 0.0118980
2: -0.0006773, 0.0369984, -0.0021099, 0.0248771, -0.0255545, 0.0391083
3: -0.0229222, 0.0260318, -0.0119838, 0.0107750, -0.0336973, 0.0380156
4: -0.0196455, 0.0202121, -0.0118923, 0.0095600, -0.0292056, 0.0321044
5: 0.9703432, 1.0240122, 0.9873876, 1.0105280, -0.0401848, 0.0366246
6: -0.0235478, 0.0210216, -0.0101382, 0.0135971, -0.0371448, 0.0311598
7: -0.0345664, 0.0019123, -0.0274480, -0.0019661, -0.0326003, 0.0293603
8: -0.0190134, 0.0414587, -0.0095959, 0.0221616, -0.0411750, 0.0510546
9: -0.0155073, 0.0195246, -0.0087938, 0.0080267, -0.0235339, 0.0283184

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308662, upper bound: 0.0304555
time: 5.38 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308867, upper bound: 0.0304566
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0087471, 0.0129486, -0.0086859, 0.0119366, -0.0206836, 0.0216345
1: -0.0099233, -0.0012987, -0.0093348, -0.0016578, -0.0082655, 0.0080362
2: -0.0013179, 0.0243018, -0.0011906, 0.0228055, -0.0241234, 0.0254923
3: -0.0114645, 0.0100508, -0.0101143, 0.0081676, -0.0196321, 0.0201651
4: -0.0115243, 0.0090544, -0.0105672, 0.0077395, -0.0192638, 0.0196216
5: 0.9881966, 1.0098879, 0.9903004, 1.0082235, -0.0200270, 0.0195875
6: -0.0095016, 0.0132446, -0.0078464, 0.0123282, -0.0218298, 0.0210910
7: -0.0271101, -0.0025231, -0.0262314, -0.0026127, -0.0244974, 0.0237083
8: -0.0091489, 0.0212455, -0.0079864, 0.0188636, -0.0280124, 0.0292319
9: -0.0084751, 0.0074808, -0.0076464, 0.0060616, -0.0145367, 0.0151273

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.20 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0101426, 0.0138928, -0.0088193, 0.0126219, -0.0227645, 0.0227121
1: -0.0104723, -0.0009636, -0.0097333, -0.0014146, -0.0090576, 0.0087697
2: -0.0020202, 0.0256978, -0.0014681, 0.0238187, -0.0258389, 0.0271659
3: -0.0127244, 0.0118080, -0.0110286, 0.0094428, -0.0221672, 0.0228367
4: -0.0124173, 0.0102812, -0.0112153, 0.0086299, -0.0210472, 0.0214965
5: 0.9862335, 1.0114410, 0.9888757, 1.0093509, -0.0231174, 0.0225652
6: -0.0110461, 0.0140997, -0.0089672, 0.0129488, -0.0239948, 0.0230670
7: -0.0279299, -0.0020292, -0.0268264, -0.0024174, -0.0255125, 0.0247972
8: -0.0102335, 0.0234681, -0.0087736, 0.0204765, -0.0307101, 0.0322417
9: -0.0092484, 0.0088051, -0.0082076, 0.0070226, -0.0162710, 0.0170127

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.22 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.39 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0374014, 0.0449168, -0.0088208, 0.0122197, -0.0496211, 0.0537376
1: -0.0159977, 0.0035053, -0.0094995, -0.0015573, -0.0144404, 0.0130048
2: -0.0016955, 0.0479525, -0.0014711, 0.0232242, -0.0249197, 0.0494236
3: -0.0254037, 0.0360943, -0.0104921, 0.0086945, -0.0340983, 0.0465865
4: -0.0214044, 0.0241038, -0.0108351, 0.0081074, -0.0295118, 0.0349388
5: 0.9203963, 1.0270714, 0.9897117, 1.0086893, -0.0882930, 0.0373597
6: -0.0309753, 0.0227059, -0.0083095, 0.0125846, -0.0435599, 0.0310154
7: -0.0361812, 0.0151990, -0.0264773, -0.0024153, -0.0337659, 0.0416763
8: -0.0211498, 0.0529247, -0.0083117, 0.0195301, -0.0406799, 0.0612364
9: -0.0170303, 0.0227230, -0.0078783, 0.0064587, -0.0234890, 0.0306013

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310321, upper bound: 0.0308049
time: 2.29 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310470, upper bound: 0.0308049
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0373753, 0.0448842, -0.0088170, 0.0116825, -0.0490578, 0.0537012
1: -0.0159924, 0.0035008, -0.0091871, -0.0017479, -0.0142445, 0.0126880
2: -0.0017910, 0.0479301, -0.0014634, 0.0224299, -0.0242209, 0.0493935
3: -0.0253916, 0.0360702, -0.0097754, 0.0076948, -0.0330864, 0.0458456
4: -0.0213958, 0.0240903, -0.0103270, 0.0074094, -0.0288053, 0.0344173
5: 0.9204656, 1.0270565, 0.9908287, 1.0078057, -0.0873401, 0.0362278
6: -0.0309557, 0.0226977, -0.0074308, 0.0120981, -0.0430538, 0.0301285
7: -0.0361733, 0.0151741, -0.0260108, -0.0024208, -0.0337526, 0.0411849
8: -0.0211394, 0.0528955, -0.0076946, 0.0182656, -0.0394050, 0.0605901
9: -0.0170229, 0.0227096, -0.0074384, 0.0057053, -0.0227281, 0.0301480

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310083, upper bound: 0.0305134
time: 2.27 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310289, upper bound: 0.0305134
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0087471, 0.0129486, -0.0100343, 0.0138550, -0.0226021, 0.0229829
1: -0.0099233, -0.0012987, -0.0104503, -0.0009770, -0.0089462, 0.0091516
2: -0.0013179, 0.0243018, -0.0019226, 0.0256420, -0.0269598, 0.0262243
3: -0.0114645, 0.0100508, -0.0126740, 0.0117377, -0.0232023, 0.0227248
4: -0.0115243, 0.0090544, -0.0123815, 0.0102321, -0.0217564, 0.0214359
5: 0.9881966, 1.0098879, 0.9863120, 1.0113789, -0.0231823, 0.0235759
6: -0.0095016, 0.0132446, -0.0109843, 0.0140655, -0.0235672, 0.0242289
7: -0.0271101, -0.0025231, -0.0278971, -0.0020978, -0.0250123, 0.0253740
8: -0.0091489, 0.0212455, -0.0101901, 0.0233792, -0.0325281, 0.0314357
9: -0.0084751, 0.0074808, -0.0092174, 0.0087521, -0.0172273, 0.0166982

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.07 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0101426, 0.0138928, -0.0120229, 0.0145483, -0.0246910, 0.0259157
1: -0.0104723, -0.0009636, -0.0108534, -0.0007310, -0.0097412, 0.0098898
2: -0.0020202, 0.0256978, -0.0022036, 0.0266670, -0.0286871, 0.0279014
3: -0.0127244, 0.0118080, -0.0135989, 0.0130278, -0.0257522, 0.0254070
4: -0.0124173, 0.0102812, -0.0130372, 0.0111329, -0.0235502, 0.0233184
5: 0.9862335, 1.0114410, 0.9848709, 1.0125191, -0.0262856, 0.0265701
6: -0.0110461, 0.0140997, -0.0121182, 0.0146934, -0.0257394, 0.0262179
7: -0.0279299, -0.0020292, -0.0284991, -0.0019002, -0.0260297, 0.0264699
8: -0.0102335, 0.0234681, -0.0109865, 0.0250110, -0.0352445, 0.0344546
9: -0.0092484, 0.0088051, -0.0097851, 0.0097244, -0.0189728, 0.0185902

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.43 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0374014, 0.0449168, -0.0107696, 0.0141114, -0.0515128, 0.0556864
1: -0.0159977, 0.0035053, -0.0105993, -0.0008861, -0.0151116, 0.0141046
2: -0.0016955, 0.0479525, -0.0022066, 0.0260209, -0.0277164, 0.0501590
3: -0.0254037, 0.0360943, -0.0130160, 0.0122147, -0.0376184, 0.0491103
4: -0.0214044, 0.0241038, -0.0126239, 0.0105652, -0.0319696, 0.0367277
5: 0.9203963, 1.0270714, 0.9857792, 1.0118006, -0.0914043, 0.0412922
6: -0.0309753, 0.0227059, -0.0114035, 0.0142977, -0.0452730, 0.0341094
7: -0.0361812, 0.0151990, -0.0281197, -0.0018981, -0.0342831, 0.0433187
8: -0.0211498, 0.0529247, -0.0104846, 0.0239825, -0.0451323, 0.0634093
9: -0.0170303, 0.0227230, -0.0094273, 0.0091116, -0.0261419, 0.0321503

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309935, upper bound: 0.0308049
time: 2.06 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310087, upper bound: 0.0308049
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0373753, 0.0448842, -0.0092724, 0.0135894, -0.0509647, 0.0541566
1: -0.0159924, 0.0035008, -0.0102958, -0.0010713, -0.0149211, 0.0137967
2: -0.0017910, 0.0479301, -0.0021977, 0.0252492, -0.0270402, 0.0501278
3: -0.0253916, 0.0360702, -0.0123195, 0.0112434, -0.0366350, 0.0483897
4: -0.0213958, 0.0240903, -0.0121303, 0.0098870, -0.0312828, 0.0362207
5: 0.9204656, 1.0270565, 0.9868644, 1.0109420, -0.0904764, 0.0401921
6: -0.0309557, 0.0226977, -0.0105498, 0.0138250, -0.0447807, 0.0332474
7: -0.0361733, 0.0151741, -0.0276665, -0.0019043, -0.0342690, 0.0428406
8: -0.0211394, 0.0528955, -0.0098850, 0.0227539, -0.0438933, 0.0627805
9: -0.0170229, 0.0227096, -0.0089999, 0.0083796, -0.0254025, 0.0317095

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309703, upper bound: 0.0305134
time: 2.03 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309902, upper bound: 0.0305134
time: 2.22 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.30 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0309030, upper bound: 0.0308063
NS_A1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
NS_A1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0308891, upper bound: 0.0306951
NS_A1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0309049, upper bound: 0.0306951
NS_A1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
NS_A1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
NS_A1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0308662, upper bound: 0.0304555
NS_A1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0308867, upper bound: 0.0304566
NS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0310321, upper bound: 0.0308049
NS_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0310470, upper bound: 0.0308049
NS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0310083, upper bound: 0.0305134
NS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0310289, upper bound: 0.0305134
NS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0309935, upper bound: 0.0308049
NS_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0310087, upper bound: 0.0308049
NS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0309703, upper bound: 0.0305134
NS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.30
Output dim: 5, lower bound: -0.0309902, upper bound: 0.0305134

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0084077, 0.0102160, -0.0083416, 0.0106107, -0.0190184, 0.0185576
1: -0.0083493, -0.0022682, -0.0085640, -0.0021282, -0.0062211, 0.0062957
2: -0.0006118, 0.0203888, -0.0004744, 0.0208453, -0.0214571, 0.0208632
3: -0.0078188, 0.0052459, -0.0083454, 0.0057003, -0.0135190, 0.0135913
4: -0.0090230, 0.0065491, -0.0093134, 0.0064464, -0.0154694, 0.0158625
5: 0.9931333, 1.0053937, 0.9930568, 1.0060430, -0.0129097, 0.0123369
6: -0.0050322, 0.0108281, -0.0056778, 0.0111275, -0.0161597, 0.0165059
7: -0.0247376, -0.0030197, -0.0250802, -0.0031164, -0.0216212, 0.0220605
8: -0.0068426, 0.0150821, -0.0067674, 0.0157428, -0.0225854, 0.0218495
9: -0.0062376, 0.0039036, -0.0065607, 0.0042021, -0.0104397, 0.0104643

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0309300
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0304971
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0085421, 0.0108868, -0.0086804, 0.0115931, -0.0201352, 0.0195673
1: -0.0087245, -0.0020302, -0.0091351, -0.0017796, -0.0069449, 0.0071049
2: -0.0008914, 0.0212536, -0.0011792, 0.0222977, -0.0231891, 0.0224328
3: -0.0087138, 0.0062141, -0.0096561, 0.0075284, -0.0162421, 0.0158702
4: -0.0095746, 0.0067580, -0.0102424, 0.0072933, -0.0168678, 0.0170005
5: 0.9924828, 1.0064970, 0.9910146, 1.0076587, -0.0151759, 0.0154824
6: -0.0061294, 0.0113775, -0.0072846, 0.0120171, -0.0181465, 0.0186621
7: -0.0253200, -0.0028231, -0.0259332, -0.0026206, -0.0226993, 0.0231101
8: -0.0069956, 0.0163928, -0.0075919, 0.0180551, -0.0250508, 0.0239846
9: -0.0067868, 0.0045894, -0.0073652, 0.0055799, -0.0123667, 0.0119546

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307342
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0307342
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0085117, 0.0122915, -0.0083373, 0.0105013, -0.0190129, 0.0206288
1: -0.0095412, -0.0015318, -0.0085037, -0.0021670, -0.0073742, 0.0069719
2: -0.0008282, 0.0233304, -0.0004654, 0.0207123, -0.0215404, 0.0237957
3: -0.0105879, 0.0088281, -0.0081993, 0.0055601, -0.0161480, 0.0170274
4: -0.0109030, 0.0082007, -0.0092287, 0.0064396, -0.0173426, 0.0174294
5: 0.9895625, 1.0088074, 0.9931157, 1.0058628, -0.0163004, 0.0156918
6: -0.0084269, 0.0126496, -0.0054987, 0.0110415, -0.0194685, 0.0181483
7: -0.0265396, -0.0028676, -0.0249852, -0.0031227, -0.0234169, 0.0221176
8: -0.0083942, 0.0196991, -0.0067624, 0.0155460, -0.0239401, 0.0264615
9: -0.0079371, 0.0065594, -0.0064711, 0.0041064, -0.0120435, 0.0130305

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306342
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0304644
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0086432, 0.0129640, -0.0086731, 0.0114875, -0.0201307, 0.0216372
1: -0.0099322, -0.0012932, -0.0090737, -0.0018171, -0.0081152, 0.0077806
2: -0.0011017, 0.0243246, -0.0011641, 0.0221416, -0.0232433, 0.0254887
3: -0.0114852, 0.0100796, -0.0095152, 0.0073319, -0.0188171, 0.0195948
4: -0.0115389, 0.0090745, -0.0101426, 0.0071561, -0.0186950, 0.0192171
5: 0.9881644, 1.0099134, 0.9912341, 1.0074848, -0.0193204, 0.0186793
6: -0.0095269, 0.0132587, -0.0071118, 0.0119215, -0.0214484, 0.0203705
7: -0.0271235, -0.0026751, -0.0258415, -0.0026313, -0.0244922, 0.0231664
8: -0.0091667, 0.0212820, -0.0074706, 0.0178066, -0.0269732, 0.0287526
9: -0.0084878, 0.0075025, -0.0072787, 0.0054318, -0.0139196, 0.0147813

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306614
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306614
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0084077, 0.0102160, -0.0086902, 0.0125613, -0.0209690, 0.0189062
1: -0.0083493, -0.0022682, -0.0096981, -0.0014361, -0.0069132, 0.0074298
2: -0.0006118, 0.0203888, -0.0011994, 0.0237292, -0.0243410, 0.0215883
3: -0.0078188, 0.0052459, -0.0109478, 0.0093301, -0.0171489, 0.0161937
4: -0.0090230, 0.0065491, -0.0111580, 0.0085512, -0.0175742, 0.0177071
5: 0.9931333, 1.0053937, 0.9890018, 1.0092509, -0.0161176, 0.0163919
6: -0.0050322, 0.0108281, -0.0088682, 0.0128939, -0.0179261, 0.0196963
7: -0.0247376, -0.0030197, -0.0267738, -0.0026064, -0.0221311, 0.0237541
8: -0.0068426, 0.0150821, -0.0087040, 0.0203340, -0.0271766, 0.0237861
9: -0.0062376, 0.0039036, -0.0081580, 0.0069377, -0.0131753, 0.0120615

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0309810
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0305344
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0085117, 0.0122915, -0.0086825, 0.0124519, -0.0209636, 0.0209741
1: -0.0095412, -0.0015318, -0.0096345, -0.0014749, -0.0080663, 0.0081027
2: -0.0008282, 0.0233304, -0.0011836, 0.0235674, -0.0243956, 0.0245140
3: -0.0105879, 0.0088281, -0.0108019, 0.0091265, -0.0197144, 0.0196300
4: -0.0109030, 0.0082007, -0.0110546, 0.0084091, -0.0193120, 0.0192553
5: 0.9895625, 1.0088074, 0.9892291, 1.0090712, -0.0195088, 0.0195783
6: -0.0084269, 0.0126496, -0.0086892, 0.0127949, -0.0212218, 0.0213389
7: -0.0265396, -0.0028676, -0.0266788, -0.0026175, -0.0239221, 0.0238113
8: -0.0083942, 0.0196991, -0.0085784, 0.0200765, -0.0284707, 0.0282774
9: -0.0079371, 0.0065594, -0.0080684, 0.0067843, -0.0147214, 0.0146278

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306343
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0304979
time: 1.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0087624, 0.0122417, -0.0088137, 0.0120765, -0.0208389, 0.0210554
1: -0.0095122, -0.0015495, -0.0094162, -0.0016081, -0.0079041, 0.0078667
2: -0.0013498, 0.0232566, -0.0014564, 0.0230124, -0.0243622, 0.0247130
3: -0.0105213, 0.0087353, -0.0103010, 0.0084279, -0.0189492, 0.0190363
4: -0.0108558, 0.0081359, -0.0106996, 0.0079213, -0.0187770, 0.0188355
5: 0.9896663, 1.0087253, 0.9900096, 1.0084537, -0.0187874, 0.0187157
6: -0.0083453, 0.0126045, -0.0080752, 0.0124549, -0.0208002, 0.0206796
7: -0.0264963, -0.0025007, -0.0263529, -0.0024257, -0.0240706, 0.0238522
8: -0.0083369, 0.0195816, -0.0081471, 0.0191928, -0.0275297, 0.0277287
9: -0.0078962, 0.0064894, -0.0077610, 0.0062578, -0.0141540, 0.0142504

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306614
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307340, upper bound: 0.0306614
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0087562, 0.0121365, -0.0107332, 0.0140987, -0.0228549, 0.0228697
1: -0.0094511, -0.0015868, -0.0105920, -0.0008906, -0.0085605, 0.0090051
2: -0.0013367, 0.0231011, -0.0016416, 0.0260022, -0.0273389, 0.0247427
3: -0.0103810, 0.0085395, -0.0129990, 0.0121911, -0.0225721, 0.0215386
4: -0.0107563, 0.0079992, -0.0126119, 0.0105487, -0.0213050, 0.0206112
5: 0.9898849, 1.0085523, 0.9858055, 1.0117797, -0.0218948, 0.0227468
6: -0.0081733, 0.0125092, -0.0113828, 0.0142862, -0.0224595, 0.0238920
7: -0.0264050, -0.0025099, -0.0281087, -0.0022954, -0.0241095, 0.0255988
8: -0.0082160, 0.0193341, -0.0104700, 0.0239527, -0.0321687, 0.0298041
9: -0.0078101, 0.0063419, -0.0094169, 0.0090938, -0.0169040, 0.0157588

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0311350
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0306614
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0221107, 0.0180653, -0.0089956, 0.0132284, -0.0353391, 0.0270609
1: -0.0128982, 0.0005169, -0.0100860, -0.0011994, -0.0116988, 0.0106028
2: 0.0004571, 0.0318667, -0.0018349, 0.0247155, -0.0242584, 0.0337015
3: -0.0182913, 0.0195726, -0.0118379, 0.0105716, -0.0288629, 0.0314105
4: -0.0163631, 0.0157024, -0.0117889, 0.0094180, -0.0257811, 0.0274913
5: 0.9775592, 1.0183036, 0.9876148, 1.0103483, -0.0327891, 0.0306888
6: -0.0178706, 0.0178783, -0.0099594, 0.0134981, -0.0313687, 0.0278377
7: -0.0315527, -0.0031865, -0.0273531, -0.0021595, -0.0293932, 0.0241665
8: -0.0150264, 0.0332890, -0.0094704, 0.0219043, -0.0369306, 0.0427593
9: -0.0126650, 0.0146568, -0.0087043, 0.0078734, -0.0205384, 0.0233611

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_A1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309030, upper bound: 0.0308063
time: 2.07 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309030, upper bound: 0.0308063
time: 3.05 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0248501, 0.0190204, -0.0102249, 0.0139215, -0.0387716, 0.0292452
1: -0.0134535, 0.0008557, -0.0104889, -0.0009535, -0.0125000, 0.0113447
2: -0.0002320, 0.0332787, -0.0021194, 0.0257402, -0.0259722, 0.0353981
3: -0.0195655, 0.0213499, -0.0127626, 0.0118613, -0.0314268, 0.0341125
4: -0.0172663, 0.0169432, -0.0124444, 0.0103184, -0.0275847, 0.0293876
5: 0.9755737, 1.0198746, 0.9861740, 1.0114882, -0.0359145, 0.0337006
6: -0.0194327, 0.0187432, -0.0110929, 0.0141257, -0.0335584, 0.0298361
7: -0.0323819, -0.0017835, -0.0279548, -0.0019594, -0.0304225, 0.0261713
8: -0.0161234, 0.0355370, -0.0102664, 0.0235355, -0.0396589, 0.0458034
9: -0.0134471, 0.0159962, -0.0092718, 0.0088453, -0.0222924, 0.0252680

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_A1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0281136, 0.0201581, -0.0089904, 0.0131152, -0.0412288, 0.0291485
1: -0.0141150, 0.0012594, -0.0100201, -0.0012396, -0.0128755, 0.0112796
2: 0.0002806, 0.0349608, -0.0018241, 0.0245481, -0.0242675, 0.0367849
3: -0.0210835, 0.0234672, -0.0116869, 0.0103609, -0.0314444, 0.0351540
4: -0.0183422, 0.0184215, -0.0116819, 0.0092709, -0.0276131, 0.0301034
5: 0.9732083, 1.0217458, 0.9878502, 1.0101621, -0.0369538, 0.0338956
6: -0.0212936, 0.0197736, -0.0097741, 0.0133955, -0.0346892, 0.0295477
7: -0.0333698, -0.0001122, -0.0272547, -0.0021671, -0.0312027, 0.0271425
8: -0.0174303, 0.0382149, -0.0093403, 0.0216377, -0.0390681, 0.0475552
9: -0.0143788, 0.0175919, -0.0086116, 0.0077145, -0.0220933, 0.0262034

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308891, upper bound: 0.0306951
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308891, upper bound: 0.0306951
time: 2.02 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0307958, 0.0210932, -0.0098998, 0.0138082, -0.0446039, 0.0309930
1: -0.0146587, 0.0015912, -0.0104230, -0.0009937, -0.0136650, 0.0120143
2: -0.0004008, 0.0363434, -0.0021056, 0.0255726, -0.0259735, 0.0384490
3: -0.0223312, 0.0252074, -0.0126114, 0.0116504, -0.0339816, 0.0378188
4: -0.0192266, 0.0196364, -0.0123372, 0.0101712, -0.0293978, 0.0319736
5: 0.9712642, 1.0232837, 0.9864097, 1.0113018, -0.0400375, 0.0368741
6: -0.0228231, 0.0206204, -0.0109075, 0.0140231, -0.0368462, 0.0315279
7: -0.0341817, 0.0012615, -0.0278564, -0.0019691, -0.0322126, 0.0291179
8: -0.0185045, 0.0404160, -0.0101363, 0.0232688, -0.0417732, 0.0505522
9: -0.0151445, 0.0189033, -0.0091790, 0.0086864, -0.0238309, 0.0280823

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300041, upper bound: 0.0303130
time: 2.04 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299838, upper bound: 0.0297756
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0264882, 0.0195914, -0.0087601, 0.0113078, -0.0377960, 0.0283515
1: -0.0137856, 0.0010584, -0.0089693, -0.0018809, -0.0119047, 0.0100276
2: -0.0005038, 0.0341231, -0.0013449, 0.0218759, -0.0223797, 0.0354680
3: -0.0203275, 0.0224127, -0.0092754, 0.0069975, -0.0273249, 0.0316881
4: -0.0178064, 0.0176852, -0.0099726, 0.0070969, -0.0249033, 0.0276579
5: 0.9743863, 1.0208138, 0.9916077, 1.0071894, -0.0328031, 0.0292062
6: -0.0203668, 0.0192604, -0.0068179, 0.0117588, -0.0321256, 0.0260783
7: -0.0328778, -0.0009446, -0.0256855, -0.0025041, -0.0303736, 0.0247408
8: -0.0167794, 0.0368812, -0.0072642, 0.0173836, -0.0341630, 0.0441453
9: -0.0139148, 0.0167971, -0.0071315, 0.0051798, -0.0190945, 0.0239287

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308776, upper bound: 0.0305470
time: 1.95 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0264882, 0.0195914, -0.0091129, 0.0132025, -0.0396907, 0.0287043
1: -0.0137856, 0.0010584, -0.0100709, -0.0012086, -0.0125770, 0.0111293
2: -0.0005038, 0.0341231, -0.0020789, 0.0246772, -0.0251810, 0.0362019
3: -0.0203275, 0.0224127, -0.0118033, 0.0105234, -0.0308509, 0.0342160
4: -0.0178064, 0.0176852, -0.0117644, 0.0093843, -0.0271907, 0.0294497
5: 0.9743863, 1.0208138, 0.9876686, 1.0103056, -0.0359193, 0.0331452
6: -0.0203668, 0.0192604, -0.0099169, 0.0134746, -0.0338414, 0.0291773
7: -0.0328778, -0.0009446, -0.0273306, -0.0019879, -0.0308899, 0.0263859
8: -0.0167794, 0.0368812, -0.0094406, 0.0218433, -0.0386227, 0.0463217
9: -0.0139148, 0.0167971, -0.0086831, 0.0078370, -0.0217518, 0.0254802

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308776, upper bound: 0.0305470
time: 2.19 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
time: 2.16 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0276122, 0.0199833, -0.0089865, 0.0125994, -0.0402115, 0.0289698
1: -0.0140134, 0.0011974, -0.0097202, -0.0014226, -0.0125908, 0.0109176
2: 0.0001922, 0.0347024, -0.0018158, 0.0237855, -0.0235933, 0.0365182
3: -0.0208503, 0.0231419, -0.0109986, 0.0094010, -0.0302513, 0.0341405
4: -0.0181769, 0.0181944, -0.0111941, 0.0086007, -0.0267776, 0.0293884
5: 0.9735717, 1.0214581, 0.9889225, 1.0093136, -0.0357419, 0.0325356
6: -0.0210078, 0.0196153, -0.0089304, 0.0129284, -0.0339362, 0.0285457
7: -0.0332180, -0.0003690, -0.0268069, -0.0021729, -0.0310451, 0.0264379
8: -0.0172295, 0.0378035, -0.0087478, 0.0204236, -0.0376532, 0.0465513
9: -0.0142357, 0.0173467, -0.0081892, 0.0069911, -0.0212268, 0.0255359

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308662, upper bound: 0.0304555
time: 2.24 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308662, upper bound: 0.0304555
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0300501, 0.0208333, -0.0091215, 0.0132907, -0.0433408, 0.0299548
1: -0.0145075, 0.0014990, -0.0101222, -0.0011773, -0.0133303, 0.0116212
2: -0.0004950, 0.0359590, -0.0020968, 0.0248076, -0.0253025, 0.0380558
3: -0.0219843, 0.0247235, -0.0119210, 0.0106875, -0.0326717, 0.0366445
4: -0.0189807, 0.0192986, -0.0118478, 0.0094989, -0.0284796, 0.0311465
5: 0.9718047, 1.0228560, 0.9874853, 1.0104505, -0.0386457, 0.0353706
6: -0.0223979, 0.0203850, -0.0100612, 0.0135545, -0.0359524, 0.0304461
7: -0.0339559, 0.0008796, -0.0274071, -0.0019753, -0.0319806, 0.0282867
8: -0.0182058, 0.0398040, -0.0095419, 0.0220508, -0.0402566, 0.0493459
9: -0.0149316, 0.0185387, -0.0087553, 0.0079607, -0.0228923, 0.0272940

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299950, upper bound: 0.0301970
time: 2.25 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299781, upper bound: 0.0295970
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0086902, 0.0125613, -0.0084077, 0.0102160, -0.0189062, 0.0209690
1: -0.0096981, -0.0014361, -0.0083493, -0.0022682, -0.0074298, 0.0069132
2: -0.0011994, 0.0237292, -0.0006118, 0.0203888, -0.0215883, 0.0243410
3: -0.0109478, 0.0093301, -0.0078188, 0.0052459, -0.0161937, 0.0171489
4: -0.0111580, 0.0085512, -0.0090230, 0.0065491, -0.0177071, 0.0175742
5: 0.9890018, 1.0092509, 0.9931333, 1.0053937, -0.0163919, 0.0161176
6: -0.0088682, 0.0128939, -0.0050322, 0.0108281, -0.0196963, 0.0179261
7: -0.0267738, -0.0026064, -0.0247376, -0.0030197, -0.0237541, 0.0221311
8: -0.0087040, 0.0203340, -0.0068426, 0.0150821, -0.0237861, 0.0271766
9: -0.0081580, 0.0069377, -0.0062376, 0.0039036, -0.0120615, 0.0131753

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309810, upper bound: 0.0301871
time: 1.86 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305343, upper bound: 0.0301871
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086825, 0.0124519, -0.0085117, 0.0122915, -0.0209741, 0.0209636
1: -0.0096345, -0.0014749, -0.0095412, -0.0015318, -0.0081027, 0.0080663
2: -0.0011836, 0.0235674, -0.0008282, 0.0233304, -0.0245140, 0.0243956
3: -0.0108019, 0.0091265, -0.0105879, 0.0088281, -0.0196300, 0.0197144
4: -0.0110546, 0.0084091, -0.0109030, 0.0082007, -0.0192553, 0.0193120
5: 0.9892291, 1.0090712, 0.9895625, 1.0088074, -0.0195783, 0.0195088
6: -0.0086892, 0.0127949, -0.0084269, 0.0126496, -0.0213389, 0.0212218
7: -0.0266788, -0.0026175, -0.0265396, -0.0028676, -0.0238113, 0.0239221
8: -0.0085784, 0.0200765, -0.0083942, 0.0196991, -0.0282774, 0.0284707
9: -0.0080684, 0.0067843, -0.0079371, 0.0065594, -0.0146278, 0.0147214

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306343, upper bound: 0.0311350
time: 1.97 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304978, upper bound: 0.0301871
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0088137, 0.0120765, -0.0087624, 0.0122417, -0.0210554, 0.0208389
1: -0.0094162, -0.0016081, -0.0095122, -0.0015495, -0.0078667, 0.0079041
2: -0.0014564, 0.0230124, -0.0013498, 0.0232566, -0.0247130, 0.0243622
3: -0.0103010, 0.0084279, -0.0105213, 0.0087353, -0.0190363, 0.0189492
4: -0.0106996, 0.0079213, -0.0108558, 0.0081359, -0.0188355, 0.0187770
5: 0.9900096, 1.0084537, 0.9896663, 1.0087253, -0.0187157, 0.0187874
6: -0.0080752, 0.0124549, -0.0083453, 0.0126045, -0.0206796, 0.0208002
7: -0.0263529, -0.0024257, -0.0264963, -0.0025007, -0.0238522, 0.0240706
8: -0.0081471, 0.0191928, -0.0083369, 0.0195816, -0.0277287, 0.0275297
9: -0.0077610, 0.0062578, -0.0078962, 0.0064894, -0.0142504, 0.0141540

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0311350
time: 1.86 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0307340
time: 2.27 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0107332, 0.0140987, -0.0087562, 0.0121365, -0.0228697, 0.0228549
1: -0.0105920, -0.0008906, -0.0094511, -0.0015868, -0.0090051, 0.0085605
2: -0.0016416, 0.0260022, -0.0013367, 0.0231011, -0.0247427, 0.0273389
3: -0.0129990, 0.0121911, -0.0103810, 0.0085395, -0.0215386, 0.0225721
4: -0.0126119, 0.0105487, -0.0107563, 0.0079992, -0.0206112, 0.0213050
5: 0.9858055, 1.0117797, 0.9898849, 1.0085523, -0.0227468, 0.0218948
6: -0.0113828, 0.0142862, -0.0081733, 0.0125092, -0.0238920, 0.0224595
7: -0.0281087, -0.0022954, -0.0264050, -0.0025099, -0.0255988, 0.0241095
8: -0.0104700, 0.0239527, -0.0082160, 0.0193341, -0.0298041, 0.0321687
9: -0.0094169, 0.0090938, -0.0078101, 0.0063419, -0.0157588, 0.0169040

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306599
time: 1.56 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306599
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0330634, 0.0394918, -0.0086813, 0.0114896, -0.0445530, 0.0481731
1: -0.0151184, 0.0027692, -0.0090750, -0.0018163, -0.0133020, 0.0118442
2: -0.0008220, 0.0442249, -0.0011811, 0.0221447, -0.0229667, 0.0454060
3: -0.0233859, 0.0320797, -0.0095180, 0.0073358, -0.0307218, 0.0415977
4: -0.0199742, 0.0218706, -0.0101446, 0.0071588, -0.0271330, 0.0320152
5: 0.9319181, 1.0245839, 0.9912297, 1.0074884, -0.0755702, 0.0333543
6: -0.0277043, 0.0213363, -0.0071153, 0.0119234, -0.0396278, 0.0284516
7: -0.0348681, 0.0110584, -0.0258433, -0.0026193, -0.0322488, 0.0369018
8: -0.0194126, 0.0480762, -0.0074730, 0.0178115, -0.0372241, 0.0555493
9: -0.0157919, 0.0204947, -0.0072804, 0.0054348, -0.0212266, 0.0277752

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309451, upper bound: 0.0307875
time: 2.06 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309274, upper bound: 0.0306869
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0353778, 0.0423862, -0.0088147, 0.0121730, -0.0475508, 0.0512009
1: -0.0155875, 0.0031619, -0.0094723, -0.0015739, -0.0140136, 0.0126342
2: -0.0015037, 0.0462137, -0.0014585, 0.0231551, -0.0246588, 0.0476722
3: -0.0244625, 0.0342217, -0.0104297, 0.0086075, -0.0330700, 0.0446514
4: -0.0207373, 0.0230620, -0.0107908, 0.0080467, -0.0287839, 0.0338529
5: 0.9257709, 1.0259109, 0.9898090, 1.0086124, -0.0828415, 0.0361018
6: -0.0294495, 0.0220671, -0.0082330, 0.0125423, -0.0419918, 0.0303001
7: -0.0355687, 0.0132675, -0.0264367, -0.0024242, -0.0331445, 0.0397042
8: -0.0203394, 0.0506630, -0.0082580, 0.0194200, -0.0397594, 0.0589210
9: -0.0164526, 0.0216835, -0.0078400, 0.0063931, -0.0228457, 0.0295236

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309600, upper bound: 0.0307875
time: 2.59 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309460, upper bound: 0.0306869
time: 2.29 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0330357, 0.0394572, -0.0086776, 0.0109513, -0.0439870, 0.0481347
1: -0.0151127, 0.0027645, -0.0087620, -0.0020073, -0.0131054, 0.0115265
2: -0.0009117, 0.0442011, -0.0011732, 0.0213489, -0.0222606, 0.0453743
3: -0.0233730, 0.0320541, -0.0087998, 0.0063341, -0.0297072, 0.0408539
4: -0.0199650, 0.0218563, -0.0096355, 0.0069687, -0.0269337, 0.0314919
5: 0.9319918, 1.0245681, 0.9923487, 1.0066032, -0.0746115, 0.0322194
6: -0.0276835, 0.0213276, -0.0062349, 0.0114359, -0.0391194, 0.0275625
7: -0.0348597, 0.0110320, -0.0253760, -0.0026248, -0.0322349, 0.0364080
8: -0.0194015, 0.0480453, -0.0071500, 0.0165446, -0.0359460, 0.0551952
9: -0.0157840, 0.0204805, -0.0068396, 0.0046798, -0.0204638, 0.0273201

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309221, upper bound: 0.0305134
time: 2.05 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309067, upper bound: 0.0304250
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0353472, 0.0423478, -0.0088110, 0.0116357, -0.0469829, 0.0511588
1: -0.0155813, 0.0031567, -0.0091599, -0.0017645, -0.0138168, 0.0123166
2: -0.0016026, 0.0461873, -0.0014508, 0.0223607, -0.0239633, 0.0476381
3: -0.0244482, 0.0341933, -0.0097129, 0.0076077, -0.0320559, 0.0439062
4: -0.0207272, 0.0230462, -0.0102827, 0.0073486, -0.0280758, 0.0333290
5: 0.9258524, 1.0258934, 0.9909260, 1.0077289, -0.0818765, 0.0349675
6: -0.0294264, 0.0220574, -0.0073543, 0.0120557, -0.0414821, 0.0294116
7: -0.0355594, 0.0132382, -0.0259702, -0.0024296, -0.0331297, 0.0392084
8: -0.0203272, 0.0506287, -0.0076408, 0.0181554, -0.0384826, 0.0582696
9: -0.0164439, 0.0216678, -0.0074001, 0.0056396, -0.0220835, 0.0290679

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309413, upper bound: 0.0305134
time: 1.90 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309288, upper bound: 0.0304250
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0086902, 0.0125613, -0.0087620, 0.0120428, -0.0207330, 0.0213233
1: -0.0096981, -0.0014361, -0.0093966, -0.0016201, -0.0080780, 0.0079605
2: -0.0011994, 0.0237292, -0.0013490, 0.0229626, -0.0241620, 0.0250781
3: -0.0109478, 0.0093301, -0.0102561, 0.0083652, -0.0193130, 0.0195862
4: -0.0111580, 0.0085512, -0.0106677, 0.0078776, -0.0190356, 0.0192189
5: 0.9890018, 1.0092509, 0.9900796, 1.0083982, -0.0193964, 0.0191712
6: -0.0088682, 0.0128939, -0.0080201, 0.0124244, -0.0212925, 0.0209140
7: -0.0267738, -0.0026064, -0.0263236, -0.0025013, -0.0242726, 0.0237172
8: -0.0087040, 0.0203340, -0.0081085, 0.0191136, -0.0278176, 0.0284424
9: -0.0081580, 0.0069377, -0.0077334, 0.0062106, -0.0143686, 0.0146712

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309808, upper bound: 0.0301871
time: 1.76 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305343, upper bound: 0.0301871
time: 2.13 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086825, 0.0124519, -0.0107004, 0.0140873, -0.0227698, 0.0231523
1: -0.0096345, -0.0014749, -0.0105853, -0.0008946, -0.0087398, 0.0091104
2: -0.0011836, 0.0235674, -0.0015482, 0.0259853, -0.0271689, 0.0251156
3: -0.0108019, 0.0091265, -0.0129838, 0.0121698, -0.0229717, 0.0221103
4: -0.0110546, 0.0084091, -0.0126012, 0.0105339, -0.0215884, 0.0210102
5: 0.9892291, 1.0090712, 0.9858293, 1.0117608, -0.0225317, 0.0232419
6: -0.0086892, 0.0127949, -0.0113641, 0.0142758, -0.0229651, 0.0241589
7: -0.0266788, -0.0026175, -0.0280988, -0.0023611, -0.0243178, 0.0254812
8: -0.0085784, 0.0200765, -0.0104569, 0.0239257, -0.0325041, 0.0305334
9: -0.0080684, 0.0067843, -0.0094076, 0.0090778, -0.0171462, 0.0161918

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306328, upper bound: 0.0311350
time: 2.00 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304978, upper bound: 0.0301871
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0088137, 0.0120765, -0.0109163, 0.0141626, -0.0229763, 0.0229928
1: -0.0094162, -0.0016081, -0.0106291, -0.0008679, -0.0085483, 0.0090210
2: -0.0014564, 0.0230124, -0.0020852, 0.0260966, -0.0275530, 0.0250975
3: -0.0103010, 0.0084279, -0.0130842, 0.0123099, -0.0226109, 0.0215121
4: -0.0106996, 0.0079213, -0.0126723, 0.0106317, -0.0213312, 0.0205936
5: 0.9900096, 1.0084537, 0.9856728, 1.0118846, -0.0218750, 0.0227809
6: -0.0080752, 0.0124549, -0.0114872, 0.0143440, -0.0224192, 0.0239421
7: -0.0263529, -0.0024257, -0.0281641, -0.0019835, -0.0243694, 0.0257385
8: -0.0081471, 0.0191928, -0.0105434, 0.0241029, -0.0322501, 0.0297362
9: -0.0077610, 0.0062578, -0.0094692, 0.0091834, -0.0169444, 0.0157270

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B2_A1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0311350
time: 1.52 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0307340
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0107332, 0.0140987, -0.0105872, 0.0140478, -0.0247810, 0.0246859
1: -0.0105920, -0.0008906, -0.0105624, -0.0009086, -0.0096833, 0.0096718
2: -0.0016416, 0.0260022, -0.0020709, 0.0259270, -0.0275685, 0.0280731
3: -0.0129990, 0.0121911, -0.0129311, 0.0120964, -0.0250955, 0.0251223
4: -0.0126119, 0.0105487, -0.0125638, 0.0104826, -0.0230945, 0.0231125
5: 0.9858055, 1.0117797, 0.9859113, 1.0116961, -0.0258906, 0.0258684
6: -0.0113828, 0.0142862, -0.0112995, 0.0142401, -0.0256229, 0.0255857
7: -0.0281087, -0.0022954, -0.0280645, -0.0019935, -0.0261152, 0.0257691
8: -0.0104700, 0.0239527, -0.0104116, 0.0238329, -0.0343029, 0.0343642
9: -0.0094169, 0.0090938, -0.0093753, 0.0090225, -0.0184394, 0.0184691

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B2_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306599
time: 1.47 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0306599
time: 1.57 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0330634, 0.0394918, -0.0090328, 0.0133714, -0.0464348, 0.0485246
1: -0.0151184, 0.0027692, -0.0101691, -0.0011486, -0.0139697, 0.0129383
2: -0.0008220, 0.0442249, -0.0019121, 0.0249269, -0.0257489, 0.0461370
3: -0.0233859, 0.0320797, -0.0120286, 0.0108376, -0.0342236, 0.0441084
4: -0.0199742, 0.0218706, -0.0119242, 0.0096037, -0.0295779, 0.0337947
5: 0.9319181, 1.0245839, 0.9873176, 1.0105835, -0.0786654, 0.0372664
6: -0.0277043, 0.0213363, -0.0101932, 0.0136275, -0.0413319, 0.0315295
7: -0.0348681, 0.0110584, -0.0274772, -0.0021051, -0.0327630, 0.0385356
8: -0.0194126, 0.0480762, -0.0096346, 0.0222407, -0.0416533, 0.0577108
9: -0.0157919, 0.0204947, -0.0088214, 0.0080738, -0.0238657, 0.0293161

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B2_A2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309156, upper bound: 0.0307875
time: 1.98 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308981, upper bound: 0.0306869
time: 2.44 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0353778, 0.0423862, -0.0106336, 0.0140640, -0.0494418, 0.0530198
1: -0.0155875, 0.0031619, -0.0105718, -0.0009029, -0.0146846, 0.0137337
2: -0.0015037, 0.0462137, -0.0021931, 0.0259508, -0.0274546, 0.0484067
3: -0.0244625, 0.0342217, -0.0129527, 0.0121265, -0.0365890, 0.0471744
4: -0.0207373, 0.0230620, -0.0125791, 0.0105036, -0.0312409, 0.0356412
5: 0.9257709, 1.0259109, 0.9858778, 1.0117224, -0.0859515, 0.0400331
6: -0.0294495, 0.0220671, -0.0113260, 0.0142547, -0.0437043, 0.0333931
7: -0.0355687, 0.0132675, -0.0280785, -0.0019076, -0.0336611, 0.0413460
8: -0.0203394, 0.0506630, -0.0104301, 0.0238710, -0.0442104, 0.0610932
9: -0.0164526, 0.0216835, -0.0093885, 0.0090452, -0.0254978, 0.0310720

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309296, upper bound: 0.0307875
time: 1.94 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0306869
time: 2.33 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0330357, 0.0394572, -0.0090285, 0.0128515, -0.0458872, 0.0484856
1: -0.0151127, 0.0027645, -0.0098668, -0.0013331, -0.0137796, 0.0126313
2: -0.0009117, 0.0442011, -0.0019032, 0.0241582, -0.0250698, 0.0461043
3: -0.0233730, 0.0320541, -0.0113350, 0.0098701, -0.0332431, 0.0433891
4: -0.0199650, 0.0218563, -0.0114325, 0.0089282, -0.0288933, 0.0332888
5: 0.9319918, 1.0245681, 0.9883986, 1.0097283, -0.0777366, 0.0361695
6: -0.0276835, 0.0213276, -0.0093428, 0.0131567, -0.0408402, 0.0306704
7: -0.0348597, 0.0110320, -0.0270258, -0.0021114, -0.0327483, 0.0380577
8: -0.0194015, 0.0480453, -0.0090373, 0.0210170, -0.0404185, 0.0570826
9: -0.0157840, 0.0204805, -0.0083956, 0.0073447, -0.0231287, 0.0288761

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B2_A2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308907, upper bound: 0.0305134
time: 1.89 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308767, upper bound: 0.0304250
time: 2.47 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0353472, 0.0423478, -0.0091636, 0.0135421, -0.0488892, 0.0515114
1: -0.0155813, 0.0031567, -0.0102683, -0.0010881, -0.0144932, 0.0134250
2: -0.0016026, 0.0461873, -0.0021842, 0.0251792, -0.0267818, 0.0483715
3: -0.0244482, 0.0341933, -0.0122563, 0.0111552, -0.0356035, 0.0464496
4: -0.0207272, 0.0230462, -0.0120855, 0.0098255, -0.0305526, 0.0351318
5: 0.9258524, 1.0258934, 0.9869628, 1.0108641, -0.0850117, 0.0389307
6: -0.0294264, 0.0220574, -0.0104723, 0.0137821, -0.0432085, 0.0325297
7: -0.0355594, 0.0132382, -0.0276254, -0.0019138, -0.0336456, 0.0408636
8: -0.0203272, 0.0506287, -0.0098306, 0.0226424, -0.0429696, 0.0604593
9: -0.0164439, 0.0216678, -0.0089611, 0.0083132, -0.0247571, 0.0306289

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309103, upper bound: 0.0305134
time: 2.12 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308990, upper bound: 0.0304250
time: 1.90 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.13 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0309300
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0304971
NS_A1_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307342
NS_A1_B1_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0307342
NS_A1_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306342
NS_A1_B1_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0304644
NS_A1_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306614
NS_A1_B1_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306614
NS_A1_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0309810
NS_A1_B1_A1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0305344
NS_A1_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306343
NS_A1_B1_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0301871, upper bound: 0.0304979
NS_A1_B1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306614
NS_A1_B1_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0307340, upper bound: 0.0306614
NS_A1_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0311350
NS_A1_B1_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0306614
NS_A1_B1_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309030, upper bound: 0.0308063
NS_A1_B1_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309030, upper bound: 0.0308063
NS_A1_B1_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
NS_A1_B1_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0308063
NS_A1_B1_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308891, upper bound: 0.0306951
NS_A1_B1_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308891, upper bound: 0.0306951
NS_A1_B1_A2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0300041, upper bound: 0.0303130
NS_A1_B1_A2_A1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0299838, upper bound: 0.0297756
NS_A1_B1_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308776, upper bound: 0.0305470
NS_A1_B1_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
NS_A1_B1_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308776, upper bound: 0.0305470
NS_A1_B1_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308970, upper bound: 0.0305470
NS_A1_B1_A2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308662, upper bound: 0.0304555
NS_A1_B1_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308662, upper bound: 0.0304555
NS_A1_B1_A2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0299950, upper bound: 0.0301970
NS_A1_B1_A2_A2_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0299781, upper bound: 0.0295970
NS_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309810, upper bound: 0.0301871
NS_A2_B1_B1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0305343, upper bound: 0.0301871
NS_A2_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306343, upper bound: 0.0311350
NS_A2_B1_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0304978, upper bound: 0.0301871
NS_A2_B1_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0311350
NS_A2_B1_B1_A1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0307340
NS_A2_B1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306599
NS_A2_B1_B1_A1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306599
NS_A2_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309451, upper bound: 0.0307875
NS_A2_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309274, upper bound: 0.0306869
NS_A2_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309600, upper bound: 0.0307875
NS_A2_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309460, upper bound: 0.0306869
NS_A2_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309221, upper bound: 0.0305134
NS_A2_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309067, upper bound: 0.0304250
NS_A2_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309413, upper bound: 0.0305134
NS_A2_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309288, upper bound: 0.0304250
NS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309808, upper bound: 0.0301871
NS_A2_B1_B2_A1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0305343, upper bound: 0.0301871
NS_A2_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306328, upper bound: 0.0311350
NS_A2_B1_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0304978, upper bound: 0.0301871
NS_A2_B1_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0311350
NS_A2_B1_B2_A1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0307340
NS_A2_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306599
NS_A2_B1_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0306599, upper bound: 0.0306599
NS_A2_B1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309156, upper bound: 0.0307875
NS_A2_B1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308981, upper bound: 0.0306869
NS_A2_B1_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309296, upper bound: 0.0307875
NS_A2_B1_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309157, upper bound: 0.0306869
NS_A2_B1_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308907, upper bound: 0.0305134
NS_A2_B1_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308767, upper bound: 0.0304250
NS_A2_B1_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0309103, upper bound: 0.0305134
NS_A2_B1_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.13
Output dim: 5, lower bound: -0.0308990, upper bound: 0.0304250

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0081930, 0.0093715, -0.0083389, 0.0105990, -0.0187920, 0.0177104
1: -0.0078921, -0.0025679, -0.0085571, -0.0021324, -0.0057597, 0.0059892
2: -0.0001653, 0.0194310, -0.0004687, 0.0208279, -0.0209932, 0.0198997
3: -0.0066920, 0.0043156, -0.0083297, 0.0056783, -0.0123704, 0.0126453
4: -0.0084141, 0.0062154, -0.0093023, 0.0064421, -0.0148562, 0.0155177
5: 0.9931856, 1.0040048, 0.9930813, 1.0060235, -0.0128379, 0.0109235
6: -0.0036509, 0.0101962, -0.0056585, 0.0111168, -0.0147677, 0.0158547
7: -0.0240043, -0.0033337, -0.0250700, -0.0031204, -0.0208839, 0.0217363
8: -0.0065982, 0.0137083, -0.0067643, 0.0157151, -0.0223133, 0.0204726
9: -0.0055460, 0.0033028, -0.0065511, 0.0041856, -0.0097316, 0.0098539

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299527, upper bound: 0.0305656
time: 1.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0298009, upper bound: 0.0305411
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0085389, 0.0108754, -0.0084619, 0.0107531, -0.0192920, 0.0193373
1: -0.0087179, -0.0020343, -0.0086467, -0.0020777, -0.0066402, 0.0066125
2: -0.0008848, 0.0212366, -0.0007246, 0.0210558, -0.0219406, 0.0219612
3: -0.0086985, 0.0061928, -0.0085353, 0.0059652, -0.0146637, 0.0147281
4: -0.0095637, 0.0067531, -0.0094481, 0.0066333, -0.0161970, 0.0162011
5: 0.9925066, 1.0064782, 0.9927609, 1.0062770, -0.0137703, 0.0137173
6: -0.0061107, 0.0113672, -0.0059106, 0.0112564, -0.0173671, 0.0172778
7: -0.0253100, -0.0028277, -0.0252038, -0.0029404, -0.0223696, 0.0223761
8: -0.0069920, 0.0163658, -0.0069043, 0.0160779, -0.0230700, 0.0232702
9: -0.0067775, 0.0045733, -0.0066773, 0.0044018, -0.0111793, 0.0112507

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310190, upper bound: 0.0303547
time: 2.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309596, upper bound: 0.0303547
time: 2.23 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0085085, 0.0122803, -0.0081363, 0.0096544, -0.0181630, 0.0204167
1: -0.0095347, -0.0015358, -0.0080452, -0.0024675, -0.0070672, 0.0065095
2: -0.0008216, 0.0233138, -0.0000474, 0.0197519, -0.0205735, 0.0233612
3: -0.0105730, 0.0088073, -0.0070695, 0.0046273, -0.0152002, 0.0158768
4: -0.0108924, 0.0081862, -0.0086181, 0.0061273, -0.0170196, 0.0168043
5: 0.9895858, 1.0087891, 0.9931681, 1.0044701, -0.0148844, 0.0156210
6: -0.0084086, 0.0126395, -0.0041137, 0.0104079, -0.0188165, 0.0167531
7: -0.0265299, -0.0028721, -0.0242500, -0.0034167, -0.0231132, 0.0213778
8: -0.0083813, 0.0196727, -0.0065337, 0.0141685, -0.0225498, 0.0262063
9: -0.0079279, 0.0065437, -0.0057777, 0.0035040, -0.0114320, 0.0123213

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309719, upper bound: 0.0302936
time: 2.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309144, upper bound: 0.0302884
time: 2.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0086400, 0.0129529, -0.0084546, 0.0106453, -0.0192852, 0.0214075
1: -0.0099257, -0.0012972, -0.0085841, -0.0021159, -0.0078098, 0.0072869
2: -0.0010951, 0.0243081, -0.0007094, 0.0208964, -0.0219915, 0.0250175
3: -0.0114702, 0.0100588, -0.0083915, 0.0057646, -0.0172348, 0.0184503
4: -0.0115283, 0.0090599, -0.0093461, 0.0066220, -0.0181504, 0.0184060
5: 0.9881876, 1.0098951, 0.9929851, 1.0060996, -0.0179120, 0.0169100
6: -0.0095086, 0.0132485, -0.0057343, 0.0111588, -0.0206674, 0.0189828
7: -0.0271138, -0.0026798, -0.0251102, -0.0029511, -0.0241627, 0.0224304
8: -0.0091538, 0.0212556, -0.0068960, 0.0158242, -0.0249780, 0.0281516
9: -0.0084786, 0.0074869, -0.0065890, 0.0042506, -0.0127292, 0.0140759

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309661, upper bound: 0.0304311
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309596, upper bound: 0.0303186
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0081930, 0.0093715, -0.0086872, 0.0125495, -0.0207425, 0.0180587
1: -0.0078921, -0.0025679, -0.0096912, -0.0014403, -0.0064518, 0.0071233
2: -0.0001653, 0.0194310, -0.0011932, 0.0237117, -0.0238770, 0.0206242
3: -0.0066920, 0.0043156, -0.0109321, 0.0093081, -0.0160001, 0.0152477
4: -0.0084141, 0.0062154, -0.0111469, 0.0085359, -0.0169500, 0.0173623
5: 0.9931856, 1.0040048, 0.9890262, 1.0092316, -0.0160460, 0.0149786
6: -0.0036509, 0.0101962, -0.0088488, 0.0128832, -0.0165341, 0.0190450
7: -0.0240043, -0.0033337, -0.0267636, -0.0026108, -0.0213935, 0.0234298
8: -0.0065982, 0.0137083, -0.0086905, 0.0203062, -0.0269044, 0.0223988
9: -0.0055460, 0.0033028, -0.0081483, 0.0069212, -0.0124672, 0.0114511

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0298009, upper bound: 0.0306542
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0298009, upper bound: 0.0306045
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0085085, 0.0122803, -0.0084779, 0.0115839, -0.0200924, 0.0207582
1: -0.0095347, -0.0015358, -0.0091298, -0.0017829, -0.0077518, 0.0075940
2: -0.0008216, 0.0233138, -0.0007578, 0.0222841, -0.0231058, 0.0240716
3: -0.0105730, 0.0088073, -0.0096438, 0.0075113, -0.0180843, 0.0184511
4: -0.0108924, 0.0081862, -0.0102337, 0.0072814, -0.0181737, 0.0184199
5: 0.9895858, 1.0087891, 0.9910337, 1.0076435, -0.0180577, 0.0177554
6: -0.0084086, 0.0126395, -0.0072695, 0.0120088, -0.0204175, 0.0199090
7: -0.0265299, -0.0028721, -0.0259252, -0.0029170, -0.0236129, 0.0230531
8: -0.0083813, 0.0196727, -0.0075813, 0.0180335, -0.0264148, 0.0272540
9: -0.0079279, 0.0065437, -0.0073576, 0.0055670, -0.0134949, 0.0139013

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308829, upper bound: 0.0303551
time: 2.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308789, upper bound: 0.0302904
time: 2.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0087592, 0.0122301, -0.0085900, 0.0112082, -0.0199674, 0.0208201
1: -0.0095055, -0.0015536, -0.0089113, -0.0019162, -0.0075893, 0.0073577
2: -0.0013431, 0.0232395, -0.0009909, 0.0217286, -0.0230717, 0.0242305
3: -0.0105059, 0.0087138, -0.0091425, 0.0068121, -0.0173180, 0.0178563
4: -0.0108449, 0.0081209, -0.0098784, 0.0068324, -0.0176773, 0.0179994
5: 0.9896902, 1.0087065, 0.9918148, 1.0070256, -0.0173354, 0.0168917
6: -0.0083265, 0.0125940, -0.0066550, 0.0116686, -0.0199950, 0.0192490
7: -0.0264863, -0.0025053, -0.0255990, -0.0027530, -0.0237332, 0.0230936
8: -0.0083236, 0.0195545, -0.0071497, 0.0171491, -0.0254727, 0.0267042
9: -0.0078868, 0.0064732, -0.0070500, 0.0050401, -0.0129269, 0.0135232

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310143, upper bound: 0.0303186
time: 2.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309497, upper bound: 0.0303186
time: 2.38 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0085328, 0.0113093, -0.0107002, 0.0140872, -0.0226200, 0.0220094
1: -0.0089701, -0.0018803, -0.0105853, -0.0008947, -0.0080755, 0.0087049
2: -0.0008722, 0.0218781, -0.0016349, 0.0259852, -0.0268574, 0.0235130
3: -0.0092774, 0.0070002, -0.0129836, 0.0121697, -0.0214470, 0.0199839
4: -0.0099740, 0.0069245, -0.0126011, 0.0105337, -0.0205078, 0.0195256
5: 0.9916046, 1.0071917, 0.9858294, 1.0117606, -0.0201560, 0.0213622
6: -0.0068203, 0.0117601, -0.0113639, 0.0142757, -0.0210961, 0.0231240
7: -0.0256868, -0.0028366, -0.0280987, -0.0023002, -0.0233866, 0.0252621
8: -0.0072659, 0.0173871, -0.0104568, 0.0239255, -0.0311914, 0.0278438
9: -0.0071328, 0.0051818, -0.0094075, 0.0090777, -0.0162104, 0.0145893

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0304274, upper bound: 0.0309721
time: 1.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0303186, upper bound: 0.0309658
time: 2.18 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0221107, 0.0180653, -0.0086243, 0.0111135, -0.0332241, 0.0266896
1: -0.0128982, 0.0005169, -0.0088563, -0.0019498, -0.0109484, 0.0093731
2: 0.0004571, 0.0318667, -0.0010625, 0.0215886, -0.0211314, 0.0329292
3: -0.0182913, 0.0195726, -0.0090161, 0.0066358, -0.0249271, 0.0285887
4: -0.0163631, 0.0157024, -0.0097888, 0.0068859, -0.0232490, 0.0254912
5: 0.9775592, 1.0183036, 0.9920117, 1.0068698, -0.0293106, 0.0262920
6: -0.0178706, 0.0178783, -0.0065000, 0.0115828, -0.0294534, 0.0243784
7: -0.0315527, -0.0031865, -0.0255167, -0.0027027, -0.0288499, 0.0223302
8: -0.0150264, 0.0332890, -0.0070893, 0.0169261, -0.0319525, 0.0403783
9: -0.0126650, 0.0146568, -0.0069724, 0.0049072, -0.0175722, 0.0216292

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0296555, upper bound: 0.0290299
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293429, upper bound: 0.0290226
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0221107, 0.0180653, -0.0089752, 0.0129851, -0.0350958, 0.0270405
1: -0.0128982, 0.0005169, -0.0099445, -0.0012857, -0.0116125, 0.0104613
2: 0.0004571, 0.0318667, -0.0017924, 0.0243557, -0.0238986, 0.0336591
3: -0.0182913, 0.0195726, -0.0115132, 0.0101187, -0.0284100, 0.0310858
4: -0.0163631, 0.0157024, -0.0115588, 0.0091018, -0.0254649, 0.0272612
5: 0.9775592, 1.0183036, 0.9881207, 1.0099480, -0.0323888, 0.0301829
6: -0.0178706, 0.0178783, -0.0095613, 0.0132777, -0.0311483, 0.0274396
7: -0.0315527, -0.0031865, -0.0271418, -0.0021893, -0.0293634, 0.0239552
8: -0.0150264, 0.0332890, -0.0091908, 0.0213314, -0.0363578, 0.0424798
9: -0.0126650, 0.0146568, -0.0085050, 0.0075320, -0.0201971, 0.0231618

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0296555, upper bound: 0.0290294
time: 1.72 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293429, upper bound: 0.0290230
time: 1.78 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0248501, 0.0190204, -0.0087578, 0.0117971, -0.0366473, 0.0277781
1: -0.0134535, 0.0008557, -0.0092538, -0.0017072, -0.0117463, 0.0101095
2: -0.0002320, 0.0332787, -0.0013401, 0.0225994, -0.0228314, 0.0346189
3: -0.0195655, 0.0213499, -0.0099283, 0.0079081, -0.0274736, 0.0312782
4: -0.0172663, 0.0169432, -0.0104354, 0.0075584, -0.0248247, 0.0273787
5: 0.9755737, 1.0198746, 0.9905904, 1.0079944, -0.0324208, 0.0292842
6: -0.0194327, 0.0187432, -0.0076183, 0.0122019, -0.0316346, 0.0263615
7: -0.0323819, -0.0017835, -0.0261104, -0.0025074, -0.0298745, 0.0243268
8: -0.0161234, 0.0355370, -0.0078263, 0.0185354, -0.0346588, 0.0433632
9: -0.0134471, 0.0159962, -0.0075323, 0.0058660, -0.0193131, 0.0235285

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0303863, upper bound: 0.0298529
time: 2.79 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299931, upper bound: 0.0298468
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0248501, 0.0190204, -0.0095271, 0.0136782, -0.0385284, 0.0285475
1: -0.0134535, 0.0008557, -0.0103475, -0.0010398, -0.0124137, 0.0112032
2: -0.0002320, 0.0332787, -0.0020745, 0.0253805, -0.0256125, 0.0353532
3: -0.0195655, 0.0213499, -0.0124380, 0.0114086, -0.0309742, 0.0337880
4: -0.0172663, 0.0169432, -0.0122143, 0.0100024, -0.0272687, 0.0291576
5: 0.9755737, 1.0198746, 0.9866797, 1.0110880, -0.0355144, 0.0331948
6: -0.0194327, 0.0187432, -0.0106950, 0.0139054, -0.0333381, 0.0294382
7: -0.0323819, -0.0017835, -0.0277436, -0.0019909, -0.0303910, 0.0259601
8: -0.0161234, 0.0355370, -0.0099870, 0.0229630, -0.0390864, 0.0455240
9: -0.0134471, 0.0159962, -0.0090726, 0.0085042, -0.0219513, 0.0250688

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0303863, upper bound: 0.0298532
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299931, upper bound: 0.0298488
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0281136, 0.0201581, -0.0086186, 0.0110088, -0.0391224, 0.0287767
1: -0.0141150, 0.0012594, -0.0087954, -0.0019869, -0.0121281, 0.0100549
2: 0.0002806, 0.0349608, -0.0010506, 0.0214339, -0.0211533, 0.0360114
3: -0.0210835, 0.0234672, -0.0088765, 0.0064411, -0.0275247, 0.0323437
4: -0.0183422, 0.0184215, -0.0096899, 0.0068770, -0.0252193, 0.0281114
5: 0.9732083, 1.0217458, 0.9922292, 1.0066974, -0.0334891, 0.0295166
6: -0.0212936, 0.0197736, -0.0063289, 0.0114880, -0.0327817, 0.0261025
7: -0.0333698, -0.0001122, -0.0254259, -0.0027111, -0.0306587, 0.0253137
8: -0.0174303, 0.0382149, -0.0070828, 0.0166799, -0.0341103, 0.0452977
9: -0.0143788, 0.0175919, -0.0068867, 0.0047605, -0.0191393, 0.0244786

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295156, upper bound: 0.0288413
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0291942, upper bound: 0.0288411
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0281136, 0.0201581, -0.0089696, 0.0128709, -0.0409845, 0.0291277
1: -0.0141150, 0.0012594, -0.0098781, -0.0013262, -0.0127888, 0.0111375
2: 0.0002806, 0.0349608, -0.0017808, 0.0241869, -0.0239063, 0.0367416
3: -0.0210835, 0.0234672, -0.0113609, 0.0099063, -0.0309898, 0.0348281
4: -0.0183422, 0.0184215, -0.0114508, 0.0089535, -0.0272957, 0.0298723
5: 0.9732083, 1.0217458, 0.9883581, 1.0097603, -0.0365520, 0.0333877
6: -0.0212936, 0.0197736, -0.0093746, 0.0131743, -0.0344680, 0.0291481
7: -0.0333698, -0.0001122, -0.0270426, -0.0021975, -0.0311723, 0.0269304
8: -0.0174303, 0.0382149, -0.0090597, 0.0210628, -0.0384931, 0.0472746
9: -0.0143788, 0.0175919, -0.0084115, 0.0073719, -0.0217507, 0.0260034

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295156, upper bound: 0.0288413
time: 1.89 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0291942, upper bound: 0.0288411
time: 1.58 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0218922, 0.0179891, -0.0086206, 0.0105763, -0.0324686, 0.0266097
1: -0.0128539, 0.0004898, -0.0085443, -0.0021404, -0.0107135, 0.0090342
2: 0.0003627, 0.0317541, -0.0010547, 0.0207974, -0.0204347, 0.0328088
3: -0.0181897, 0.0194309, -0.0082994, 0.0056427, -0.0238324, 0.0277303
4: -0.0162911, 0.0156034, -0.0092828, 0.0068800, -0.0231711, 0.0248862
5: 0.9777174, 1.0181785, 0.9931110, 1.0059861, -0.0282687, 0.0250674
6: -0.0177460, 0.0178094, -0.0056215, 0.0110977, -0.0288437, 0.0234308
7: -0.0314866, -0.0032984, -0.0250504, -0.0027082, -0.0287784, 0.0217520
8: -0.0149389, 0.0331097, -0.0070850, 0.0156681, -0.0306070, 0.0401948
9: -0.0126027, 0.0145500, -0.0065325, 0.0041598, -0.0167625, 0.0210825

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293573, upper bound: 0.0286762
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0290177, upper bound: 0.0286760
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0244859, 0.0188934, -0.0087541, 0.0112609, -0.0357469, 0.0276475
1: -0.0133797, 0.0008107, -0.0089420, -0.0018975, -0.0114822, 0.0097527
2: -0.0003327, 0.0330910, -0.0013324, 0.0218067, -0.0221393, 0.0344234
3: -0.0193961, 0.0211136, -0.0092129, 0.0069103, -0.0263064, 0.0303265
4: -0.0171462, 0.0167783, -0.0099283, 0.0070876, -0.0242338, 0.0267066
5: 0.9758376, 1.0196655, 0.9917050, 1.0071125, -0.0312749, 0.0279605
6: -0.0192251, 0.0186283, -0.0067413, 0.0117163, -0.0309414, 0.0253696
7: -0.0322717, -0.0019701, -0.0256448, -0.0025129, -0.0297588, 0.0236747
8: -0.0159776, 0.0352381, -0.0072371, 0.0172733, -0.0332509, 0.0424752
9: -0.0133431, 0.0158182, -0.0070932, 0.0051140, -0.0184572, 0.0229113

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304227, upper bound: 0.0296426
time: 2.28 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299855, upper bound: 0.0296426
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0218922, 0.0179891, -0.0089709, 0.0124641, -0.0343563, 0.0269600
1: -0.0128539, 0.0004898, -0.0096415, -0.0014706, -0.0113834, 0.0101314
2: 0.0003627, 0.0317541, -0.0017834, 0.0235854, -0.0232227, 0.0335375
3: -0.0181897, 0.0194309, -0.0108181, 0.0091492, -0.0273389, 0.0302490
4: -0.0162911, 0.0156034, -0.0110661, 0.0084249, -0.0247159, 0.0266695
5: 0.9777174, 1.0181785, 0.9892040, 1.0090913, -0.0313739, 0.0289745
6: -0.0177460, 0.0178094, -0.0087091, 0.0128059, -0.0305519, 0.0265185
7: -0.0314866, -0.0032984, -0.0266894, -0.0021957, -0.0292909, 0.0233910
8: -0.0149389, 0.0331097, -0.0085923, 0.0201051, -0.0350440, 0.0417021
9: -0.0126027, 0.0145500, -0.0080784, 0.0068013, -0.0194040, 0.0226284

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293402, upper bound: 0.0286762
time: 1.74 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0290149, upper bound: 0.0286762
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0244859, 0.0188934, -0.0091066, 0.0131553, -0.0376412, 0.0280000
1: -0.0133797, 0.0008107, -0.0100434, -0.0012253, -0.0121543, 0.0108541
2: -0.0003327, 0.0330910, -0.0020657, 0.0246073, -0.0249400, 0.0351567
3: -0.0193961, 0.0211136, -0.0117403, 0.0104354, -0.0298316, 0.0328539
4: -0.0171462, 0.0167783, -0.0117198, 0.0093229, -0.0264691, 0.0284980
5: 0.9758376, 1.0196655, 0.9877669, 1.0102280, -0.0343904, 0.0318986
6: -0.0192251, 0.0186283, -0.0098397, 0.0134318, -0.0326569, 0.0284679
7: -0.0322717, -0.0019701, -0.0272895, -0.0019972, -0.0302745, 0.0253195
8: -0.0159776, 0.0352381, -0.0093863, 0.0217320, -0.0377096, 0.0446244
9: -0.0133431, 0.0158182, -0.0086444, 0.0077707, -0.0211139, 0.0244625

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0303722, upper bound: 0.0296426
time: 1.94 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299839, upper bound: 0.0296426
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0276122, 0.0199833, -0.0086149, 0.0104721, -0.0380842, 0.0285982
1: -0.0140134, 0.0011974, -0.0084879, -0.0021774, -0.0118360, 0.0096853
2: 0.0001922, 0.0347024, -0.0010428, 0.0206792, -0.0204870, 0.0357452
3: -0.0208503, 0.0231419, -0.0081604, 0.0055279, -0.0263782, 0.0313023
4: -0.0181769, 0.0181944, -0.0092076, 0.0068712, -0.0250481, 0.0274020
5: 0.9735717, 1.0214581, 0.9931176, 1.0058149, -0.0322433, 0.0283406
6: -0.0210078, 0.0196153, -0.0054510, 0.0110197, -0.0320274, 0.0250662
7: -0.0332180, -0.0003690, -0.0249599, -0.0027165, -0.0305014, 0.0245909
8: -0.0172295, 0.0378035, -0.0070786, 0.0154985, -0.0327281, 0.0448821
9: -0.0142357, 0.0173467, -0.0064472, 0.0040857, -0.0183213, 0.0237939

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0286001, upper bound: 0.0282030
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0282776, upper bound: 0.0282030
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0276122, 0.0199833, -0.0089653, 0.0123529, -0.0399651, 0.0289487
1: -0.0140134, 0.0011974, -0.0095769, -0.0015100, -0.0125034, 0.0107743
2: 0.0001922, 0.0347024, -0.0017719, 0.0234211, -0.0232289, 0.0364743
3: -0.0208503, 0.0231419, -0.0106698, 0.0089423, -0.0297926, 0.0338117
4: -0.0181769, 0.0181944, -0.0109610, 0.0082805, -0.0264574, 0.0291554
5: 0.9735717, 1.0214581, 0.9894350, 1.0089083, -0.0353366, 0.0320231
6: -0.0210078, 0.0196153, -0.0085273, 0.0127052, -0.0337130, 0.0281426
7: -0.0332180, -0.0003690, -0.0265929, -0.0022038, -0.0310142, 0.0262239
8: -0.0172295, 0.0378035, -0.0084646, 0.0198435, -0.0370730, 0.0462682
9: -0.0142357, 0.0173467, -0.0079874, 0.0066455, -0.0208811, 0.0253341

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_A2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0286001, upper bound: 0.0282030
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0282776, upper bound: 0.0282030
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0086872, 0.0125495, -0.0081930, 0.0093715, -0.0180587, 0.0207425
1: -0.0096912, -0.0014403, -0.0078921, -0.0025679, -0.0071233, 0.0064518
2: -0.0011932, 0.0237117, -0.0001653, 0.0194310, -0.0206242, 0.0238770
3: -0.0109321, 0.0093081, -0.0066920, 0.0043156, -0.0152477, 0.0160001
4: -0.0111469, 0.0085359, -0.0084141, 0.0062154, -0.0173623, 0.0169500
5: 0.9890262, 1.0092316, 0.9931856, 1.0040048, -0.0149786, 0.0160460
6: -0.0088488, 0.0128832, -0.0036509, 0.0101962, -0.0190450, 0.0165341
7: -0.0267636, -0.0026108, -0.0240043, -0.0033337, -0.0234298, 0.0213935
8: -0.0086905, 0.0203062, -0.0065982, 0.0137083, -0.0223988, 0.0269044
9: -0.0081483, 0.0069212, -0.0055460, 0.0033028, -0.0114511, 0.0124672

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306542, upper bound: 0.0298009
time: 1.87 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306045, upper bound: 0.0298009
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0084779, 0.0115839, -0.0085085, 0.0122803, -0.0207582, 0.0200924
1: -0.0091298, -0.0017829, -0.0095347, -0.0015358, -0.0075940, 0.0077518
2: -0.0007578, 0.0222841, -0.0008216, 0.0233138, -0.0240716, 0.0231058
3: -0.0096438, 0.0075113, -0.0105730, 0.0088073, -0.0184511, 0.0180843
4: -0.0102337, 0.0072814, -0.0108924, 0.0081862, -0.0184199, 0.0181737
5: 0.9910337, 1.0076435, 0.9895858, 1.0087891, -0.0177554, 0.0180577
6: -0.0072695, 0.0120088, -0.0084086, 0.0126395, -0.0199090, 0.0204175
7: -0.0259252, -0.0029170, -0.0265299, -0.0028721, -0.0230531, 0.0236129
8: -0.0075813, 0.0180335, -0.0083813, 0.0196727, -0.0272540, 0.0264148
9: -0.0073576, 0.0055670, -0.0079279, 0.0065437, -0.0139013, 0.0134949

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0303551, upper bound: 0.0308829
time: 2.10 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0302904, upper bound: 0.0308789
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0085900, 0.0112082, -0.0087592, 0.0122301, -0.0208201, 0.0199674
1: -0.0089113, -0.0019162, -0.0095055, -0.0015536, -0.0073577, 0.0075893
2: -0.0009909, 0.0217286, -0.0013431, 0.0232395, -0.0242305, 0.0230717
3: -0.0091425, 0.0068121, -0.0105059, 0.0087138, -0.0178563, 0.0173180
4: -0.0098784, 0.0068324, -0.0108449, 0.0081209, -0.0179994, 0.0176773
5: 0.9918148, 1.0070256, 0.9896902, 1.0087065, -0.0168917, 0.0173354
6: -0.0066550, 0.0116686, -0.0083265, 0.0125940, -0.0192490, 0.0199950
7: -0.0255990, -0.0027530, -0.0264863, -0.0025053, -0.0230936, 0.0237332
8: -0.0071497, 0.0171491, -0.0083236, 0.0195545, -0.0267042, 0.0254727
9: -0.0070500, 0.0050401, -0.0078868, 0.0064732, -0.0135232, 0.0129269

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B1_A1_A2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0303186, upper bound: 0.0310143
time: 2.00 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0303186, upper bound: 0.0309497
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0107002, 0.0140872, -0.0085328, 0.0113093, -0.0220094, 0.0226200
1: -0.0105853, -0.0008947, -0.0089701, -0.0018803, -0.0087049, 0.0080755
2: -0.0016349, 0.0259852, -0.0008722, 0.0218781, -0.0235130, 0.0268574
3: -0.0129836, 0.0121697, -0.0092774, 0.0070002, -0.0199839, 0.0214470
4: -0.0126011, 0.0105337, -0.0099740, 0.0069245, -0.0195256, 0.0205078
5: 0.9858294, 1.0117606, 0.9916046, 1.0071917, -0.0213622, 0.0201560
6: -0.0113639, 0.0142757, -0.0068203, 0.0117601, -0.0231240, 0.0210961
7: -0.0280987, -0.0023002, -0.0256868, -0.0028366, -0.0252621, 0.0233866
8: -0.0104568, 0.0239255, -0.0072659, 0.0173871, -0.0278438, 0.0311914
9: -0.0094075, 0.0090777, -0.0071328, 0.0051818, -0.0145893, 0.0162104

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B1_A1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309721, upper bound: 0.0304274
time: 2.02 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309658, upper bound: 0.0303186
time: 5.15 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0278662, 0.0329923, -0.0086243, 0.0111135, -0.0389796, 0.0416166
1: -0.0140649, 0.0018874, -0.0088563, -0.0019498, -0.0121150, 0.0107436
2: -0.0002546, 0.0397589, -0.0010625, 0.0215886, -0.0218432, 0.0408214
3: -0.0209684, 0.0272699, -0.0090161, 0.0066358, -0.0276042, 0.0362860
4: -0.0182606, 0.0191951, -0.0097888, 0.0068859, -0.0251466, 0.0289839
5: 0.9457223, 1.0216038, 0.9920117, 1.0068698, -0.0611475, 0.0295922
6: -0.0237855, 0.0196954, -0.0065000, 0.0115828, -0.0353682, 0.0261955
7: -0.0332949, 0.0060977, -0.0255167, -0.0027027, -0.0305921, 0.0316144
8: -0.0173312, 0.0422674, -0.0070893, 0.0169261, -0.0342574, 0.0493568
9: -0.0143081, 0.0178250, -0.0069724, 0.0049072, -0.0192153, 0.0247975

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0297966, upper bound: 0.0302315
time: 2.01 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0294682, upper bound: 0.0290545
time: 1.53 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0333311, 0.0398265, -0.0086186, 0.0110088, -0.0443399, 0.0484451
1: -0.0151726, 0.0028146, -0.0087954, -0.0019869, -0.0131857, 0.0116101
2: -0.0003799, 0.0444549, -0.0010506, 0.0214339, -0.0218138, 0.0455055
3: -0.0235104, 0.0323274, -0.0088765, 0.0064411, -0.0299515, 0.0412040
4: -0.0200624, 0.0220084, -0.0096899, 0.0068770, -0.0269394, 0.0316983
5: 0.9312072, 1.0247372, 0.9922292, 1.0066974, -0.0754902, 0.0325080
6: -0.0279062, 0.0214208, -0.0063289, 0.0114880, -0.0393942, 0.0277497
7: -0.0349491, 0.0113139, -0.0254259, -0.0027111, -0.0322380, 0.0367398
8: -0.0195198, 0.0483754, -0.0070828, 0.0166799, -0.0361997, 0.0554582
9: -0.0158683, 0.0206322, -0.0068867, 0.0047605, -0.0206288, 0.0275189

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0297702, upper bound: 0.0301777
time: 2.00 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0293381, upper bound: 0.0288733
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0302347, 0.0359544, -0.0087578, 0.0117971, -0.0420319, 0.0447122
1: -0.0145450, 0.0022893, -0.0092538, -0.0017072, -0.0128378, 0.0115431
2: -0.0009332, 0.0417943, -0.0013401, 0.0225994, -0.0235326, 0.0431344
3: -0.0220702, 0.0294620, -0.0099283, 0.0079081, -0.0299783, 0.0393903
4: -0.0190416, 0.0204144, -0.0104354, 0.0075584, -0.0265999, 0.0308498
5: 0.9394310, 1.0229620, 0.9905904, 1.0079944, -0.0685634, 0.0323716
6: -0.0255715, 0.0204433, -0.0076183, 0.0122019, -0.0377734, 0.0280616
7: -0.0340118, 0.0083585, -0.0261104, -0.0025074, -0.0315044, 0.0344689
8: -0.0182798, 0.0449148, -0.0078263, 0.0185354, -0.0368152, 0.0527411
9: -0.0149843, 0.0190417, -0.0075323, 0.0058660, -0.0208503, 0.0265740

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300228, upper bound: 0.0303306
time: 1.87 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300121, upper bound: 0.0298468
time: 2.12 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0356282, 0.0426993, -0.0087515, 0.0116922, -0.0473204, 0.0514508
1: -0.0156383, 0.0032044, -0.0091928, -0.0017445, -0.0138938, 0.0123972
2: -0.0010677, 0.0464288, -0.0013270, 0.0224443, -0.0235120, 0.0477558
3: -0.0245790, 0.0344534, -0.0097883, 0.0077128, -0.0322918, 0.0442417
4: -0.0208198, 0.0231909, -0.0103362, 0.0074221, -0.0282419, 0.0335271
5: 0.9251059, 1.0260545, 0.9908085, 1.0078217, -0.0827157, 0.0352460
6: -0.0296383, 0.0221461, -0.0074467, 0.0121069, -0.0417452, 0.0295928
7: -0.0356445, 0.0135065, -0.0260192, -0.0025167, -0.0331278, 0.0395258
8: -0.0204397, 0.0509429, -0.0077057, 0.0182884, -0.0387281, 0.0586486
9: -0.0165241, 0.0218122, -0.0074463, 0.0057189, -0.0222430, 0.0292585

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300120, upper bound: 0.0302806
time: 2.01 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300007, upper bound: 0.0297756
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0278674, 0.0329939, -0.0086206, 0.0105763, -0.0384437, 0.0416144
1: -0.0140651, 0.0018876, -0.0085443, -0.0021404, -0.0119247, 0.0104319
2: -0.0003455, 0.0397600, -0.0010547, 0.0207974, -0.0211428, 0.0408147
3: -0.0209690, 0.0272711, -0.0082994, 0.0056427, -0.0266118, 0.0355706
4: -0.0182611, 0.0191957, -0.0092828, 0.0068800, -0.0251411, 0.0284785
5: 0.9457190, 1.0216045, 0.9931110, 1.0059861, -0.0602671, 0.0284935
6: -0.0237864, 0.0196958, -0.0056215, 0.0110977, -0.0348841, 0.0253173
7: -0.0332952, 0.0060989, -0.0250504, -0.0027082, -0.0305870, 0.0311493
8: -0.0173318, 0.0422689, -0.0070850, 0.0156681, -0.0329999, 0.0493539
9: -0.0143085, 0.0178257, -0.0065325, 0.0041598, -0.0184683, 0.0243582

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0296124, upper bound: 0.0287618
time: 1.71 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0292508, upper bound: 0.0287618
time: 1.45 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0329356, 0.0393320, -0.0086149, 0.0104721, -0.0434077, 0.0479469
1: -0.0150925, 0.0027475, -0.0084879, -0.0021774, -0.0129151, 0.0112354
2: -0.0004741, 0.0441151, -0.0010428, 0.0206792, -0.0211533, 0.0451579
3: -0.0233265, 0.0319615, -0.0081604, 0.0055279, -0.0288544, 0.0401219
4: -0.0199320, 0.0218048, -0.0092076, 0.0068712, -0.0268032, 0.0310125
5: 0.9322576, 1.0245105, 0.9931176, 1.0058149, -0.0735573, 0.0313929
6: -0.0276080, 0.0212960, -0.0054510, 0.0110197, -0.0386277, 0.0267469
7: -0.0348294, 0.0109365, -0.0249599, -0.0027165, -0.0321129, 0.0358963
8: -0.0193614, 0.0479334, -0.0070786, 0.0154985, -0.0348599, 0.0550120
9: -0.0157554, 0.0204291, -0.0064472, 0.0040857, -0.0198410, 0.0268763

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0292713, upper bound: 0.0284731
time: 1.60 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0288881, upper bound: 0.0284731
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0302129, 0.0359272, -0.0087541, 0.0112609, -0.0414739, 0.0446813
1: -0.0145406, 0.0022856, -0.0089420, -0.0018975, -0.0126431, 0.0112276
2: -0.0010321, 0.0417756, -0.0013324, 0.0218067, -0.0228388, 0.0431079
3: -0.0220601, 0.0294418, -0.0092129, 0.0069103, -0.0289703, 0.0386548
4: -0.0190344, 0.0204032, -0.0099283, 0.0070876, -0.0261220, 0.0303315
5: 0.9394889, 1.0229495, 0.9917050, 1.0071125, -0.0676236, 0.0312445
6: -0.0255551, 0.0204364, -0.0067413, 0.0117163, -0.0372714, 0.0271777
7: -0.0340053, 0.0083377, -0.0256448, -0.0025129, -0.0314923, 0.0339825
8: -0.0182711, 0.0448905, -0.0072371, 0.0172733, -0.0355444, 0.0521275
9: -0.0149781, 0.0190305, -0.0070932, 0.0051140, -0.0200922, 0.0261237

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300121, upper bound: 0.0301989
time: 2.34 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300047, upper bound: 0.0296426
time: 2.43 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0351901, 0.0421515, -0.0087478, 0.0111567, -0.0463468, 0.0508992
1: -0.0155495, 0.0031301, -0.0088814, -0.0019345, -0.0136150, 0.0120115
2: -0.0011704, 0.0460523, -0.0013193, 0.0216526, -0.0228230, 0.0473716
3: -0.0243752, 0.0340479, -0.0090738, 0.0067163, -0.0310915, 0.0431218
4: -0.0206754, 0.0229654, -0.0098298, 0.0070778, -0.0277531, 0.0327952
5: 0.9262696, 1.0258031, 0.9919218, 1.0069410, -0.0806714, 0.0338812
6: -0.0293080, 0.0220078, -0.0065708, 0.0116219, -0.0409299, 0.0285786
7: -0.0355118, 0.0130883, -0.0255543, -0.0025221, -0.0329897, 0.0386426
8: -0.0202643, 0.0504532, -0.0072299, 0.0170280, -0.0372922, 0.0576831
9: -0.0163990, 0.0215871, -0.0070078, 0.0049679, -0.0213669, 0.0285950

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 74
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300031, upper bound: 0.0301583
time: 2.07 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299952, upper bound: 0.0295970
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0086872, 0.0125495, -0.0085454, 0.0111904, -0.0198776, 0.0210949
1: -0.0096912, -0.0014403, -0.0089010, -0.0019225, -0.0077687, 0.0074607
2: -0.0011932, 0.0237117, -0.0008983, 0.0217024, -0.0228956, 0.0246100
3: -0.0109321, 0.0093081, -0.0091188, 0.0067790, -0.0177111, 0.0184269
4: -0.0111469, 0.0085359, -0.0098616, 0.0067701, -0.0179169, 0.0183975
5: 0.9890262, 1.0092316, 0.9918517, 1.0069964, -0.0179701, 0.0173799
6: -0.0088488, 0.0128832, -0.0066259, 0.0116525, -0.0205013, 0.0195092
7: -0.0267636, -0.0026108, -0.0255836, -0.0028182, -0.0239454, 0.0229727
8: -0.0086905, 0.0203062, -0.0071293, 0.0171073, -0.0257978, 0.0274355
9: -0.0081483, 0.0069212, -0.0070354, 0.0050151, -0.0131635, 0.0139566

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 74
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306236, upper bound: 0.0299527
time: 1.97 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306039, upper bound: 0.0298009
time: 2.06 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.07 + 596.46 = 600.53 seconds
