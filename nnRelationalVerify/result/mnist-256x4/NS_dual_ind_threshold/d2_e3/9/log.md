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
execution time: IAR + RelationalAnalysis = 2.19 + 2.30 = 4.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.25 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.71
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.71
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

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.52 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.45 seconds

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

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.16 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.03 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350

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

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

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
time: 1.48 seconds

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.09 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0140331, 0.0156932, -0.0090002, 0.0133130, -0.0273461, 0.0246934
1: -0.0112609, -0.0004597, -0.0101351, -0.0011694, -0.0100915, 0.0096754
2: -0.0025816, 0.0278724, -0.0018444, 0.0248405, -0.0274220, 0.0297168
3: -0.0145340, 0.0144683, -0.0119507, 0.0107289, -0.0252629, 0.0264189
4: -0.0137000, 0.0120739, -0.0118689, 0.0095278, -0.0232277, 0.0239428
5: 0.9824629, 1.0136719, 0.9874392, 1.0104873, -0.0280244, 0.0262327
6: -0.0133550, 0.0153281, -0.0100976, 0.0135746, -0.0269296, 0.0254256
7: -0.0291076, -0.0016343, -0.0274265, -0.0021527, -0.0269548, 0.0257921
8: -0.0117915, 0.0268068, -0.0095674, 0.0221032, -0.0338947, 0.0363743
9: -0.0103590, 0.0107195, -0.0087735, 0.0079918, -0.0183509, 0.0194930

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 2.32 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0140331, 0.0156932, -0.0140331, 0.0156932, -0.0297263, 0.0297263
1: -0.0112609, -0.0004597, -0.0112609, -0.0004597, -0.0108011, 0.0108011
2: -0.0025816, 0.0278724, -0.0025816, 0.0278724, -0.0304540, 0.0304540
3: -0.0145340, 0.0144683, -0.0145340, 0.0144683, -0.0290023, 0.0290023
4: -0.0137000, 0.0120739, -0.0137000, 0.0120739, -0.0257739, 0.0257739
5: 0.9824629, 1.0136719, 0.9824629, 1.0136719, -0.0312089, 0.0312089
6: -0.0133550, 0.0153281, -0.0133550, 0.0153281, -0.0286831, 0.0286831
7: -0.0291076, -0.0016343, -0.0291076, -0.0016343, -0.0274732, 0.0274732
8: -0.0117915, 0.0268068, -0.0117915, 0.0268068, -0.0385984, 0.0385984
9: -0.0103590, 0.0107195, -0.0103590, 0.0107195, -0.0210785, 0.0210785

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 1.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.87 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.87
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.87
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.87
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.87
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.87
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.87
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.87
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.87
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

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.48 seconds

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

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0084472, 0.0110938, -0.0434992, 0.0525426, -0.0609898, 0.0545930
1: -0.0088448, -0.0019568, -0.0172337, 0.0045399, -0.0133847, 0.0152769
2: -0.0006941, 0.0215595, -0.0015170, 0.0531922, -0.0538863, 0.0230765
3: -0.0089898, 0.0065991, -0.0282401, 0.0417376, -0.0507274, 0.0348393
4: -0.0097702, 0.0066445, -0.0234149, 0.0272429, -0.0370131, 0.0300593
5: 0.9920526, 1.0068376, 0.9042003, 1.0305678, -0.0385152, 0.1026373
6: -0.0064678, 0.0115649, -0.0355733, 0.0246311, -0.0310989, 0.0471382
7: -0.0254996, -0.0029618, -0.0380270, 0.0210192, -0.0465189, 0.0350652
8: -0.0070183, 0.0168798, -0.0235918, 0.0597400, -0.0667583, 0.0404716
9: -0.0069563, 0.0048796, -0.0187711, 0.0258552, -0.0328115, 0.0236507

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087879, 0.0120854, -0.0454844, 0.0550252, -0.0638131, 0.0575698
1: -0.0094214, -0.0016049, -0.0176361, 0.0048768, -0.0142981, 0.0160312
2: -0.0014028, 0.0230256, -0.0018016, 0.0548981, -0.0563009, 0.0248272
3: -0.0103129, 0.0084445, -0.0291636, 0.0435748, -0.0538877, 0.0376081
4: -0.0107080, 0.0079329, -0.0240694, 0.0282649, -0.0389729, 0.0320023
5: 0.9899911, 1.0084684, 0.8989277, 1.0317062, -0.0417151, 0.1095407
6: -0.0080898, 0.0124630, -0.0370702, 0.0252579, -0.0333477, 0.0495331
7: -0.0263606, -0.0024634, -0.0386280, 0.0229141, -0.0492747, 0.0361646
8: -0.0081574, 0.0192139, -0.0243868, 0.0619588, -0.0701161, 0.0436007
9: -0.0077683, 0.0062703, -0.0193379, 0.0268750, -0.0346433, 0.0256082

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0121589, 0.0145958, -0.0090002, 0.0133130, -0.0254719, 0.0235960
1: -0.0108810, -0.0007142, -0.0101351, -0.0011694, -0.0097116, 0.0094209
2: -0.0022171, 0.0267371, -0.0018444, 0.0248405, -0.0270575, 0.0285815
3: -0.0136622, 0.0131161, -0.0119507, 0.0107289, -0.0243911, 0.0250668
4: -0.0130820, 0.0111945, -0.0118689, 0.0095278, -0.0226098, 0.0230634
5: 0.9847723, 1.0125971, 0.9874392, 1.0104873, -0.0257151, 0.0251579
6: -0.0121958, 0.0147363, -0.0100976, 0.0135746, -0.0257704, 0.0248339
7: -0.0285403, -0.0018907, -0.0274265, -0.0021527, -0.0263875, 0.0255358
8: -0.0110410, 0.0251226, -0.0095674, 0.0221032, -0.0331441, 0.0346900
9: -0.0098239, 0.0097909, -0.0087735, 0.0079918, -0.0178158, 0.0185644

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
time: 1.45 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0448981, 0.0542921, -0.0088757, 0.0127839, -0.0576821, 0.0631678
1: -0.0175173, 0.0047773, -0.0098275, -0.0013571, -0.0161602, 0.0146048
2: -0.0017526, 0.0543944, -0.0015853, 0.0240583, -0.0258109, 0.0559797
3: -0.0288909, 0.0430322, -0.0112449, 0.0097444, -0.0386353, 0.0542771
4: -0.0238761, 0.0279631, -0.0113686, 0.0088405, -0.0327166, 0.0393317
5: 0.9004847, 1.0313700, 0.9885388, 1.0096172, -0.1091325, 0.0428312
6: -0.0366282, 0.0250728, -0.0092323, 0.0130955, -0.0497237, 0.0343052
7: -0.0384505, 0.0223546, -0.0269671, -0.0023350, -0.0361155, 0.0493217
8: -0.0241521, 0.0613036, -0.0089597, 0.0208580, -0.0450101, 0.0702633
9: -0.0191705, 0.0265738, -0.0083403, 0.0072499, -0.0264205, 0.0349141

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
time: 1.82 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0121589, 0.0145958, -0.0140331, 0.0156932, -0.0278521, 0.0286289
1: -0.0108810, -0.0007142, -0.0112609, -0.0004597, -0.0104212, 0.0105466
2: -0.0022171, 0.0267371, -0.0025816, 0.0278724, -0.0300895, 0.0293186
3: -0.0136622, 0.0131161, -0.0145340, 0.0144683, -0.0281305, 0.0276501
4: -0.0130820, 0.0111945, -0.0137000, 0.0120739, -0.0251559, 0.0248945
5: 0.9847723, 1.0125971, 0.9824629, 1.0136719, -0.0288996, 0.0301341
6: -0.0121958, 0.0147363, -0.0133550, 0.0153281, -0.0275238, 0.0280913
7: -0.0285403, -0.0018907, -0.0291076, -0.0016343, -0.0269059, 0.0272169
8: -0.0110410, 0.0251226, -0.0117915, 0.0268068, -0.0378478, 0.0369142
9: -0.0098239, 0.0097909, -0.0103590, 0.0107195, -0.0205434, 0.0201500

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 1.37 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0448981, 0.0542921, -0.0125026, 0.0147156, -0.0596137, 0.0667947
1: -0.0175173, 0.0047773, -0.0109506, -0.0006717, -0.0168456, 0.0157279
2: -0.0017526, 0.0543944, -0.0023180, 0.0269142, -0.0286668, 0.0567124
3: -0.0288909, 0.0430322, -0.0138220, 0.0133390, -0.0422299, 0.0568543
4: -0.0238761, 0.0279631, -0.0131953, 0.0113502, -0.0352263, 0.0411584
5: 0.9004847, 1.0313700, 0.9845230, 1.0127941, -0.1123095, 0.0468470
6: -0.0366282, 0.0250728, -0.0123917, 0.0148448, -0.0514730, 0.0374646
7: -0.0384505, 0.0223546, -0.0286443, -0.0018197, -0.0366308, 0.0509988
8: -0.0241521, 0.0613036, -0.0111786, 0.0254046, -0.0495566, 0.0724822
9: -0.0191705, 0.0265738, -0.0099221, 0.0099590, -0.0291295, 0.0364959

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
time: 1.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.66 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0310085
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310085

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

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.17 seconds

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

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0389911, 0.0239504, -0.0088254, 0.0126686, -0.0516597, 0.0327758
1: -0.0163199, 0.0026051, -0.0097605, -0.0013980, -0.0149219, 0.0123655
2: -0.0010463, 0.0405677, -0.0014807, 0.0238878, -0.0249341, 0.0420484
3: -0.0261432, 0.0305244, -0.0110910, 0.0095298, -0.0356731, 0.0416154
4: -0.0219286, 0.0233487, -0.0112595, 0.0086906, -0.0306192, 0.0346082
5: 0.9653244, 1.0279828, 0.9887787, 1.0094277, -0.0441033, 0.0392041
6: -0.0274964, 0.0232079, -0.0090437, 0.0129911, -0.0404875, 0.0322516
7: -0.0366624, 0.0054587, -0.0268670, -0.0024086, -0.0342539, 0.0323257
8: -0.0217865, 0.0471410, -0.0088273, 0.0205866, -0.0423731, 0.0559683
9: -0.0174842, 0.0229104, -0.0082459, 0.0070882, -0.0245724, 0.0311562

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0389911, 0.0239504, -0.0121589, 0.0145958, -0.0535869, 0.0361094
1: -0.0163199, 0.0026051, -0.0108810, -0.0007142, -0.0156057, 0.0134860
2: -0.0010463, 0.0405677, -0.0022171, 0.0267371, -0.0277833, 0.0427847
3: -0.0261432, 0.0305244, -0.0136622, 0.0131161, -0.0392593, 0.0441866
4: -0.0219286, 0.0233487, -0.0130820, 0.0111945, -0.0331231, 0.0364307
5: 0.9653244, 1.0279828, 0.9847723, 1.0125971, -0.0472727, 0.0432106
6: -0.0274964, 0.0232079, -0.0121958, 0.0147363, -0.0422327, 0.0354036
7: -0.0366624, 0.0054587, -0.0285403, -0.0018907, -0.0347718, 0.0339990
8: -0.0217865, 0.0471410, -0.0110410, 0.0251226, -0.0469091, 0.0581820
9: -0.0174842, 0.0229104, -0.0098239, 0.0097909, -0.0272751, 0.0327343

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0084472, 0.0110938, -0.0369596, 0.0443644, -0.0528116, 0.0480534
1: -0.0088448, -0.0019568, -0.0159082, 0.0034303, -0.0122751, 0.0139513
2: -0.0006941, 0.0215595, -0.0007540, 0.0475729, -0.0482670, 0.0223135
3: -0.0089898, 0.0065991, -0.0251983, 0.0356856, -0.0446754, 0.0317974
4: -0.0097702, 0.0066445, -0.0212588, 0.0238764, -0.0336466, 0.0279033
5: 0.9920526, 1.0068376, 0.9215696, 1.0268180, -0.0347654, 0.0852680
6: -0.0064678, 0.0115649, -0.0306423, 0.0225665, -0.0290343, 0.0422072
7: -0.0254996, -0.0029618, -0.0360475, 0.0147773, -0.0402770, 0.0330857
8: -0.0070183, 0.0168798, -0.0209729, 0.0524309, -0.0594492, 0.0378527
9: -0.0069563, 0.0048796, -0.0169042, 0.0224961, -0.0294523, 0.0217838

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300133, upper bound: 0.0306105
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0296662, upper bound: 0.0292505
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0084472, 0.0110938, -0.0427770, 0.0516393, -0.0600866, 0.0538707
1: -0.0088448, -0.0019568, -0.0170873, 0.0044174, -0.0132622, 0.0151305
2: -0.0006941, 0.0215595, -0.0014602, 0.0525716, -0.0532657, 0.0230196
3: -0.0089898, 0.0065991, -0.0279042, 0.0410691, -0.0500590, 0.0345033
4: -0.0097702, 0.0066445, -0.0231767, 0.0268711, -0.0366413, 0.0298212
5: 0.9920526, 1.0068376, 0.9061186, 1.0301538, -0.0381011, 0.1007190
6: -0.0064678, 0.0115649, -0.0350287, 0.0244031, -0.0308709, 0.0465936
7: -0.0254996, -0.0029618, -0.0378084, 0.0203299, -0.0458295, 0.0348466
8: -0.0070183, 0.0168798, -0.0233026, 0.0589328, -0.0659510, 0.0401823
9: -0.0069563, 0.0048796, -0.0185649, 0.0254842, -0.0324405, 0.0234445

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300133, upper bound: 0.0306542
time: 1.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0296662, upper bound: 0.0292505
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0087879, 0.0120854, -0.0389385, 0.0468391, -0.0556270, 0.0510239
1: -0.0094214, -0.0016049, -0.0163093, 0.0037661, -0.0131875, 0.0147043
2: -0.0014028, 0.0230256, -0.0010338, 0.0492733, -0.0506761, 0.0240594
3: -0.0103129, 0.0084445, -0.0261187, 0.0375169, -0.0478297, 0.0345633
4: -0.0107080, 0.0079329, -0.0219112, 0.0248951, -0.0356031, 0.0298441
5: 0.9899911, 1.0084684, 0.9163138, 1.0279528, -0.0379617, 0.0921546
6: -0.0080898, 0.0124630, -0.0321344, 0.0231912, -0.0312810, 0.0445973
7: -0.0263606, -0.0024634, -0.0366465, 0.0166661, -0.0430267, 0.0341831
8: -0.0081574, 0.0192139, -0.0217654, 0.0546426, -0.0628000, 0.0409792
9: -0.0077683, 0.0062703, -0.0174691, 0.0235125, -0.0312808, 0.0237395

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0087879, 0.0120854, -0.0447629, 0.0541228, -0.0629108, 0.0568483
1: -0.0094214, -0.0016049, -0.0174899, 0.0047543, -0.0141757, 0.0158849
2: -0.0014028, 0.0230256, -0.0017391, 0.0542781, -0.0556809, 0.0247647
3: -0.0103129, 0.0084445, -0.0288279, 0.0429070, -0.0532199, 0.0372724
4: -0.0107080, 0.0079329, -0.0238315, 0.0278934, -0.0386014, 0.0317644
5: 0.9899911, 1.0084684, 0.9008442, 1.0312923, -0.0413013, 0.1076242
6: -0.0080898, 0.0124630, -0.0365261, 0.0250301, -0.0331199, 0.0489890
7: -0.0263606, -0.0024634, -0.0384096, 0.0222254, -0.0485860, 0.0359462
8: -0.0081574, 0.0192139, -0.0240979, 0.0611523, -0.0693097, 0.0433118
9: -0.0077683, 0.0062703, -0.0191319, 0.0265043, -0.0342726, 0.0254022

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A1_B1

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

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
time: 1.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0121589, 0.0145958, -0.0389911, 0.0239504, -0.0361094, 0.0535869
1: -0.0108810, -0.0007142, -0.0163199, 0.0026051, -0.0134860, 0.0156057
2: -0.0022171, 0.0267371, -0.0010463, 0.0405677, -0.0427847, 0.0277833
3: -0.0136622, 0.0131161, -0.0261432, 0.0305244, -0.0441866, 0.0392593
4: -0.0130820, 0.0111945, -0.0219286, 0.0233487, -0.0364307, 0.0331231
5: 0.9847723, 1.0125971, 0.9653244, 1.0279828, -0.0432106, 0.0472727
6: -0.0121958, 0.0147363, -0.0274964, 0.0232079, -0.0354036, 0.0422327
7: -0.0285403, -0.0018907, -0.0366624, 0.0054587, -0.0339990, 0.0347718
8: -0.0110410, 0.0251226, -0.0217865, 0.0471410, -0.0581820, 0.0469091
9: -0.0098239, 0.0097909, -0.0174842, 0.0229104, -0.0327343, 0.0272751

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0427770, 0.0516393, -0.0084472, 0.0110938, -0.0538707, 0.0600866
1: -0.0170873, 0.0044174, -0.0088448, -0.0019568, -0.0151305, 0.0132622
2: -0.0014602, 0.0525716, -0.0006941, 0.0215595, -0.0230196, 0.0532657
3: -0.0279042, 0.0410691, -0.0089898, 0.0065991, -0.0345033, 0.0500590
4: -0.0231767, 0.0268711, -0.0097702, 0.0066445, -0.0298212, 0.0366413
5: 0.9061186, 1.0301538, 0.9920526, 1.0068376, -0.1007190, 0.0381011
6: -0.0350287, 0.0244031, -0.0064678, 0.0115649, -0.0465936, 0.0308709
7: -0.0378084, 0.0203299, -0.0254996, -0.0029618, -0.0348466, 0.0458295
8: -0.0233026, 0.0589328, -0.0070183, 0.0168798, -0.0401823, 0.0659510
9: -0.0185649, 0.0254842, -0.0069563, 0.0048796, -0.0234445, 0.0324405

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0310084
time: 1.75 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0308987
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0447629, 0.0541228, -0.0087879, 0.0120854, -0.0568483, 0.0629108
1: -0.0174899, 0.0047543, -0.0094214, -0.0016049, -0.0158849, 0.0141757
2: -0.0017391, 0.0542781, -0.0014028, 0.0230256, -0.0247647, 0.0556809
3: -0.0288279, 0.0429070, -0.0103129, 0.0084445, -0.0372724, 0.0532199
4: -0.0238315, 0.0278934, -0.0107080, 0.0079329, -0.0317644, 0.0386014
5: 0.9008442, 1.0312923, 0.9899911, 1.0084684, -0.1076242, 0.0413013
6: -0.0365261, 0.0250301, -0.0080898, 0.0124630, -0.0489890, 0.0331199
7: -0.0384096, 0.0222254, -0.0263606, -0.0024634, -0.0359462, 0.0485860
8: -0.0240979, 0.0611523, -0.0081574, 0.0192139, -0.0433118, 0.0693097
9: -0.0191319, 0.0265043, -0.0077683, 0.0062703, -0.0254022, 0.0342726

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0310085
time: 6.42 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0308987
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1

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

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0121589, 0.0145958, -0.0448981, 0.0542921, -0.0664510, 0.0594939
1: -0.0108810, -0.0007142, -0.0175173, 0.0047773, -0.0156582, 0.0168031
2: -0.0022171, 0.0267371, -0.0017526, 0.0543944, -0.0566115, 0.0284896
3: -0.0136622, 0.0131161, -0.0288909, 0.0430322, -0.0566945, 0.0420070
4: -0.0130820, 0.0111945, -0.0238761, 0.0279631, -0.0410451, 0.0350707
5: 0.9847723, 1.0125971, 0.9004847, 1.0313700, -0.0465978, 0.1121124
6: -0.0121958, 0.0147363, -0.0366282, 0.0250728, -0.0372686, 0.0513645
7: -0.0285403, -0.0018907, -0.0384505, 0.0223546, -0.0508948, 0.0365599
8: -0.0110410, 0.0251226, -0.0241521, 0.0613036, -0.0723446, 0.0492747
9: -0.0098239, 0.0097909, -0.0191705, 0.0265738, -0.0363978, 0.0289615

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0427770, 0.0516393, -0.0087932, 0.0130526, -0.0558296, 0.0604325
1: -0.0170873, 0.0044174, -0.0099837, -0.0012618, -0.0158256, 0.0144011
2: -0.0014602, 0.0525716, -0.0014138, 0.0244556, -0.0259158, 0.0539854
3: -0.0279042, 0.0410691, -0.0116033, 0.0102444, -0.0381486, 0.0526725
4: -0.0231767, 0.0268711, -0.0116227, 0.0091896, -0.0323663, 0.0384938
5: 0.9061186, 1.0301538, 0.9879803, 1.0100591, -0.1039405, 0.0421734
6: -0.0350287, 0.0244031, -0.0096718, 0.0133389, -0.0483675, 0.0340749
7: -0.0378084, 0.0203299, -0.0272004, -0.0024556, -0.0353528, 0.0475303
8: -0.0233026, 0.0589328, -0.0092684, 0.0214904, -0.0447930, 0.0682012
9: -0.0185649, 0.0254842, -0.0085603, 0.0076268, -0.0261917, 0.0340445

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310084
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0447629, 0.0541228, -0.0104738, 0.0140083, -0.0587711, 0.0645966
1: -0.0174899, 0.0047543, -0.0105394, -0.0009227, -0.0165672, 0.0152937
2: -0.0017391, 0.0542781, -0.0021242, 0.0258685, -0.0276076, 0.0564023
3: -0.0288279, 0.0429070, -0.0128784, 0.0120228, -0.0408507, 0.0557854
4: -0.0238315, 0.0278934, -0.0125264, 0.0104312, -0.0342627, 0.0404198
5: 0.9008442, 1.0312923, 0.9859936, 1.0116309, -0.1107867, 0.0452988
6: -0.0365261, 0.0250301, -0.0112348, 0.0142043, -0.0507304, 0.0362650
7: -0.0384096, 0.0222254, -0.0280302, -0.0019560, -0.0364536, 0.0502555
8: -0.0240979, 0.0611523, -0.0103661, 0.0237398, -0.0478377, 0.0715184
9: -0.0191319, 0.0265043, -0.0093429, 0.0089670, -0.0280989, 0.0358472

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310085
time: 1.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987
time: 1.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.69 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0310273
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0300133, upper bound: 0.0306105
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0296662, upper bound: 0.0292505
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0300133, upper bound: 0.0306542
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0296662, upper bound: 0.0292505
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0310273
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0310084
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0308987
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0310085
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0308987
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310084
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310085
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.69
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0083986, 0.0109956, -0.0086859, 0.0119366, -0.0203352, 0.0196815
1: -0.0087878, -0.0019916, -0.0093348, -0.0016578, -0.0071300, 0.0073432
2: -0.0005930, 0.0214143, -0.0011906, 0.0228055, -0.0233985, 0.0226049
3: -0.0088589, 0.0064165, -0.0101143, 0.0081676, -0.0170264, 0.0165308
4: -0.0096774, 0.0065350, -0.0105672, 0.0077395, -0.0174169, 0.0171022
5: 0.9922568, 1.0066760, 0.9903004, 1.0082235, -0.0159668, 0.0163755
6: -0.0063073, 0.0114761, -0.0078464, 0.0123282, -0.0186354, 0.0193224
7: -0.0254144, -0.0030330, -0.0262314, -0.0026127, -0.0228018, 0.0231984
8: -0.0069055, 0.0166487, -0.0079864, 0.0188636, -0.0257691, 0.0246351
9: -0.0068759, 0.0047419, -0.0076464, 0.0060616, -0.0129375, 0.0123883

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0087370, 0.0119738, -0.0088193, 0.0126219, -0.0213589, 0.0207931
1: -0.0093565, -0.0016445, -0.0097333, -0.0014146, -0.0079419, 0.0080887
2: -0.0012970, 0.0228606, -0.0014681, 0.0238187, -0.0251157, 0.0243287
3: -0.0101640, 0.0082368, -0.0110286, 0.0094428, -0.0196068, 0.0192655
4: -0.0106025, 0.0077879, -0.0112153, 0.0086299, -0.0192324, 0.0190032
5: 0.9902231, 1.0082847, 0.9888757, 1.0093509, -0.0191278, 0.0194089
6: -0.0079072, 0.0123619, -0.0089672, 0.0129488, -0.0208560, 0.0213291
7: -0.0262637, -0.0025378, -0.0268264, -0.0024174, -0.0238463, 0.0242886
8: -0.0080292, 0.0189512, -0.0087736, 0.0204765, -0.0285057, 0.0277248
9: -0.0076769, 0.0061138, -0.0082076, 0.0070226, -0.0146996, 0.0143214

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0083986, 0.0109956, -0.0100343, 0.0138550, -0.0222536, 0.0210299
1: -0.0087878, -0.0019916, -0.0104503, -0.0009770, -0.0078107, 0.0084586
2: -0.0005930, 0.0214143, -0.0019226, 0.0256420, -0.0262349, 0.0233369
3: -0.0088589, 0.0064165, -0.0126740, 0.0117377, -0.0205966, 0.0190905
4: -0.0096774, 0.0065350, -0.0123815, 0.0102321, -0.0199096, 0.0189165
5: 0.9922568, 1.0066760, 0.9863120, 1.0113789, -0.0191221, 0.0203639
6: -0.0063073, 0.0114761, -0.0109843, 0.0140655, -0.0203728, 0.0224603
7: -0.0254144, -0.0030330, -0.0278971, -0.0020978, -0.0233166, 0.0248642
8: -0.0069055, 0.0166487, -0.0101901, 0.0233792, -0.0302847, 0.0268389
9: -0.0068759, 0.0047419, -0.0092174, 0.0087521, -0.0156280, 0.0139593

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309889, upper bound: 0.0303039
time: 2.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305590, upper bound: 0.0303039
time: 1.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087370, 0.0119738, -0.0120229, 0.0145483, -0.0232854, 0.0239967
1: -0.0093565, -0.0016445, -0.0108534, -0.0007310, -0.0086254, 0.0092088
2: -0.0012970, 0.0228606, -0.0022036, 0.0266670, -0.0279640, 0.0250642
3: -0.0101640, 0.0082368, -0.0135989, 0.0130278, -0.0231918, 0.0218358
4: -0.0106025, 0.0077879, -0.0130372, 0.0111329, -0.0217354, 0.0208251
5: 0.9902231, 1.0082847, 0.9848709, 1.0125191, -0.0222960, 0.0234138
6: -0.0079072, 0.0123619, -0.0121182, 0.0146934, -0.0226006, 0.0244801
7: -0.0262637, -0.0025378, -0.0284991, -0.0019002, -0.0243636, 0.0259613
8: -0.0080292, 0.0189512, -0.0109865, 0.0250110, -0.0330401, 0.0299377
9: -0.0076769, 0.0061138, -0.0097851, 0.0097244, -0.0174014, 0.0158989

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307342
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307340, upper bound: 0.0307342
time: 1.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0343371, 0.0223278, -0.0086859, 0.0119366, -0.0462737, 0.0310137
1: -0.0153766, 0.0020293, -0.0093348, -0.0016578, -0.0137188, 0.0113642
2: -0.0001689, 0.0381687, -0.0011906, 0.0228055, -0.0229745, 0.0393593
3: -0.0239784, 0.0275049, -0.0101143, 0.0081676, -0.0321460, 0.0376192
4: -0.0203941, 0.0212405, -0.0105672, 0.0077395, -0.0281337, 0.0318078
5: 0.9686975, 1.0253141, 0.9903004, 1.0082235, -0.0395260, 0.0350137
6: -0.0248425, 0.0217385, -0.0078464, 0.0123282, -0.0371707, 0.0295848
7: -0.0352536, 0.0030751, -0.0262314, -0.0026127, -0.0326410, 0.0293066
8: -0.0199227, 0.0433220, -0.0079864, 0.0188636, -0.0387862, 0.0513084
9: -0.0161555, 0.0206348, -0.0076464, 0.0060616, -0.0222171, 0.0282812

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0370016, 0.0232568, -0.0088193, 0.0126219, -0.0496234, 0.0320761
1: -0.0159166, 0.0023589, -0.0097333, -0.0014146, -0.0145020, 0.0120922
2: -0.0008611, 0.0395421, -0.0014681, 0.0238187, -0.0246798, 0.0410102
3: -0.0252178, 0.0292335, -0.0110286, 0.0094428, -0.0346606, 0.0402622
4: -0.0212726, 0.0224475, -0.0112153, 0.0086299, -0.0299025, 0.0336628
5: 0.9667663, 1.0268422, 0.9888757, 1.0093509, -0.0425846, 0.0379665
6: -0.0263619, 0.0225797, -0.0089672, 0.0129488, -0.0393106, 0.0315469
7: -0.0360602, 0.0044397, -0.0268264, -0.0024174, -0.0336428, 0.0312661
8: -0.0209897, 0.0455084, -0.0087736, 0.0204765, -0.0414663, 0.0542820
9: -0.0169162, 0.0219376, -0.0082076, 0.0070226, -0.0239388, 0.0301452

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0343371, 0.0223278, -0.0100343, 0.0138550, -0.0481922, 0.0323622
1: -0.0153766, 0.0020293, -0.0104503, -0.0009770, -0.0143995, 0.0124796
2: -0.0001689, 0.0381687, -0.0019226, 0.0256420, -0.0258109, 0.0400913
3: -0.0239784, 0.0275049, -0.0126740, 0.0117377, -0.0357161, 0.0401789
4: -0.0203941, 0.0212405, -0.0123815, 0.0102321, -0.0306263, 0.0336221
5: 0.9686975, 1.0253141, 0.9863120, 1.0113789, -0.0426813, 0.0390021
6: -0.0248425, 0.0217385, -0.0109843, 0.0140655, -0.0389081, 0.0327227
7: -0.0352536, 0.0030751, -0.0278971, -0.0020978, -0.0331558, 0.0309723
8: -0.0199227, 0.0433220, -0.0101901, 0.0233792, -0.0433018, 0.0535121
9: -0.0161555, 0.0206348, -0.0092174, 0.0087521, -0.0249076, 0.0298522

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0302093, upper bound: 0.0292505
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0298640, upper bound: 0.0292505
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0370016, 0.0232568, -0.0120229, 0.0145483, -0.0515499, 0.0352797
1: -0.0159166, 0.0023589, -0.0108534, -0.0007310, -0.0151856, 0.0132123
2: -0.0008611, 0.0395421, -0.0022036, 0.0266670, -0.0275281, 0.0417457
3: -0.0252178, 0.0292335, -0.0135989, 0.0130278, -0.0382456, 0.0428325
4: -0.0212726, 0.0224475, -0.0130372, 0.0111329, -0.0324055, 0.0354846
5: 0.9667663, 1.0268422, 0.9848709, 1.0125191, -0.0457528, 0.0419714
6: -0.0263619, 0.0225797, -0.0121182, 0.0146934, -0.0410553, 0.0346979
7: -0.0360602, 0.0044397, -0.0284991, -0.0019002, -0.0341600, 0.0329388
8: -0.0209897, 0.0455084, -0.0109865, 0.0250110, -0.0460007, 0.0564949
9: -0.0169162, 0.0219376, -0.0097851, 0.0097244, -0.0266406, 0.0317227

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308615, upper bound: 0.0300699
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304305, upper bound: 0.0300699
time: 1.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0087370, 0.0119738, -0.0389385, 0.0468391, -0.0555761, 0.0509123
1: -0.0093565, -0.0016445, -0.0163093, 0.0037661, -0.0131226, 0.0146647
2: -0.0012970, 0.0228606, -0.0010338, 0.0492733, -0.0505703, 0.0238944
3: -0.0101640, 0.0082368, -0.0261187, 0.0375169, -0.0476808, 0.0343556
4: -0.0106025, 0.0077879, -0.0219112, 0.0248951, -0.0354975, 0.0296991
5: 0.9902231, 1.0082847, 0.9163138, 1.0279528, -0.0377297, 0.0919709
6: -0.0079072, 0.0123619, -0.0321344, 0.0231912, -0.0310985, 0.0444962
7: -0.0262637, -0.0025378, -0.0366465, 0.0166661, -0.0429298, 0.0341087
8: -0.0080292, 0.0189512, -0.0217654, 0.0546426, -0.0626718, 0.0407165
9: -0.0076769, 0.0061138, -0.0174691, 0.0235125, -0.0311894, 0.0235829

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0309088
time: 1.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0309088
time: 1.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0370015, 0.0232568, -0.0389385, 0.0468391, -0.0838406, 0.0621953
1: -0.0159166, 0.0023589, -0.0163093, 0.0037661, -0.0196827, 0.0186682
2: -0.0008609, 0.0395421, -0.0010338, 0.0492733, -0.0501342, 0.0405759
3: -0.0252178, 0.0292336, -0.0261187, 0.0375169, -0.0627346, 0.0553523
4: -0.0212726, 0.0224475, -0.0219112, 0.0248951, -0.0461676, 0.0443587
5: 0.9667664, 1.0268421, 0.9163138, 1.0279528, -0.0611864, 0.1105283
6: -0.0263619, 0.0225797, -0.0321344, 0.0231912, -0.0495531, 0.0547141
7: -0.0360602, 0.0044397, -0.0366465, 0.0166661, -0.0527263, 0.0410862
8: -0.0209897, 0.0455084, -0.0217654, 0.0546426, -0.0756323, 0.0672737
9: -0.0169162, 0.0219376, -0.0174691, 0.0235125, -0.0404287, 0.0394067

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0309088
time: 1.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0309088
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0087370, 0.0119738, -0.0447629, 0.0541228, -0.0628599, 0.0567366
1: -0.0093565, -0.0016445, -0.0174899, 0.0047543, -0.0141108, 0.0158453
2: -0.0012970, 0.0228606, -0.0017391, 0.0542781, -0.0555751, 0.0245997
3: -0.0101640, 0.0082368, -0.0288279, 0.0429070, -0.0530710, 0.0370648
4: -0.0106025, 0.0077879, -0.0238315, 0.0278934, -0.0384959, 0.0316194
5: 0.9902231, 1.0082847, 0.9008442, 1.0312923, -0.0410692, 0.1074405
6: -0.0079072, 0.0123619, -0.0365261, 0.0250301, -0.0329374, 0.0488880
7: -0.0262637, -0.0025378, -0.0384096, 0.0222254, -0.0484891, 0.0358718
8: -0.0080292, 0.0189512, -0.0240979, 0.0611523, -0.0691815, 0.0430491
9: -0.0076769, 0.0061138, -0.0191319, 0.0265043, -0.0341813, 0.0252457

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0309088
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0309088
time: 1.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0370015, 0.0232568, -0.0447629, 0.0541228, -0.0911244, 0.0680196
1: -0.0159166, 0.0023589, -0.0174899, 0.0047543, -0.0206710, 0.0198488
2: -0.0008609, 0.0395421, -0.0017391, 0.0542781, -0.0551390, 0.0412812
3: -0.0252178, 0.0292336, -0.0288279, 0.0429070, -0.0681248, 0.0580615
4: -0.0212726, 0.0224475, -0.0238315, 0.0278934, -0.0491660, 0.0462789
5: 0.9667664, 1.0268421, 0.9008442, 1.0312923, -0.0645259, 0.1259980
6: -0.0263619, 0.0225797, -0.0365261, 0.0250301, -0.0513920, 0.0591058
7: -0.0360602, 0.0044397, -0.0384096, 0.0222254, -0.0582856, 0.0428493
8: -0.0209897, 0.0455084, -0.0240979, 0.0611523, -0.0821420, 0.0696062
9: -0.0169162, 0.0219376, -0.0191319, 0.0265043, -0.0434205, 0.0410695

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0309088
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0309088
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

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

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0087471, 0.0129486, -0.0368814, 0.0232149, -0.0319619, 0.0498299
1: -0.0099233, -0.0012987, -0.0158923, 0.0023441, -0.0122673, 0.0145936
2: -0.0013179, 0.0243018, -0.0007540, 0.0394802, -0.0407980, 0.0250558
3: -0.0114645, 0.0100508, -0.0251619, 0.0291556, -0.0406201, 0.0352127
4: -0.0115243, 0.0090544, -0.0212330, 0.0223930, -0.0339173, 0.0302874
5: 0.9881966, 1.0098879, 0.9668534, 1.0267730, -0.0385764, 0.0430345
6: -0.0095016, 0.0132446, -0.0262933, 0.0225418, -0.0320434, 0.0395380
7: -0.0271101, -0.0025231, -0.0360238, 0.0043782, -0.0314883, 0.0335007
8: -0.0091489, 0.0212455, -0.0209416, 0.0454097, -0.0545586, 0.0421871
9: -0.0084751, 0.0074808, -0.0168818, 0.0218788, -0.0303539, 0.0243627

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
time: 1.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0311350
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0101426, 0.0138928, -0.0388585, 0.0239042, -0.0340468, 0.0527513
1: -0.0104723, -0.0009636, -0.0162931, 0.0025886, -0.0130609, 0.0153294
2: -0.0020202, 0.0256978, -0.0010338, 0.0404993, -0.0425195, 0.0267316
3: -0.0127244, 0.0118080, -0.0260815, 0.0304383, -0.0431627, 0.0378895
4: -0.0124173, 0.0102812, -0.0218848, 0.0232886, -0.0357059, 0.0321660
5: 0.9862335, 1.0114410, 0.9654205, 1.0279068, -0.0416733, 0.0460205
6: -0.0110461, 0.0140997, -0.0274208, 0.0231660, -0.0342120, 0.0415205
7: -0.0279299, -0.0020292, -0.0366223, 0.0053908, -0.0333207, 0.0345931
8: -0.0102335, 0.0234681, -0.0217334, 0.0470322, -0.0572657, 0.0452014
9: -0.0092484, 0.0088051, -0.0174463, 0.0228455, -0.0320939, 0.0262514

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
time: 1.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0311350
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0375607, 0.0451160, -0.0083894, 0.0107108, -0.0482715, 0.0535054
1: -0.0160300, 0.0035323, -0.0086221, -0.0020927, -0.0139373, 0.0121544
2: -0.0008895, 0.0480894, -0.0005738, 0.0209932, -0.0218827, 0.0486632
3: -0.0254779, 0.0362418, -0.0084788, 0.0058864, -0.0313643, 0.0447207
4: -0.0214570, 0.0241858, -0.0094080, 0.0065207, -0.0279776, 0.0335938
5: 0.9199732, 1.0271627, 0.9928490, 1.0062076, -0.0862344, 0.0343137
6: -0.0310955, 0.0227562, -0.0058414, 0.0112181, -0.0423136, 0.0285976
7: -0.0362294, 0.0153511, -0.0251671, -0.0030465, -0.0331830, 0.0405182
8: -0.0212137, 0.0531028, -0.0068218, 0.0159783, -0.0371920, 0.0599246
9: -0.0170758, 0.0228048, -0.0066427, 0.0043424, -0.0214183, 0.0294475

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305247, upper bound: 0.0299441
time: 2.19 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0290644, upper bound: 0.0295077
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0428730, 0.0517595, -0.0083855, 0.0106107, -0.0534838, 0.0601450
1: -0.0171068, 0.0044337, -0.0085640, -0.0021282, -0.0149786, 0.0129976
2: -0.0010797, 0.0526542, -0.0005657, 0.0208453, -0.0219250, 0.0532199
3: -0.0279489, 0.0411581, -0.0083454, 0.0057002, -0.0336491, 0.0495035
4: -0.0232084, 0.0269206, -0.0093134, 0.0065146, -0.0297230, 0.0362340
5: 0.9058634, 1.0302089, 0.9930569, 1.0060428, -0.1001794, 0.0371521
6: -0.0351011, 0.0244334, -0.0056778, 0.0111275, -0.0462286, 0.0301112
7: -0.0378375, 0.0204216, -0.0250802, -0.0030521, -0.0347854, 0.0455018
8: -0.0233411, 0.0590402, -0.0068174, 0.0157428, -0.0390839, 0.0658575
9: -0.0185924, 0.0255336, -0.0065607, 0.0042021, -0.0227945, 0.0320943

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305227, upper bound: 0.0299046
time: 2.13 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0290644, upper bound: 0.0294802
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0395405, 0.0475920, -0.0087311, 0.0117068, -0.0512473, 0.0563230
1: -0.0164313, 0.0038682, -0.0092013, -0.0017393, -0.0146920, 0.0130695
2: -0.0011687, 0.0497906, -0.0012846, 0.0224658, -0.0236345, 0.0510752
3: -0.0263988, 0.0380740, -0.0098077, 0.0077399, -0.0341387, 0.0478817
4: -0.0221097, 0.0252050, -0.0103500, 0.0074410, -0.0295506, 0.0355549
5: 0.9147148, 1.0282978, 0.9907781, 1.0078455, -0.0931308, 0.0375197
6: -0.0325883, 0.0233813, -0.0074705, 0.0121201, -0.0447084, 0.0308518
7: -0.0368287, 0.0172407, -0.0260319, -0.0025465, -0.0342822, 0.0432726
8: -0.0220065, 0.0553155, -0.0077225, 0.0183227, -0.0403292, 0.0630380
9: -0.0176410, 0.0238218, -0.0074583, 0.0057393, -0.0233803, 0.0312800

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306083, upper bound: 0.0300699
time: 1.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0300699
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0448240, 0.0541993, -0.0087252, 0.0115972, -0.0564212, 0.0629245
1: -0.0175023, 0.0047647, -0.0091375, -0.0017782, -0.0157241, 0.0139022
2: -0.0013582, 0.0543306, -0.0012723, 0.0223038, -0.0236620, 0.0556029
3: -0.0288564, 0.0429636, -0.0096615, 0.0075360, -0.0363924, 0.0526251
4: -0.0238516, 0.0279249, -0.0102463, 0.0072986, -0.0311502, 0.0381712
5: 0.9006816, 1.0313275, 0.9910061, 1.0076655, -0.1069839, 0.0403214
6: -0.0365722, 0.0250494, -0.0072913, 0.0120208, -0.0485930, 0.0323407
7: -0.0384280, 0.0222837, -0.0259367, -0.0025552, -0.0358729, 0.0482205
8: -0.0241223, 0.0612207, -0.0075966, 0.0180647, -0.0421871, 0.0688173
9: -0.0191493, 0.0265357, -0.0073685, 0.0055856, -0.0247349, 0.0339043

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306056, upper bound: 0.0299981
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0299981
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

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

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310689, upper bound: 0.0303039
time: 2.21 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306008, upper bound: 0.0303039
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

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

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307340
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307340, upper bound: 0.0307340
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0087471, 0.0129486, -0.0427770, 0.0516393, -0.0603864, 0.0557255
1: -0.0099233, -0.0012987, -0.0170873, 0.0044174, -0.0143406, 0.0157886
2: -0.0013179, 0.0243018, -0.0014602, 0.0525716, -0.0538895, 0.0257619
3: -0.0114645, 0.0100508, -0.0279042, 0.0410691, -0.0525337, 0.0379550
4: -0.0115243, 0.0090544, -0.0231767, 0.0268711, -0.0383953, 0.0322311
5: 0.9881966, 1.0098879, 0.9061186, 1.0301538, -0.0419572, 0.1037694
6: -0.0095016, 0.0132446, -0.0350287, 0.0244031, -0.0339047, 0.0482733
7: -0.0271101, -0.0025231, -0.0378084, 0.0203299, -0.0474400, 0.0352853
8: -0.0091489, 0.0212455, -0.0233026, 0.0589328, -0.0680816, 0.0445481
9: -0.0084751, 0.0074808, -0.0185649, 0.0254842, -0.0339594, 0.0260458

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0311350
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0101426, 0.0138928, -0.0447629, 0.0541228, -0.0642655, 0.0586557
1: -0.0104723, -0.0009636, -0.0174899, 0.0047543, -0.0152266, 0.0165263
2: -0.0020202, 0.0256978, -0.0017391, 0.0542781, -0.0562982, 0.0274369
3: -0.0127244, 0.0118080, -0.0288279, 0.0429070, -0.0556314, 0.0406359
4: -0.0124173, 0.0102812, -0.0238315, 0.0278934, -0.0403107, 0.0341127
5: 0.9862335, 1.0114410, 0.9008442, 1.0312923, -0.0450588, 0.1105968
6: -0.0110461, 0.0140997, -0.0365261, 0.0250301, -0.0360762, 0.0506258
7: -0.0279299, -0.0020292, -0.0384096, 0.0222254, -0.0501553, 0.0363804
8: -0.0102335, 0.0234681, -0.0240979, 0.0611523, -0.0713858, 0.0475659
9: -0.0092484, 0.0088051, -0.0191319, 0.0265043, -0.0357527, 0.0279370

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0311350
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0375607, 0.0451160, -0.0087366, 0.0126668, -0.0502276, 0.0538527
1: -0.0160300, 0.0035323, -0.0097594, -0.0013986, -0.0146313, 0.0132917
2: -0.0008895, 0.0480894, -0.0012961, 0.0238852, -0.0247747, 0.0493855
3: -0.0254779, 0.0362418, -0.0110886, 0.0095265, -0.0350044, 0.0473305
4: -0.0214570, 0.0241858, -0.0112579, 0.0086883, -0.0301453, 0.0354436
5: 0.9199732, 1.0271627, 0.9887823, 1.0094247, -0.0894515, 0.0383804
6: -0.0310955, 0.0227562, -0.0090408, 0.0129895, -0.0440850, 0.0317970
7: -0.0362294, 0.0153511, -0.0268655, -0.0025384, -0.0336910, 0.0422165
8: -0.0212137, 0.0531028, -0.0088253, 0.0205824, -0.0417961, 0.0619280
9: -0.0170758, 0.0228048, -0.0082444, 0.0070857, -0.0241615, 0.0310492

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305247, upper bound: 0.0299441
time: 2.01 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0290989, upper bound: 0.0295253
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0428730, 0.0517595, -0.0087291, 0.0125549, -0.0554280, 0.0604887
1: -0.0171068, 0.0044337, -0.0096944, -0.0014383, -0.0156685, 0.0141280
2: -0.0010797, 0.0526542, -0.0012805, 0.0237198, -0.0247995, 0.0539348
3: -0.0279489, 0.0411581, -0.0109393, 0.0093182, -0.0372671, 0.0520974
4: -0.0232084, 0.0269206, -0.0111520, 0.0085429, -0.0317514, 0.0380726
5: 0.9058634, 1.0302089, 0.9890150, 1.0092404, -0.1033770, 0.0411939
6: -0.0351011, 0.0244334, -0.0088577, 0.0128882, -0.0479893, 0.0332912
7: -0.0378375, 0.0204216, -0.0267683, -0.0025494, -0.0352882, 0.0471899
8: -0.0233411, 0.0590402, -0.0086967, 0.0203190, -0.0436600, 0.0677369
9: -0.0185924, 0.0255336, -0.0081528, 0.0069288, -0.0255212, 0.0336864

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305227, upper bound: 0.0299046
time: 1.96 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0290989, upper bound: 0.0294991
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0395405, 0.0475920, -0.0093708, 0.0136237, -0.0531643, 0.0569628
1: -0.0164313, 0.0038682, -0.0103158, -0.0010591, -0.0153722, 0.0141840
2: -0.0011687, 0.0497906, -0.0020075, 0.0253000, -0.0264687, 0.0517981
3: -0.0263988, 0.0380740, -0.0123654, 0.0113073, -0.0377060, 0.0504393
4: -0.0221097, 0.0252050, -0.0121628, 0.0099316, -0.0320413, 0.0373678
5: 0.9147148, 1.0282978, 0.9867930, 1.0109984, -0.0962836, 0.0415048
6: -0.0325883, 0.0233813, -0.0106059, 0.0138561, -0.0464444, 0.0339872
7: -0.0368287, 0.0172407, -0.0276963, -0.0020381, -0.0347906, 0.0449370
8: -0.0220065, 0.0553155, -0.0099244, 0.0228348, -0.0448412, 0.0652399
9: -0.0176410, 0.0238218, -0.0090280, 0.0084277, -0.0260687, 0.0328498

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310085
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310085
time: 1.93 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0448240, 0.0541993, -0.0090701, 0.0135045, -0.0583285, 0.0632693
1: -0.0175023, 0.0047647, -0.0102465, -0.0011014, -0.0164009, 0.0150112
2: -0.0013582, 0.0543306, -0.0019898, 0.0251237, -0.0264819, 0.0563204
3: -0.0288564, 0.0429636, -0.0122063, 0.0110854, -0.0399418, 0.0551699
4: -0.0238516, 0.0279249, -0.0120500, 0.0097767, -0.0336284, 0.0399749
5: 0.9006816, 1.0313275, 0.9870409, 1.0108021, -0.1101205, 0.0442866
6: -0.0365722, 0.0250494, -0.0104109, 0.0137481, -0.0503203, 0.0354603
7: -0.0384280, 0.0222837, -0.0275928, -0.0020505, -0.0363775, 0.0498765
8: -0.0241223, 0.0612207, -0.0097875, 0.0225541, -0.0466765, 0.0710082
9: -0.0191493, 0.0265357, -0.0089304, 0.0082606, -0.0274099, 0.0354661

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987
time: 1.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.70 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0309889, upper bound: 0.0303039
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0305590, upper bound: 0.0303039
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307342
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0307340, upper bound: 0.0307342
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0309088
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0302093, upper bound: 0.0292505
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0298640, upper bound: 0.0292505
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308615, upper bound: 0.0300699
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0304305, upper bound: 0.0300699
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0309088
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0309088
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0309088
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0309088
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0309088
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0309088
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0309088
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0309088
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0311350
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0311350
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310273, upper bound: 0.0311350
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0309088, upper bound: 0.0311350
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0305247, upper bound: 0.0299441
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0290644, upper bound: 0.0295077
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0305227, upper bound: 0.0299046
NS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0290644, upper bound: 0.0294802
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0306083, upper bound: 0.0300699
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0300699
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0306056, upper bound: 0.0299981
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0299981
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310689, upper bound: 0.0303039
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0306008, upper bound: 0.0303039
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307340
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0307340, upper bound: 0.0307340
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0311350
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0310085, upper bound: 0.0311350
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0311350
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0305247, upper bound: 0.0299441
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0290989, upper bound: 0.0295253
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0305227, upper bound: 0.0299046
NS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0290989, upper bound: 0.0294991
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310085
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0310085
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.70
Output dim: 5, lower bound: -0.0308987, upper bound: 0.0308987

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0083416, 0.0106107, -0.0084077, 0.0102160, -0.0185576, 0.0190184
1: -0.0085640, -0.0021282, -0.0083493, -0.0022682, -0.0062957, 0.0062211
2: -0.0004744, 0.0208453, -0.0006118, 0.0203888, -0.0208632, 0.0214571
3: -0.0083454, 0.0057003, -0.0078188, 0.0052459, -0.0135913, 0.0135190
4: -0.0093134, 0.0064464, -0.0090230, 0.0065491, -0.0158625, 0.0154694
5: 0.9930568, 1.0060430, 0.9931333, 1.0053937, -0.0123369, 0.0129097
6: -0.0056778, 0.0111275, -0.0050322, 0.0108281, -0.0165059, 0.0161597
7: -0.0250802, -0.0031164, -0.0247376, -0.0030197, -0.0220605, 0.0216212
8: -0.0067674, 0.0157428, -0.0068426, 0.0150821, -0.0218495, 0.0225854
9: -0.0065607, 0.0042021, -0.0062376, 0.0039036, -0.0104643, 0.0104397

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306810, upper bound: 0.0311350
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304971, upper bound: 0.0301871
time: 1.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0083373, 0.0105138, -0.0085117, 0.0122915, -0.0206288, 0.0190254
1: -0.0085105, -0.0021626, -0.0095412, -0.0015318, -0.0069787, 0.0073786
2: -0.0004653, 0.0207265, -0.0008282, 0.0233304, -0.0237957, 0.0215546
3: -0.0082160, 0.0055739, -0.0105879, 0.0088281, -0.0170441, 0.0161618
4: -0.0092377, 0.0064396, -0.0109030, 0.0082007, -0.0174384, 0.0173426
5: 0.9931150, 1.0058833, 0.9895625, 1.0088074, -0.0156924, 0.0163209
6: -0.0055192, 0.0110509, -0.0084269, 0.0126496, -0.0181688, 0.0194778
7: -0.0249961, -0.0031227, -0.0265396, -0.0028676, -0.0221285, 0.0234169
8: -0.0067625, 0.0155664, -0.0083942, 0.0196991, -0.0264615, 0.0239605
9: -0.0064813, 0.0041153, -0.0079371, 0.0065594, -0.0130407, 0.0120524

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306341, upper bound: 0.0311350
time: 2.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304644, upper bound: 0.0301871
time: 2.10 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0086804, 0.0115931, -0.0085421, 0.0108868, -0.0195673, 0.0201352
1: -0.0091351, -0.0017796, -0.0087245, -0.0020302, -0.0071049, 0.0069449
2: -0.0011792, 0.0222977, -0.0008914, 0.0212536, -0.0224328, 0.0231891
3: -0.0096561, 0.0075284, -0.0087138, 0.0062141, -0.0158702, 0.0162421
4: -0.0102424, 0.0072933, -0.0095746, 0.0067580, -0.0170005, 0.0168678
5: 0.9910146, 1.0076587, 0.9924828, 1.0064970, -0.0154824, 0.0151759
6: -0.0072846, 0.0120171, -0.0061294, 0.0113775, -0.0186621, 0.0181465
7: -0.0259332, -0.0026206, -0.0253200, -0.0028231, -0.0231101, 0.0226993
8: -0.0075919, 0.0180551, -0.0069956, 0.0163928, -0.0239846, 0.0250508
9: -0.0073652, 0.0055799, -0.0067868, 0.0045894, -0.0119546, 0.0123667

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0311350
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0306614
time: 1.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0086731, 0.0114875, -0.0086432, 0.0129640, -0.0216372, 0.0201307
1: -0.0090737, -0.0018171, -0.0099322, -0.0012932, -0.0077806, 0.0081152
2: -0.0011641, 0.0221416, -0.0011017, 0.0243246, -0.0254887, 0.0232433
3: -0.0095152, 0.0073319, -0.0114852, 0.0100796, -0.0195948, 0.0188171
4: -0.0101426, 0.0071561, -0.0115389, 0.0090745, -0.0192171, 0.0186950
5: 0.9912341, 1.0074848, 0.9881644, 1.0099134, -0.0186793, 0.0193204
6: -0.0071118, 0.0119215, -0.0095269, 0.0132587, -0.0203705, 0.0214484
7: -0.0258415, -0.0026313, -0.0271235, -0.0026751, -0.0231664, 0.0244922
8: -0.0074706, 0.0178066, -0.0091667, 0.0212820, -0.0287526, 0.0269732
9: -0.0072787, 0.0054318, -0.0084878, 0.0075025, -0.0147813, 0.0139196

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0311350
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306614
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0083958, 0.0109839, -0.0088183, 0.0130019, -0.0213978, 0.0198022
1: -0.0087810, -0.0019958, -0.0099543, -0.0012797, -0.0075012, 0.0079585
2: -0.0005872, 0.0213971, -0.0014659, 0.0243806, -0.0249678, 0.0228630
3: -0.0088433, 0.0063948, -0.0115357, 0.0101501, -0.0189934, 0.0179305
4: -0.0096664, 0.0065307, -0.0115748, 0.0091237, -0.0187901, 0.0181054
5: 0.9922810, 1.0066566, 0.9880857, 1.0099758, -0.0176948, 0.0185709
6: -0.0062882, 0.0114655, -0.0095889, 0.0132930, -0.0195812, 0.0210543
7: -0.0254043, -0.0030370, -0.0271564, -0.0024190, -0.0229853, 0.0241194
8: -0.0068921, 0.0166212, -0.0092102, 0.0213711, -0.0282632, 0.0258314
9: -0.0068663, 0.0047255, -0.0085188, 0.0075557, -0.0144220, 0.0132444

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308833, upper bound: 0.0303039
time: 2.45 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0308649, upper bound: 0.0301871
time: 2.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0087339, 0.0119622, -0.0095800, 0.0136966, -0.0224305, 0.0215421
1: -0.0093497, -0.0016487, -0.0103582, -0.0010332, -0.0083165, 0.0087095
2: -0.0012904, 0.0228434, -0.0017323, 0.0254077, -0.0266981, 0.0245756
3: -0.0101485, 0.0082152, -0.0124626, 0.0114429, -0.0215914, 0.0206778
4: -0.0105915, 0.0077728, -0.0122317, 0.0100263, -0.0206177, 0.0200045
5: 0.9902472, 1.0082656, 0.9866413, 1.0111183, -0.0208710, 0.0216243
6: -0.0078882, 0.0123514, -0.0107251, 0.0139221, -0.0218103, 0.0230765
7: -0.0262536, -0.0025424, -0.0277596, -0.0022317, -0.0240220, 0.0252172
8: -0.0080158, 0.0189238, -0.0100081, 0.0230063, -0.0310221, 0.0289319
9: -0.0076674, 0.0060975, -0.0090877, 0.0085300, -0.0161974, 0.0151851

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307342
time: 1.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306614
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0333005, 0.0219665, -0.0084077, 0.0102160, -0.0435166, 0.0303741
1: -0.0151664, 0.0019011, -0.0083493, -0.0022682, -0.0128982, 0.0102504
2: -0.0000485, 0.0376345, -0.0006118, 0.0203888, -0.0204373, 0.0382463
3: -0.0234962, 0.0268324, -0.0078188, 0.0052459, -0.0287421, 0.0346512
4: -0.0200524, 0.0207710, -0.0090230, 0.0065491, -0.0266014, 0.0297941
5: 0.9694489, 1.0247198, 0.9931333, 1.0053937, -0.0359449, 0.0315865
6: -0.0242514, 0.0214112, -0.0050322, 0.0108281, -0.0350795, 0.0264434
7: -0.0349399, 0.0025443, -0.0247376, -0.0030197, -0.0319201, 0.0272819
8: -0.0195076, 0.0424713, -0.0068426, 0.0150821, -0.0345896, 0.0493140
9: -0.0158596, 0.0201280, -0.0062376, 0.0039036, -0.0197631, 0.0263655

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309539, upper bound: 0.0306951
time: 2.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309324, upper bound: 0.0304555
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0329706, 0.0218514, -0.0085117, 0.0122915, -0.0452621, 0.0303631
1: -0.0150995, 0.0018603, -0.0095412, -0.0015318, -0.0135677, 0.0114015
2: -0.0000414, 0.0374644, -0.0008282, 0.0233304, -0.0233718, 0.0382925
3: -0.0233428, 0.0266183, -0.0105879, 0.0088281, -0.0321709, 0.0372062
4: -0.0199436, 0.0206216, -0.0109030, 0.0082007, -0.0281443, 0.0315245
5: 0.9696879, 1.0245309, 0.9895625, 1.0088074, -0.0391195, 0.0349684
6: -0.0240633, 0.0213070, -0.0084269, 0.0126496, -0.0367129, 0.0297340
7: -0.0348400, 0.0023753, -0.0265396, -0.0028676, -0.0319724, 0.0289149
8: -0.0193754, 0.0422006, -0.0083942, 0.0196991, -0.0390745, 0.0505947
9: -0.0157654, 0.0199666, -0.0079371, 0.0065594, -0.0223248, 0.0279037

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309277, upper bound: 0.0306951
time: 1.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309066, upper bound: 0.0304555
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0359778, 0.0228999, -0.0085421, 0.0108868, -0.0468647, 0.0314420
1: -0.0157091, 0.0022323, -0.0087245, -0.0020302, -0.0136789, 0.0109568
2: -0.0007404, 0.0390145, -0.0008914, 0.0212536, -0.0219939, 0.0399058
3: -0.0247416, 0.0285694, -0.0087138, 0.0062141, -0.0309557, 0.0372831
4: -0.0209351, 0.0219838, -0.0095746, 0.0067580, -0.0276931, 0.0315583
5: 0.9675084, 1.0262550, 0.9924828, 1.0064970, -0.0389886, 0.0337722
6: -0.0257781, 0.0222565, -0.0061294, 0.0113775, -0.0371557, 0.0283859
7: -0.0357503, 0.0039155, -0.0253200, -0.0028231, -0.0329272, 0.0292354
8: -0.0205798, 0.0446683, -0.0069956, 0.0163928, -0.0369725, 0.0516640
9: -0.0166239, 0.0214370, -0.0067868, 0.0045894, -0.0212133, 0.0282239

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309752, upper bound: 0.0306951
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309589, upper bound: 0.0304566
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0356282, 0.0227780, -0.0086432, 0.0129640, -0.0485922, 0.0314211
1: -0.0156382, 0.0021890, -0.0099322, -0.0012932, -0.0143451, 0.0121213
2: -0.0007323, 0.0388342, -0.0011017, 0.0243246, -0.0250569, 0.0399359
3: -0.0245789, 0.0283425, -0.0114852, 0.0100796, -0.0346585, 0.0398277
4: -0.0208198, 0.0218254, -0.0115389, 0.0090745, -0.0298943, 0.0333643
5: 0.9677617, 1.0260544, 0.9881644, 1.0099134, -0.0421517, 0.0378900
6: -0.0255787, 0.0221461, -0.0095269, 0.0132587, -0.0388374, 0.0316730
7: -0.0356444, 0.0037364, -0.0271235, -0.0026751, -0.0329693, 0.0308599
8: -0.0204397, 0.0443814, -0.0091667, 0.0212820, -0.0417217, 0.0535480
9: -0.0165241, 0.0212660, -0.0084878, 0.0075025, -0.0240266, 0.0297538

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309461, upper bound: 0.0306951
time: 2.39 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309289, upper bound: 0.0304566
time: 2.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0369689, 0.0232454, -0.0095800, 0.0136966, -0.0506656, 0.0328254
1: -0.0159100, 0.0023549, -0.0103582, -0.0010332, -0.0148768, 0.0127131
2: -0.0008550, 0.0395253, -0.0017323, 0.0254077, -0.0262627, 0.0412575
3: -0.0252026, 0.0292124, -0.0124626, 0.0114429, -0.0366455, 0.0416749
4: -0.0212618, 0.0224327, -0.0122317, 0.0100263, -0.0312881, 0.0346644
5: 0.9667900, 1.0268233, 0.9866413, 1.0111183, -0.0443283, 0.0401820
6: -0.0263433, 0.0225694, -0.0107251, 0.0139221, -0.0402653, 0.0332945
7: -0.0360503, 0.0044230, -0.0277596, -0.0022317, -0.0338186, 0.0321826
8: -0.0209766, 0.0454816, -0.0100081, 0.0230063, -0.0439829, 0.0554897
9: -0.0169068, 0.0219216, -0.0090877, 0.0085300, -0.0254368, 0.0310093

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307658, upper bound: 0.0300699
time: 2.35 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307636, upper bound: 0.0299981
time: 2.48 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0086804, 0.0115931, -0.0338105, 0.0404262, -0.0491066, 0.0454036
1: -0.0091351, -0.0017796, -0.0152698, 0.0028960, -0.0120311, 0.0134902
2: -0.0011792, 0.0222977, -0.0004448, 0.0448669, -0.0460461, 0.0227425
3: -0.0096561, 0.0075284, -0.0237335, 0.0327712, -0.0424273, 0.0312618
4: -0.0102424, 0.0072933, -0.0202205, 0.0222552, -0.0324976, 0.0275138
5: 0.9910146, 1.0076587, 0.9299338, 1.0250121, -0.0339975, 0.0777249
6: -0.0072846, 0.0120171, -0.0282677, 0.0215722, -0.0288568, 0.0402848
7: -0.0259332, -0.0026206, -0.0350943, 0.0117715, -0.0377047, 0.0324736
8: -0.0075919, 0.0180551, -0.0197118, 0.0489113, -0.0565032, 0.0377669
9: -0.0073652, 0.0055799, -0.0160052, 0.0208785, -0.0282437, 0.0215850

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0308136
time: 2.14 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303603
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086731, 0.0114875, -0.0396239, 0.0476963, -0.0563694, 0.0511114
1: -0.0090737, -0.0018171, -0.0164482, 0.0038824, -0.0129561, 0.0146311
2: -0.0011641, 0.0221416, -0.0006762, 0.0498623, -0.0510263, 0.0228178
3: -0.0095152, 0.0073319, -0.0264375, 0.0381512, -0.0476663, 0.0337694
4: -0.0101426, 0.0071561, -0.0221372, 0.0252479, -0.0353905, 0.0292933
5: 0.9912341, 1.0074848, 0.9144933, 1.0283456, -0.0371115, 0.0929915
6: -0.0071118, 0.0119215, -0.0326511, 0.0234076, -0.0305195, 0.0445727
7: -0.0258415, -0.0026313, -0.0368540, 0.0173203, -0.0431618, 0.0342227
8: -0.0074706, 0.0178066, -0.0220399, 0.0554087, -0.0628793, 0.0398465
9: -0.0072787, 0.0054318, -0.0176648, 0.0238646, -0.0311433, 0.0230966

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0308090
time: 2.07 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303513
time: 2.10 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0359778, 0.0228999, -0.0338105, 0.0404262, -0.0764040, 0.0567104
1: -0.0157091, 0.0022323, -0.0152698, 0.0028960, -0.0186051, 0.0175021
2: -0.0007402, 0.0390145, -0.0004448, 0.0448669, -0.0456070, 0.0394592
3: -0.0247416, 0.0285694, -0.0237335, 0.0327712, -0.0575128, 0.0523028
4: -0.0209351, 0.0219838, -0.0202205, 0.0222552, -0.0431903, 0.0422043
5: 0.9675083, 1.0262550, 0.9299338, 1.0250121, -0.0575038, 0.0963212
6: -0.0257781, 0.0222565, -0.0282677, 0.0215722, -0.0473504, 0.0505242
7: -0.0357503, 0.0039155, -0.0350943, 0.0117715, -0.0475218, 0.0390097
8: -0.0205797, 0.0446683, -0.0197118, 0.0489113, -0.0694910, 0.0643801
9: -0.0166239, 0.0214370, -0.0160052, 0.0208785, -0.0375024, 0.0374422

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305470, upper bound: 0.0306892
time: 2.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305470, upper bound: 0.0304566
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0356282, 0.0227780, -0.0396239, 0.0476963, -0.0833244, 0.0624018
1: -0.0156383, 0.0021890, -0.0164482, 0.0038824, -0.0195206, 0.0186372
2: -0.0007321, 0.0388342, -0.0006762, 0.0498623, -0.0505943, 0.0395104
3: -0.0245789, 0.0283425, -0.0264375, 0.0381512, -0.0627301, 0.0547800
4: -0.0208198, 0.0218254, -0.0221372, 0.0252479, -0.0460677, 0.0439625
5: 0.9677618, 1.0260544, 0.9144933, 1.0283456, -0.0605838, 0.1115611
6: -0.0255787, 0.0221461, -0.0326511, 0.0234076, -0.0489864, 0.0547972
7: -0.0356444, 0.0037364, -0.0368540, 0.0173203, -0.0529648, 0.0405903
8: -0.0204397, 0.0443814, -0.0220399, 0.0554087, -0.0758484, 0.0664213
9: -0.0165241, 0.0212660, -0.0176648, 0.0238646, -0.0403887, 0.0389308

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304566, upper bound: 0.0306886
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304566, upper bound: 0.0304566
time: 1.97 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0086804, 0.0115931, -0.0395405, 0.0475920, -0.0562724, 0.0511336
1: -0.0091351, -0.0017796, -0.0164313, 0.0038682, -0.0130034, 0.0146517
2: -0.0011792, 0.0222977, -0.0011687, 0.0497906, -0.0509698, 0.0234664
3: -0.0096561, 0.0075284, -0.0263988, 0.0380740, -0.0477300, 0.0339272
4: -0.0102424, 0.0072933, -0.0221097, 0.0252050, -0.0354474, 0.0294030
5: 0.9910146, 1.0076587, 0.9147148, 1.0282978, -0.0372832, 0.0929440
6: -0.0072846, 0.0120171, -0.0325883, 0.0233813, -0.0306658, 0.0446054
7: -0.0259332, -0.0026206, -0.0368287, 0.0172407, -0.0431739, 0.0342081
8: -0.0075919, 0.0180551, -0.0220065, 0.0553155, -0.0629074, 0.0400616
9: -0.0073652, 0.0055799, -0.0176410, 0.0238218, -0.0311870, 0.0232209

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0308388
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303712
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086731, 0.0114875, -0.0448240, 0.0541993, -0.0628724, 0.0563115
1: -0.0090737, -0.0018171, -0.0175023, 0.0047647, -0.0138384, 0.0156852
2: -0.0011641, 0.0221416, -0.0013582, 0.0543306, -0.0554947, 0.0234998
3: -0.0095152, 0.0073319, -0.0288564, 0.0429636, -0.0524787, 0.0361882
4: -0.0101426, 0.0071561, -0.0238516, 0.0279249, -0.0380675, 0.0310077
5: 0.9912341, 1.0074848, 0.9006816, 1.0313275, -0.0400934, 0.1068032
6: -0.0071118, 0.0119215, -0.0365722, 0.0250494, -0.0321613, 0.0484937
7: -0.0258415, -0.0026313, -0.0384280, 0.0222837, -0.0481252, 0.0357967
8: -0.0074706, 0.0178066, -0.0241223, 0.0612207, -0.0686913, 0.0419289
9: -0.0072787, 0.0054318, -0.0191493, 0.0265357, -0.0338145, 0.0245811

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0308317
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303593
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0359778, 0.0228999, -0.0395405, 0.0475920, -0.0835698, 0.0624404
1: -0.0157091, 0.0022323, -0.0164313, 0.0038682, -0.0195774, 0.0186636
2: -0.0007402, 0.0390145, -0.0011687, 0.0497906, -0.0505308, 0.0401832
3: -0.0247416, 0.0285694, -0.0263988, 0.0380740, -0.0628156, 0.0549682
4: -0.0209351, 0.0219838, -0.0221097, 0.0252050, -0.0461401, 0.0440934
5: 0.9675083, 1.0262550, 0.9147148, 1.0282978, -0.0607895, 0.1115403
6: -0.0257781, 0.0222565, -0.0325883, 0.0233813, -0.0491594, 0.0548448
7: -0.0357503, 0.0039155, -0.0368287, 0.0172407, -0.0529910, 0.0407442
8: -0.0205797, 0.0446683, -0.0220065, 0.0553155, -0.0758953, 0.0666748
9: -0.0166239, 0.0214370, -0.0176410, 0.0238218, -0.0404457, 0.0390780

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0306892
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0304566
time: 1.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0356282, 0.0227780, -0.0448240, 0.0541993, -0.0898274, 0.0676019
1: -0.0156383, 0.0021890, -0.0175023, 0.0047647, -0.0204029, 0.0196913
2: -0.0007321, 0.0388342, -0.0013582, 0.0543306, -0.0550627, 0.0401924
3: -0.0245789, 0.0283425, -0.0288564, 0.0429636, -0.0675425, 0.0571989
4: -0.0208198, 0.0218254, -0.0238516, 0.0279249, -0.0487446, 0.0456770
5: 0.9677618, 1.0260544, 0.9006816, 1.0313275, -0.0635657, 0.1253728
6: -0.0255787, 0.0221461, -0.0365722, 0.0250494, -0.0506281, 0.0587182
7: -0.0356444, 0.0037364, -0.0384280, 0.0222837, -0.0579282, 0.0421644
8: -0.0204397, 0.0443814, -0.0241223, 0.0612207, -0.0816604, 0.0685037
9: -0.0165241, 0.0212660, -0.0191493, 0.0265357, -0.0430598, 0.0404154

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304250, upper bound: 0.0306886
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304250, upper bound: 0.0304566
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

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

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306815, upper bound: 0.0311350
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305343, upper bound: 0.0301871
time: 2.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

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

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306343, upper bound: 0.0311350
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304978, upper bound: 0.0301871
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090338, 0.0135062, -0.0085421, 0.0108868, -0.0199206, 0.0220483
1: -0.0102475, -0.0011008, -0.0087245, -0.0020302, -0.0082173, 0.0076237
2: -0.0019038, 0.0251262, -0.0008914, 0.0212536, -0.0231573, 0.0260176
3: -0.0122086, 0.0110886, -0.0087138, 0.0062141, -0.0184227, 0.0198023
4: -0.0120517, 0.0097789, -0.0095746, 0.0067580, -0.0188097, 0.0193535
5: 0.9870372, 1.0108054, 0.9924828, 1.0064970, -0.0194598, 0.0183226
6: -0.0104137, 0.0137497, -0.0061294, 0.0113775, -0.0217913, 0.0198790
7: -0.0275943, -0.0021110, -0.0253200, -0.0028231, -0.0247712, 0.0232089
8: -0.0097895, 0.0225581, -0.0069956, 0.0163928, -0.0261822, 0.0295538
9: -0.0089318, 0.0082629, -0.0067868, 0.0045894, -0.0135211, 0.0150498

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0311350
time: 2.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0306599
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090193, 0.0133909, -0.0086432, 0.0129640, -0.0219833, 0.0220341
1: -0.0101804, -0.0011417, -0.0099322, -0.0012932, -0.0088872, 0.0087905
2: -0.0018841, 0.0249557, -0.0011017, 0.0243246, -0.0262087, 0.0260574
3: -0.0120547, 0.0108739, -0.0114852, 0.0100796, -0.0221343, 0.0223591
4: -0.0119426, 0.0096291, -0.0115389, 0.0090745, -0.0210171, 0.0211680
5: 0.9872769, 1.0106156, 0.9881644, 1.0099134, -0.0226365, 0.0224512
6: -0.0102251, 0.0136452, -0.0095269, 0.0132587, -0.0234837, 0.0231721
7: -0.0274941, -0.0021249, -0.0271235, -0.0026751, -0.0248190, 0.0249987
8: -0.0096570, 0.0222867, -0.0091667, 0.0212820, -0.0309389, 0.0314533
9: -0.0088373, 0.0081012, -0.0084878, 0.0075025, -0.0163399, 0.0165890

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0311350
time: 1.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306599
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0086902, 0.0125613, -0.0318269, 0.0214527, -0.0301429, 0.0443882
1: -0.0096981, -0.0014361, -0.0148677, 0.0017188, -0.0114169, 0.0134316
2: -0.0011994, 0.0237292, -0.0001649, 0.0368748, -0.0380743, 0.0238941
3: -0.0109478, 0.0093301, -0.0228108, 0.0258763, -0.0368241, 0.0321409
4: -0.0111580, 0.0085512, -0.0195665, 0.0201035, -0.0312616, 0.0281177
5: 0.9890018, 1.0092509, 0.9705168, 1.0238750, -0.0348732, 0.0387341
6: -0.0088682, 0.0128939, -0.0234111, 0.0209459, -0.0298141, 0.0363050
7: -0.0267738, -0.0026064, -0.0344938, 0.0017896, -0.0285634, 0.0318874
8: -0.0087040, 0.0203340, -0.0189174, 0.0412621, -0.0499661, 0.0392514
9: -0.0081580, 0.0069377, -0.0154389, 0.0194075, -0.0275655, 0.0223766

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299960, upper bound: 0.0306805
time: 1.92 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0297564, upper bound: 0.0297778
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086825, 0.0124519, -0.0376893, 0.0234965, -0.0321791, 0.0501412
1: -0.0096345, -0.0014749, -0.0160560, 0.0024440, -0.0120785, 0.0145811
2: -0.0011836, 0.0235674, -0.0004012, 0.0398966, -0.0410802, 0.0239686
3: -0.0108019, 0.0091265, -0.0255376, 0.0296797, -0.0404816, 0.0346642
4: -0.0110546, 0.0084091, -0.0214993, 0.0227590, -0.0338135, 0.0299084
5: 0.9892291, 1.0090712, 0.9662679, 1.0272365, -0.0380074, 0.0428033
6: -0.0086892, 0.0127949, -0.0267540, 0.0227968, -0.0314860, 0.0395489
7: -0.0266788, -0.0026175, -0.0362683, 0.0047919, -0.0314708, 0.0336508
8: -0.0085784, 0.0200765, -0.0212651, 0.0460727, -0.0546511, 0.0413416
9: -0.0080684, 0.0067843, -0.0171125, 0.0222738, -0.0303422, 0.0238968

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299473, upper bound: 0.0306757
time: 1.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0297170, upper bound: 0.0297602
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090338, 0.0135062, -0.0338006, 0.0221408, -0.0311746, 0.0473068
1: -0.0102475, -0.0011008, -0.0152678, 0.0019629, -0.0122104, 0.0141670
2: -0.0019038, 0.0251262, -0.0004448, 0.0378922, -0.0397959, 0.0255710
3: -0.0122086, 0.0110886, -0.0237288, 0.0271568, -0.0393654, 0.0348174
4: -0.0120517, 0.0097789, -0.0202172, 0.0209975, -0.0330492, 0.0299962
5: 0.9870372, 1.0108054, 0.9690865, 1.0250065, -0.0379694, 0.0417189
6: -0.0104137, 0.0137497, -0.0245366, 0.0215691, -0.0319828, 0.0382862
7: -0.0275943, -0.0021110, -0.0350912, 0.0028004, -0.0303946, 0.0329802
8: -0.0097895, 0.0225581, -0.0197078, 0.0428816, -0.0526711, 0.0422660
9: -0.0089318, 0.0082629, -0.0160023, 0.0203725, -0.0293042, 0.0242653

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0307658
time: 2.06 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303524
time: 1.98 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090193, 0.0133909, -0.0396239, 0.0241710, -0.0331903, 0.0530148
1: -0.0101804, -0.0011417, -0.0164482, 0.0026833, -0.0128637, 0.0153065
2: -0.0018841, 0.0249557, -0.0006762, 0.0408938, -0.0427779, 0.0256319
3: -0.0120547, 0.0108739, -0.0264375, 0.0309349, -0.0429895, 0.0373115
4: -0.0119426, 0.0096291, -0.0221372, 0.0236353, -0.0355779, 0.0317663
5: 0.9872769, 1.0106156, 0.9648657, 1.0283456, -0.0410687, 0.0457498
6: -0.0102251, 0.0136452, -0.0278572, 0.0234076, -0.0336327, 0.0415024
7: -0.0274941, -0.0021249, -0.0368540, 0.0057827, -0.0332769, 0.0347291
8: -0.0096570, 0.0222867, -0.0220399, 0.0476603, -0.0573172, 0.0443265
9: -0.0088373, 0.0081012, -0.0176648, 0.0232197, -0.0320570, 0.0257660

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0307636
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303437
time: 1.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087440, 0.0129368, -0.0088183, 0.0130019, -0.0217460, 0.0217551
1: -0.0099164, -0.0013028, -0.0099543, -0.0012797, -0.0086367, 0.0086514
2: -0.0013116, 0.0242844, -0.0014659, 0.0243806, -0.0256922, 0.0257504
3: -0.0114489, 0.0100290, -0.0115357, 0.0101501, -0.0215990, 0.0215647
4: -0.0115132, 0.0090391, -0.0115748, 0.0091237, -0.0206369, 0.0206139
5: 0.9882211, 1.0098687, 0.9880857, 1.0099758, -0.0217547, 0.0217830
6: -0.0094824, 0.0132340, -0.0095889, 0.0132930, -0.0227754, 0.0228229
7: -0.0270999, -0.0025275, -0.0271564, -0.0024190, -0.0246809, 0.0246289
8: -0.0091354, 0.0212179, -0.0092102, 0.0213711, -0.0305065, 0.0304281
9: -0.0084655, 0.0074644, -0.0085188, 0.0075557, -0.0160212, 0.0159832

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309711, upper bound: 0.0303039
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0309536, upper bound: 0.0301871
time: 2.46 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0101091, 0.0138811, -0.0095800, 0.0136966, -0.0238057, 0.0234611
1: -0.0104654, -0.0009678, -0.0103582, -0.0010332, -0.0094322, 0.0093904
2: -0.0020135, 0.0256805, -0.0017323, 0.0254077, -0.0274212, 0.0274127
3: -0.0127087, 0.0117862, -0.0124626, 0.0114429, -0.0241516, 0.0242488
4: -0.0124062, 0.0102660, -0.0122317, 0.0100263, -0.0224325, 0.0224977
5: 0.9862578, 1.0114218, 0.9866413, 1.0111183, -0.0248605, 0.0247805
6: -0.0110269, 0.0140891, -0.0107251, 0.0139221, -0.0249489, 0.0248143
7: -0.0279197, -0.0020339, -0.0277596, -0.0022317, -0.0256881, 0.0257257
8: -0.0102201, 0.0234405, -0.0100081, 0.0230063, -0.0332263, 0.0334486
9: -0.0092388, 0.0087887, -0.0090877, 0.0085300, -0.0177687, 0.0178763

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307340
time: 1.86 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306599
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0086902, 0.0125613, -0.0375607, 0.0451160, -0.0538062, 0.0501220
1: -0.0096981, -0.0014361, -0.0160300, 0.0035323, -0.0132304, 0.0145939
2: -0.0011994, 0.0237292, -0.0008895, 0.0480894, -0.0492888, 0.0246187
3: -0.0109478, 0.0093301, -0.0254779, 0.0362418, -0.0471896, 0.0348080
4: -0.0111580, 0.0085512, -0.0214570, 0.0241858, -0.0353438, 0.0300082
5: 0.9890018, 1.0092509, 0.9199732, 1.0271627, -0.0381609, 0.0892777
6: -0.0088682, 0.0128939, -0.0310955, 0.0227562, -0.0316244, 0.0439894
7: -0.0267738, -0.0026064, -0.0362294, 0.0153511, -0.0421249, 0.0336230
8: -0.0087040, 0.0203340, -0.0212137, 0.0531028, -0.0618068, 0.0415476
9: -0.0081580, 0.0069377, -0.0170758, 0.0228048, -0.0309628, 0.0240135

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299959, upper bound: 0.0306805
time: 2.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0297564, upper bound: 0.0297802
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0086825, 0.0124519, -0.0428730, 0.0517595, -0.0604421, 0.0553250
1: -0.0096345, -0.0014749, -0.0171068, 0.0044337, -0.0140681, 0.0156319
2: -0.0011836, 0.0235674, -0.0010797, 0.0526542, -0.0538378, 0.0246472
3: -0.0108019, 0.0091265, -0.0279489, 0.0411581, -0.0519600, 0.0370754
4: -0.0110546, 0.0084091, -0.0232084, 0.0269206, -0.0379751, 0.0316175
5: 0.9892291, 1.0090712, 0.9058634, 1.0302089, -0.0409799, 0.1032078
6: -0.0086892, 0.0127949, -0.0351011, 0.0244334, -0.0331227, 0.0478960
7: -0.0266788, -0.0026175, -0.0378375, 0.0204216, -0.0471004, 0.0352200
8: -0.0085784, 0.0200765, -0.0233411, 0.0590402, -0.0676185, 0.0434176
9: -0.0080684, 0.0067843, -0.0185924, 0.0255336, -0.0336020, 0.0253767

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299473, upper bound: 0.0306757
time: 1.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0297170, upper bound: 0.0297629
time: 2.15 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090338, 0.0135062, -0.0395405, 0.0475920, -0.0566257, 0.0530467
1: -0.0102475, -0.0011008, -0.0164313, 0.0038682, -0.0141157, 0.0153305
2: -0.0019038, 0.0251262, -0.0011687, 0.0497906, -0.0516944, 0.0262950
3: -0.0122086, 0.0110886, -0.0263988, 0.0380740, -0.0502825, 0.0374874
4: -0.0120517, 0.0097789, -0.0221097, 0.0252050, -0.0372567, 0.0318886
5: 0.9870372, 1.0108054, 0.9147148, 1.0282978, -0.0412606, 0.0960906
6: -0.0104137, 0.0137497, -0.0325883, 0.0233813, -0.0337950, 0.0463379
7: -0.0275943, -0.0021110, -0.0368287, 0.0172407, -0.0448350, 0.0347177
8: -0.0097895, 0.0225581, -0.0220065, 0.0553155, -0.0651050, 0.0445646
9: -0.0089318, 0.0082629, -0.0176410, 0.0238218, -0.0327536, 0.0259039

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0307658
time: 2.42 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303525
time: 1.97 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090193, 0.0133909, -0.0448240, 0.0541993, -0.0632186, 0.0582149
1: -0.0101804, -0.0011417, -0.0175023, 0.0047647, -0.0149451, 0.0163606
2: -0.0018841, 0.0249557, -0.0013582, 0.0543306, -0.0562147, 0.0263139
3: -0.0120547, 0.0108739, -0.0288564, 0.0429636, -0.0550182, 0.0397303
4: -0.0119426, 0.0096291, -0.0238516, 0.0279249, -0.0398675, 0.0334807
5: 0.9872769, 1.0106156, 0.9006816, 1.0313275, -0.0440506, 0.1099340
6: -0.0102251, 0.0136452, -0.0365722, 0.0250494, -0.0352745, 0.0502174
7: -0.0274941, -0.0021249, -0.0384280, 0.0222837, -0.0497778, 0.0363032
8: -0.0096570, 0.0222867, -0.0241223, 0.0612207, -0.0708776, 0.0464090
9: -0.0088373, 0.0081012, -0.0191493, 0.0265357, -0.0353731, 0.0272505

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0307636
time: 1.96 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303437
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0395405, 0.0475920, -0.0090338, 0.0135062, -0.0530467, 0.0566257
1: -0.0164313, 0.0038682, -0.0102475, -0.0011008, -0.0153305, 0.0141157
2: -0.0011687, 0.0497906, -0.0019038, 0.0251262, -0.0262950, 0.0516944
3: -0.0263988, 0.0380740, -0.0122086, 0.0110886, -0.0374874, 0.0502825
4: -0.0221097, 0.0252050, -0.0120517, 0.0097789, -0.0318886, 0.0372567
5: 0.9147148, 1.0282978, 0.9870372, 1.0108054, -0.0960906, 0.0412606
6: -0.0325883, 0.0233813, -0.0104137, 0.0137497, -0.0463379, 0.0337950
7: -0.0368287, 0.0172407, -0.0275943, -0.0021110, -0.0347177, 0.0448350
8: -0.0220065, 0.0553155, -0.0097895, 0.0225581, -0.0445646, 0.0651050
9: -0.0176410, 0.0238218, -0.0089318, 0.0082629, -0.0259039, 0.0327536

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0306134
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0300699
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0395405, 0.0475920, -0.0418136, 0.0249344, -0.0644749, 0.0894055
1: -0.0164313, 0.0038682, -0.0168920, 0.0029542, -0.0193855, 0.0207603
2: -0.0011687, 0.0497906, -0.0014346, 0.0420224, -0.0431911, 0.0512252
3: -0.0263988, 0.0380740, -0.0274560, 0.0323555, -0.0587542, 0.0655300
4: -0.0221097, 0.0252050, -0.0228591, 0.0246271, -0.0467368, 0.0480641
5: 0.9147148, 1.0282978, 0.9632787, 1.0296011, -0.1148863, 0.0650191
6: -0.0325883, 0.0233813, -0.0291058, 0.0240989, -0.0566872, 0.0524871
7: -0.0368287, 0.0172407, -0.0375168, 0.0069042, -0.0437329, 0.0547575
8: -0.0220065, 0.0553155, -0.0229167, 0.0494570, -0.0714635, 0.0782323
9: -0.0176410, 0.0238218, -0.0182899, 0.0242903, -0.0419313, 0.0421117

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0306134
time: 2.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0300699
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0448240, 0.0541993, -0.0090193, 0.0133909, -0.0582149, 0.0632186
1: -0.0175023, 0.0047647, -0.0101804, -0.0011417, -0.0163606, 0.0149451
2: -0.0013582, 0.0543306, -0.0018841, 0.0249557, -0.0263139, 0.0562147
3: -0.0288564, 0.0429636, -0.0120547, 0.0108739, -0.0397303, 0.0550182
4: -0.0238516, 0.0279249, -0.0119426, 0.0096291, -0.0334807, 0.0398675
5: 0.9006816, 1.0313275, 0.9872769, 1.0106156, -0.1099340, 0.0440506
6: -0.0365722, 0.0250494, -0.0102251, 0.0136452, -0.0502174, 0.0352745
7: -0.0384280, 0.0222837, -0.0274941, -0.0021249, -0.0363032, 0.0497778
8: -0.0241223, 0.0612207, -0.0096570, 0.0222867, -0.0464090, 0.0708776
9: -0.0191493, 0.0265357, -0.0088373, 0.0081012, -0.0272505, 0.0353731

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0305577
time: 1.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0299981
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0448240, 0.0541993, -0.0414417, 0.0248047, -0.0696287, 0.0956409
1: -0.0175023, 0.0047647, -0.0168167, 0.0029082, -0.0204105, 0.0215814
2: -0.0013582, 0.0543306, -0.0014169, 0.0418308, -0.0431890, 0.0557475
3: -0.0288564, 0.0429636, -0.0272831, 0.0321142, -0.0609706, 0.0702466
4: -0.0238516, 0.0279249, -0.0227365, 0.0244587, -0.0483103, 0.0506614
5: 0.9006816, 1.0313275, 0.9635482, 1.0293881, -0.1287065, 0.0677793
6: -0.0365722, 0.0250494, -0.0288938, 0.0239815, -0.0605537, 0.0539432
7: -0.0384280, 0.0222837, -0.0374042, 0.0067137, -0.0451418, 0.0596879
8: -0.0241223, 0.0612207, -0.0227678, 0.0491519, -0.0732742, 0.0839885
9: -0.0191493, 0.0265357, -0.0181837, 0.0241085, -0.0432579, 0.0447195

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0305577
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0299981
time: 1.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.14 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306810, upper bound: 0.0311350
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0304971, upper bound: 0.0301871
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306341, upper bound: 0.0311350
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0304644, upper bound: 0.0301871
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0306614
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0311350
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306614
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0308833, upper bound: 0.0303039
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0308649, upper bound: 0.0301871
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307342
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306614
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309539, upper bound: 0.0306951
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309324, upper bound: 0.0304555
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309277, upper bound: 0.0306951
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309066, upper bound: 0.0304555
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309752, upper bound: 0.0306951
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309589, upper bound: 0.0304566
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309461, upper bound: 0.0306951
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309289, upper bound: 0.0304566
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0307658, upper bound: 0.0300699
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0307636, upper bound: 0.0299981
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0308136
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303603
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0308090
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303513
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0305470, upper bound: 0.0306892
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0305470, upper bound: 0.0304566
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0304566, upper bound: 0.0306886
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0304566, upper bound: 0.0304566
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0308388
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303712
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0308317
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303593
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0306892
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0305134, upper bound: 0.0304566
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0304250, upper bound: 0.0306886
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0304250, upper bound: 0.0304566
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306815, upper bound: 0.0311350
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0305343, upper bound: 0.0301871
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306343, upper bound: 0.0311350
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0304978, upper bound: 0.0301871
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0311350
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0306599
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0311350
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306599
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299960, upper bound: 0.0306805
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0297564, upper bound: 0.0297778
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299473, upper bound: 0.0306757
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0297170, upper bound: 0.0297602
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0307658
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303524
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0307636
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303437
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309711, upper bound: 0.0303039
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0309536, upper bound: 0.0301871
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307340
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306599
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299959, upper bound: 0.0306805
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0297564, upper bound: 0.0297802
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299473, upper bound: 0.0306757
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0297170, upper bound: 0.0297629
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0307658
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0300699, upper bound: 0.0303525
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0307636
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0303437
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0306134
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0300699
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0306134
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0300699
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0305577
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0299981
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0305577
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.14
Output dim: 5, lower bound: -0.0299981, upper bound: 0.0299981

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0081426, 0.0097516, -0.0084045, 0.0102046, -0.0183472, 0.0181561
1: -0.0080978, -0.0024330, -0.0083431, -0.0022723, -0.0058255, 0.0059100
2: -0.0000604, 0.0198621, -0.0006053, 0.0203759, -0.0204363, 0.0204673
3: -0.0071991, 0.0047343, -0.0078035, 0.0052333, -0.0124324, 0.0125378
4: -0.0086882, 0.0061370, -0.0090148, 0.0065442, -0.0152324, 0.0151518
5: 0.9931620, 1.0046300, 0.9931341, 1.0053750, -0.0122130, 0.0114959
6: -0.0042726, 0.0104806, -0.0050135, 0.0108196, -0.0150921, 0.0154941
7: -0.0243343, -0.0034075, -0.0247276, -0.0030243, -0.0213100, 0.0213201
8: -0.0065408, 0.0143266, -0.0068390, 0.0150635, -0.0216043, 0.0211656
9: -0.0058572, 0.0035731, -0.0062282, 0.0038954, -0.0097526, 0.0098013

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304971, upper bound: 0.0301871
time: 2.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304971, upper bound: 0.0301871
time: 1.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0081363, 0.0096544, -0.0085085, 0.0122803, -0.0204167, 0.0181630
1: -0.0080452, -0.0024675, -0.0095347, -0.0015358, -0.0065095, 0.0070672
2: -0.0000474, 0.0197519, -0.0008216, 0.0233138, -0.0233612, 0.0205735
3: -0.0070695, 0.0046273, -0.0105730, 0.0088073, -0.0158768, 0.0152002
4: -0.0086181, 0.0061273, -0.0108924, 0.0081862, -0.0168043, 0.0170196
5: 0.9931681, 1.0044701, 0.9895858, 1.0087891, -0.0156210, 0.0148844
6: -0.0041137, 0.0104079, -0.0084086, 0.0126395, -0.0167531, 0.0188165
7: -0.0242500, -0.0034167, -0.0265299, -0.0028721, -0.0213778, 0.0231132
8: -0.0065337, 0.0141685, -0.0083813, 0.0196727, -0.0262063, 0.0225498
9: -0.0057777, 0.0035040, -0.0079279, 0.0065437, -0.0123213, 0.0114320

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304644, upper bound: 0.0301871
time: 1.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304644, upper bound: 0.0301871
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0084619, 0.0107531, -0.0085389, 0.0108754, -0.0193373, 0.0192920
1: -0.0086467, -0.0020777, -0.0087179, -0.0020343, -0.0066125, 0.0066402
2: -0.0007246, 0.0210558, -0.0008848, 0.0212366, -0.0219612, 0.0219406
3: -0.0085353, 0.0059652, -0.0086985, 0.0061928, -0.0147281, 0.0146637
4: -0.0094481, 0.0066333, -0.0095637, 0.0067531, -0.0162011, 0.0161970
5: 0.9927609, 1.0062770, 0.9925066, 1.0064782, -0.0137173, 0.0137703
6: -0.0059106, 0.0112564, -0.0061107, 0.0113672, -0.0172778, 0.0173671
7: -0.0252038, -0.0029404, -0.0253100, -0.0028277, -0.0223761, 0.0223696
8: -0.0069043, 0.0160779, -0.0069920, 0.0163658, -0.0232702, 0.0230700
9: -0.0066773, 0.0044018, -0.0067775, 0.0045733, -0.0112507, 0.0111793

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0306614
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0307342, upper bound: 0.0306614
time: 1.87 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0084546, 0.0106453, -0.0086400, 0.0129529, -0.0214075, 0.0192852
1: -0.0085841, -0.0021159, -0.0099257, -0.0012972, -0.0072869, 0.0078098
2: -0.0007094, 0.0208964, -0.0010951, 0.0243081, -0.0250175, 0.0219915
3: -0.0083915, 0.0057646, -0.0114702, 0.0100588, -0.0184503, 0.0172348
4: -0.0093461, 0.0066220, -0.0115283, 0.0090599, -0.0184060, 0.0181504
5: 0.9929851, 1.0060996, 0.9881876, 1.0098951, -0.0169100, 0.0179120
6: -0.0057343, 0.0111588, -0.0095086, 0.0132485, -0.0189828, 0.0206674
7: -0.0251102, -0.0029511, -0.0271138, -0.0026798, -0.0224304, 0.0241627
8: -0.0068960, 0.0158242, -0.0091538, 0.0212556, -0.0281516, 0.0249780
9: -0.0065890, 0.0042506, -0.0084786, 0.0074869, -0.0140759, 0.0127292

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306614
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0306614, upper bound: 0.0306614
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0081173, 0.0092609, -0.0087620, 0.0126265, -0.0207438, 0.0180229
1: -0.0078322, -0.0026071, -0.0097360, -0.0014130, -0.0064192, 0.0071288
2: -0.0000078, 0.0193056, -0.0013488, 0.0238255, -0.0238334, 0.0206544
3: -0.0065444, 0.0041938, -0.0110348, 0.0094514, -0.0159958, 0.0152286
4: -0.0083344, 0.0060977, -0.0112197, 0.0086359, -0.0169703, 0.0173174
5: 0.9931924, 1.0038228, 0.9888662, 1.0093582, -0.0161658, 0.0149567
6: -0.0034700, 0.0101134, -0.0089748, 0.0129530, -0.0164229, 0.0190882
7: -0.0239083, -0.0034445, -0.0268304, -0.0025014, -0.0214069, 0.0233859
8: -0.0065120, 0.0135284, -0.0087789, 0.0204874, -0.0269994, 0.0223073
9: -0.0054554, 0.0032241, -0.0082114, 0.0070291, -0.0124846, 0.0114355

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305499, upper bound: 0.0299093
time: 2.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305111, upper bound: 0.0299093
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0081966, 0.0113071, -0.0087535, 0.0125126, -0.0207092, 0.0200606
1: -0.0089689, -0.0018811, -0.0096698, -0.0014534, -0.0075155, 0.0077887
2: -0.0001727, 0.0218750, -0.0013312, 0.0236572, -0.0238300, 0.0232062
3: -0.0092745, 0.0069962, -0.0108829, 0.0092395, -0.0185141, 0.0178791
4: -0.0099720, 0.0069217, -0.0111120, 0.0084880, -0.0184600, 0.0180337
5: 0.9916091, 1.0071883, 0.9891030, 1.0091710, -0.0175619, 0.0180854
6: -0.0068168, 0.0117582, -0.0087885, 0.0128498, -0.0196667, 0.0205467
7: -0.0256849, -0.0033285, -0.0267316, -0.0025137, -0.0231712, 0.0234031
8: -0.0072634, 0.0173820, -0.0086481, 0.0202194, -0.0274828, 0.0260301
9: -0.0071310, 0.0051788, -0.0081181, 0.0068695, -0.0140005, 0.0132970

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0305219, upper bound: 0.0298009
time: 2.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0304826, upper bound: 0.0298009
time: 2.19 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0084586, 0.0102373, -0.0088895, 0.0133214, -0.0217800, 0.0191269
1: -0.0083608, -0.0022607, -0.0101400, -0.0011664, -0.0071944, 0.0078793
2: -0.0007179, 0.0204130, -0.0016142, 0.0248529, -0.0255708, 0.0220272
3: -0.0078472, 0.0052694, -0.0119619, 0.0107446, -0.0185918, 0.0172313
4: -0.0090384, 0.0066283, -0.0118768, 0.0095387, -0.0185771, 0.0185052
5: 0.9931321, 1.0054289, 0.9874215, 1.0105011, -0.0173691, 0.0180074
6: -0.0050671, 0.0108441, -0.0101114, 0.0135823, -0.0186493, 0.0209554
7: -0.0247561, -0.0029451, -0.0274338, -0.0023147, -0.0224414, 0.0244886
8: -0.0069007, 0.0151167, -0.0095771, 0.0221230, -0.0290237, 0.0246938
9: -0.0062550, 0.0039187, -0.0087804, 0.0080037, -0.0142587, 0.0126991

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306815
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0307342
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0085586, 0.0122914, -0.0088823, 0.0132075, -0.0217661, 0.0211737
1: -0.0095411, -0.0015319, -0.0100738, -0.0012068, -0.0083343, 0.0085419
2: -0.0009257, 0.0233301, -0.0015992, 0.0246846, -0.0256102, 0.0249293
3: -0.0105877, 0.0088278, -0.0118100, 0.0105326, -0.0211203, 0.0206378
4: -0.0109028, 0.0082005, -0.0117692, 0.0093908, -0.0202936, 0.0199697
5: 0.9895629, 1.0088071, 0.9876583, 1.0103139, -0.0207510, 0.0211487
6: -0.0084267, 0.0126495, -0.0099251, 0.0134791, -0.0219058, 0.0225746
7: -0.0265395, -0.0027990, -0.0273349, -0.0023253, -0.0242142, 0.0245359
8: -0.0083940, 0.0196987, -0.0094463, 0.0218550, -0.0302489, 0.0291450
9: -0.0079370, 0.0065592, -0.0086871, 0.0078440, -0.0157809, 0.0152463

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306343
time: 1.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0311350, upper bound: 0.0306614
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0262743, 0.0195169, -0.0084031, 0.0097763, -0.0360506, 0.0279200
1: -0.0137422, 0.0010319, -0.0081112, -0.0024243, -0.0113179, 0.0091431
2: -0.0000074, 0.0340128, -0.0006023, 0.0198901, -0.0198975, 0.0346151
3: -0.0202280, 0.0222739, -0.0072321, 0.0047615, -0.0249895, 0.0295060
4: -0.0177359, 0.0175884, -0.0087060, 0.0065420, -0.0242778, 0.0262944
5: 0.9745414, 1.0206909, 0.9931606, 1.0046705, -0.0301291, 0.0275303
6: -0.0202449, 0.0191929, -0.0043130, 0.0104991, -0.0307439, 0.0235058
7: -0.0328130, -0.0010541, -0.0243557, -0.0030264, -0.0297866, 0.0233016
8: -0.0166938, 0.0367057, -0.0068374, 0.0143668, -0.0310605, 0.0435431
9: -0.0138537, 0.0166926, -0.0058775, 0.0035907, -0.0174444, 0.0225700

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295548, upper bound: 0.0288413
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0292062, upper bound: 0.0288411
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0259741, 0.0194122, -0.0083993, 0.0092365, -0.0352106, 0.0278116
1: -0.0136813, 0.0009948, -0.0078184, -0.0026120, -0.0110693, 0.0088132
2: -0.0000986, 0.0338581, -0.0005945, 0.0192878, -0.0193863, 0.0344525
3: -0.0200883, 0.0220791, -0.0065104, 0.0041797, -0.0242681, 0.0285896
4: -0.0176369, 0.0174524, -0.0083226, 0.0065361, -0.0241730, 0.0257750
5: 0.9747589, 1.0205188, 0.9931940, 1.0037847, -0.0290257, 0.0273248
6: -0.0200736, 0.0190981, -0.0034386, 0.0100944, -0.0301680, 0.0225367
7: -0.0327221, -0.0012079, -0.0238861, -0.0030319, -0.0296903, 0.0226782
8: -0.0165735, 0.0364593, -0.0068331, 0.0134869, -0.0300605, 0.0432924
9: -0.0137680, 0.0165458, -0.0054430, 0.0032060, -0.0169740, 0.0219887

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0286424, upper bound: 0.0282030
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0282991, upper bound: 0.0282030
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0259456, 0.0194023, -0.0085072, 0.0118440, -0.0377896, 0.0279095
1: -0.0136756, 0.0009913, -0.0092810, -0.0016906, -0.0119849, 0.0102723
2: 0.0000005, 0.0338434, -0.0008188, 0.0226687, -0.0226682, 0.0346622
3: -0.0200751, 0.0220606, -0.0099908, 0.0079953, -0.0280703, 0.0320514
4: -0.0176275, 0.0174394, -0.0104797, 0.0076192, -0.0252467, 0.0279192
5: 0.9747797, 1.0205024, 0.9904930, 1.0080712, -0.0332915, 0.0300094
6: -0.0200574, 0.0190891, -0.0076949, 0.0122443, -0.0323017, 0.0267840
7: -0.0327135, -0.0012225, -0.0261510, -0.0028741, -0.0298394, 0.0249285
8: -0.0165621, 0.0364359, -0.0078801, 0.0186457, -0.0352078, 0.0443159
9: -0.0137598, 0.0165318, -0.0075706, 0.0059317, -0.0196916, 0.0241025

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 156

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0295411, upper bound: 0.0288413
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0291952, upper bound: 0.0288410
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0256458, 0.0192978, -0.0085033, 0.0113201, -0.0369659, 0.0278011
1: -0.0136148, 0.0009542, -0.0089764, -0.0018765, -0.0117383, 0.0099306
2: -0.0000902, 0.0336888, -0.0008108, 0.0218941, -0.0219843, 0.0344996
3: -0.0199357, 0.0218661, -0.0092918, 0.0070203, -0.0269559, 0.0311579
4: -0.0175286, 0.0173037, -0.0099842, 0.0069385, -0.0244671, 0.0272879
5: 0.9749969, 1.0203305, 0.9915822, 1.0072097, -0.0322127, 0.0287484
6: -0.0198864, 0.0189944, -0.0068380, 0.0117699, -0.0316563, 0.0258324
7: -0.0326228, -0.0013761, -0.0256961, -0.0028798, -0.0297430, 0.0243201
8: -0.0164421, 0.0361899, -0.0072782, 0.0174125, -0.0338545, 0.0434681
9: -0.0136742, 0.0163852, -0.0071416, 0.0051970, -0.0188712, 0.0235268

Time for backsubstitution: 2.19 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.49 + 596.12 = 600.61 seconds
