## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.37902039


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.6700492, -5.9498515, -6.6700492, -5.9498515, -0.5647817, 0.5647817)
1: (-8.6853065, -7.6693430, -8.6853065, -7.6693430, -0.6316841, 0.6316841)
2: (-3.7678936, -2.9706786, -3.7678936, -2.9706786, -0.5475452, 0.5475452)
3: (-6.9010534, -6.0038757, -6.9010534, -6.0038757, -0.6367562, 0.6367562)
4: (-4.1678934, -3.3740907, -4.1678934, -3.3740907, -0.4944360, 0.4944361)
5: (-0.9807291, -0.3181703, -0.9807291, -0.3181703, -0.5873303, 0.5873306)
6: (4.7614808, 5.5848999, 4.7614808, 5.5848999, -0.6318514, 0.6318514)
7: (-11.8036003, -10.8616753, -11.8036003, -10.8616753, -0.7608390, 0.7608390)
8: (-2.3824112, -1.6447504, -2.3824112, -1.6447504, -0.5266042, 0.5266042)
9: (-10.5140667, -9.8322964, -10.5140667, -9.8322964, -0.5148268, 0.5148268)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.87 + 33.50 = 56.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3867555, upper bound: 0.3867563

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4625
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851079, upper bound: 0.3865299
time: 3.69 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055
time: 3.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.78 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.78
Output dim: 6, lower bound: -0.3851079, upper bound: 0.3865299
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.78
Output dim: 6, lower bound: -0.3851054, upper bound: 0.3851055

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.6700411, -5.9498501, -6.6700492, -5.9498515, -0.5647733, 0.5647802
1: -8.6852789, -7.6693482, -8.6853065, -7.6693430, -0.6316533, 0.6316841
2: -3.7678928, -2.9707100, -3.7678936, -2.9706786, -0.5475440, 0.5475113
3: -6.9010463, -6.0038810, -6.9010534, -6.0038757, -0.6367474, 0.6367540
4: -4.1678758, -3.3740950, -4.1678934, -3.3740907, -0.4944193, 0.4944288
5: -0.9806974, -0.3181758, -0.9807291, -0.3181703, -0.5872984, 0.5873268
6: 4.7614880, 5.5848722, 4.7614808, 5.5848999, -0.6318469, 0.6318233
7: -11.8035831, -10.8616810, -11.8036003, -10.8616753, -0.7608185, 0.7608337
8: -2.3824077, -1.6447618, -2.3824112, -1.6447504, -0.5265996, 0.5265936
9: -10.5140572, -9.8323002, -10.5140667, -9.8322964, -0.5148149, 0.5148249

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851053, upper bound: 0.3864259
time: 3.71 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851043, upper bound: 0.3865272
time: 3.64 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.6873136, -5.9477110, -6.6700492, -5.9498515, -0.5806017, 0.5687873
1: -8.6880016, -7.6380687, -8.6853065, -7.6693439, -0.6389153, 0.6432695
2: -3.7982342, -2.9657719, -3.7678943, -2.9706788, -0.5601814, 0.5531034
3: -6.9062996, -6.0001082, -6.9010530, -6.0038772, -0.6412835, 0.6422021
4: -4.1680479, -3.3459213, -4.1678925, -3.3740909, -0.4993269, 0.5014155
5: -0.9845951, -0.2748175, -0.9807285, -0.3181709, -0.5908008, 0.5980606
6: 4.7045741, 5.5853782, 4.7614818, 5.5848999, -0.6499512, 0.6359439
7: -11.8075333, -10.8478394, -11.8036003, -10.8616762, -0.7639313, 0.7727814
8: -2.4038563, -1.6438179, -2.3824115, -1.6447487, -0.5378487, 0.5314031
9: -10.5158720, -9.8226547, -10.5140648, -9.8322983, -0.5163069, 0.5206907

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851028, upper bound: 0.3849669
time: 3.58 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3851017
time: 3.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.61 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 6, lower bound: -0.3851053, upper bound: 0.3864259
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 6, lower bound: -0.3851043, upper bound: 0.3865272
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 6, lower bound: -0.3851028, upper bound: 0.3849669
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3851017

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.6700411, -5.9498501, -6.6697774, -5.9498520, -0.5647597, 0.5643787
1: -8.6852789, -7.6693482, -8.6849709, -7.6694283, -0.6316085, 0.6313417
2: -3.7678928, -2.9707100, -3.7678173, -2.9709396, -0.5472867, 0.5474477
3: -6.9010463, -6.0038810, -6.9009461, -6.0039577, -0.6365006, 0.6365972
4: -4.1678758, -3.3740950, -4.1678867, -3.3743448, -0.4940603, 0.4944109
5: -0.9806974, -0.3181758, -0.9803408, -0.3182213, -0.5872343, 0.5869517
6: 4.7614880, 5.5848722, 4.7617316, 5.5848894, -0.6317885, 0.6313543
7: -11.8035831, -10.8616810, -11.8033333, -10.8618259, -0.7606745, 0.7605696
8: -2.3824077, -1.6447618, -2.3813438, -1.6447730, -0.5265348, 0.5254421
9: -10.5140572, -9.8323002, -10.5140190, -9.8323660, -0.5146501, 0.5146730

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 4625

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3864266
time: 3.75 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3864259
time: 3.79 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.6700382, -5.9498510, -6.6822252, -5.9462790, -0.5697069, 0.5806048
1: -8.6852798, -7.6693468, -8.6873417, -7.6602798, -0.6405571, 0.6364589
2: -3.7678924, -2.9707098, -3.7747006, -2.9700031, -0.5485048, 0.5551033
3: -6.9010439, -6.0038815, -6.9115129, -6.0029163, -0.6423767, 0.6498351
4: -4.1678753, -3.3740945, -4.1735592, -3.3626404, -0.5106392, 0.5006773
5: -0.9806952, -0.3181756, -0.9831413, -0.3106172, -0.5930419, 0.5907328
6: 4.7614880, 5.5848727, 4.7549601, 5.5892587, -0.6365151, 0.6433296
7: -11.8035841, -10.8616829, -11.8059635, -10.8328972, -0.7836633, 0.7652645
8: -2.3824060, -1.6447632, -2.3875833, -1.6275840, -0.5370882, 0.5418503
9: -10.5140572, -9.8323002, -10.5148449, -9.8236427, -0.5233133, 0.5153546

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 4625

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3865273
time: 3.72 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3865272
time: 3.84 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.6873136, -5.9477110, -6.6697764, -5.9498520, -0.5805879, 0.5683796
1: -8.6880016, -7.6380687, -8.6849699, -7.6694283, -0.6388705, 0.6429210
2: -3.7982342, -2.9657719, -3.7678175, -2.9709401, -0.5599239, 0.5530403
3: -6.9062996, -6.0001082, -6.9009457, -6.0039577, -0.6410367, 0.6420455
4: -4.1680479, -3.3459213, -4.1678853, -3.3743451, -0.4989698, 0.5013978
5: -0.9845951, -0.2748175, -0.9803396, -0.3182213, -0.5907362, 0.5976841
6: 4.7045741, 5.5853782, 4.7617316, 5.5848894, -0.6498883, 0.6354759
7: -11.8075333, -10.8478394, -11.8033342, -10.8618259, -0.7637877, 0.7725134
8: -2.4038563, -1.6438179, -2.3813426, -1.6447728, -0.5377834, 0.5302521
9: -10.5158720, -9.8226547, -10.5140190, -9.8323669, -0.5161419, 0.5205388

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849675
time: 4.03 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849670
time: 3.98 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.6873140, -5.9477110, -6.6822243, -5.9462786, -0.5819246, 0.5810030
1: -8.6880016, -7.6380658, -8.6873417, -7.6602783, -0.6474148, 0.6479623
2: -3.7982333, -2.9657712, -3.7747014, -2.9700034, -0.5607963, 0.5606956
3: -6.9062982, -6.0001087, -6.9115129, -6.0029173, -0.6469128, 0.6552240
4: -4.1680479, -3.3459225, -4.1735587, -3.3626409, -0.5109861, 0.5031010
5: -0.9845952, -0.2748172, -0.9831415, -0.3106177, -0.5948660, 0.6009066
6: 4.7045755, 5.5853763, 4.7549605, 5.5892577, -0.6510568, 0.6438894
7: -11.8075333, -10.8478394, -11.8059626, -10.8328972, -0.7868137, 0.7768173
8: -2.4038546, -1.6438181, -2.3875840, -1.6275826, -0.5413775, 0.5434394
9: -10.5158720, -9.8226547, -10.5148449, -9.8236427, -0.5248094, 0.5213161

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 4625

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851015
time: 3.86 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851018
time: 3.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.26 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3864266
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3864259
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3865273
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3865272
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849675
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849670
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851015
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.26
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851018

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.6697659, -5.9498529, -6.6697774, -5.9498520, -0.5643587, 0.5643654
1: -8.6849394, -7.6694331, -8.6849709, -7.6694283, -0.6312659, 0.6312966
2: -3.7678154, -2.9709716, -3.7678173, -2.9709396, -0.5472238, 0.5471907
3: -6.9009385, -6.0039639, -6.9009461, -6.0039577, -0.6363442, 0.6363504
4: -4.1678691, -3.3743489, -4.1678867, -3.3743448, -0.4940424, 0.4940513
5: -0.9803085, -0.3182267, -0.9803408, -0.3182213, -0.5868599, 0.5868878
6: 4.7617369, 5.5848618, 4.7617316, 5.5848894, -0.6313198, 0.6312957
7: -11.8033180, -10.8618317, -11.8033333, -10.8618259, -0.7604103, 0.7604246
8: -2.3813396, -1.6447852, -2.3813438, -1.6447730, -0.5253839, 0.5253770
9: -10.5140104, -9.8323708, -10.5140190, -9.8323660, -0.5144987, 0.5145078

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864259
time: 5.15 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864266
time: 3.59 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.6822157, -5.9462790, -6.6697774, -5.9498520, -0.5779231, 0.5693047
1: -8.6873112, -7.6602831, -8.6849709, -7.6694283, -0.6363068, 0.6402454
2: -3.7746983, -2.9700336, -3.7678173, -2.9709396, -0.5548790, 0.5480642
3: -6.9115047, -6.0029202, -6.9009461, -6.0039577, -0.6495829, 0.6381757
4: -4.1735411, -3.3626435, -4.1678867, -3.3743448, -0.5003085, 0.5054475
5: -0.9831100, -0.3106222, -0.9803408, -0.3182213, -0.5900595, 0.5926948
6: 4.7549644, 5.5892305, 4.7617316, 5.5848894, -0.6392348, 0.6360216
7: -11.8059473, -10.8329048, -11.8033333, -10.8618259, -0.7646842, 0.7834120
8: -2.3875799, -1.6275947, -2.3813438, -1.6447730, -0.5347970, 0.5359190
9: -10.5148373, -9.8236456, -10.5140190, -9.8323660, -0.5151813, 0.5225945

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864266
time: 3.72 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864259
time: 3.90 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.6697659, -5.9498529, -6.6822252, -5.9462790, -0.5692995, 0.5779297
1: -8.6849394, -7.6694331, -8.6873417, -7.6602798, -0.6402152, 0.6363370
2: -3.7678154, -2.9709716, -3.7747006, -2.9700031, -0.5480976, 0.5548463
3: -6.9009385, -6.0039639, -6.9115129, -6.0029163, -0.6381695, 0.6495893
4: -4.1678691, -3.3743489, -4.1735592, -3.3626404, -0.5054388, 0.5003200
5: -0.9803085, -0.3182267, -0.9831413, -0.3106172, -0.5926654, 0.5900891
6: 4.7617369, 5.5848618, 4.7549601, 5.5892587, -0.6360469, 0.6392102
7: -11.8033180, -10.8618317, -11.8059635, -10.8328972, -0.7833953, 0.7646995
8: -2.3813396, -1.6447852, -2.3875833, -1.6275840, -0.5359275, 0.5347914
9: -10.5140104, -9.8323708, -10.5148449, -9.8236427, -0.5225859, 0.5151901

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 4625

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3865278
time: 3.76 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3865278
time: 3.84 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.6822157, -5.9462790, -6.6822252, -5.9462790, -0.5816150, 0.5823767
1: -8.6873112, -7.6602831, -8.6873417, -7.6602798, -0.6449499, 0.6449795
2: -3.7746983, -2.9700336, -3.7747006, -2.9700031, -0.5543976, 0.5543644
3: -6.9115047, -6.0029202, -6.9115129, -6.0029163, -0.6522558, 0.6522624
4: -4.1735411, -3.3626435, -4.1735592, -3.3626404, -0.5129070, 0.5120609
5: -0.9831100, -0.3106222, -0.9831413, -0.3106172, -0.5958674, 0.5964777
6: 4.7549644, 5.5892305, 4.7549601, 5.5892587, -0.6441734, 0.6451428
7: -11.8059473, -10.8329048, -11.8059635, -10.8328972, -0.7876997, 0.7881384
8: -2.3875799, -1.6275947, -2.3875833, -1.6275840, -0.5451355, 0.5465218
9: -10.5148373, -9.8236456, -10.5148449, -9.8236427, -0.5240057, 0.5236835

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3864266
time: 3.83 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3864260
time: 3.96 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.6870532, -5.9477119, -6.6697764, -5.9498520, -0.5801644, 0.5683661
1: -8.6876612, -7.6381488, -8.6849699, -7.6694283, -0.6385286, 0.6428576
2: -3.7981577, -2.9660299, -3.7678175, -2.9709401, -0.5598505, 0.5527828
3: -6.9061952, -6.0001822, -6.9009457, -6.0039577, -0.6408761, 0.6417987
4: -4.1680398, -3.3461754, -4.1678853, -3.3743451, -0.4989517, 0.5010370
5: -0.9842093, -0.2748606, -0.9803396, -0.3182213, -0.5903614, 0.5976083
6: 4.7048125, 5.5853667, 4.7617316, 5.5848894, -0.6493986, 0.6354129
7: -11.8072662, -10.8479900, -11.8033342, -10.8618259, -0.7635226, 0.7723627
8: -2.4027939, -1.6438398, -2.3813426, -1.6447728, -0.5366156, 0.5301867
9: -10.5158262, -9.8227215, -10.5140190, -9.8323669, -0.5159907, 0.5203624

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849675
time: 3.79 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849676
time: 3.92 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.6997380, -5.9441409, -6.6697764, -5.9498520, -0.5921183, 0.5697169
1: -8.6900120, -7.6289992, -8.6849699, -7.6694283, -0.6435854, 0.6466281
2: -3.8050213, -2.9651279, -3.7678175, -2.9709401, -0.5632222, 0.5536454
3: -6.9168110, -5.9991355, -6.9009457, -6.0039577, -0.6542153, 0.6436553
4: -4.1737132, -3.3345041, -4.1678853, -3.3743451, -0.5006654, 0.5125697
5: -0.9870024, -0.2672167, -0.9803396, -0.3182213, -0.5936105, 0.6001325
6: 4.6979833, 5.5897322, 4.7617316, 5.5848894, -0.6568675, 0.6366289
7: -11.8098850, -10.8191185, -11.8033342, -10.8618259, -0.7678485, 0.7907443
8: -2.4091039, -1.6266813, -2.3813426, -1.6447728, -0.5455823, 0.5374418
9: -10.5166206, -9.8140364, -10.5140190, -9.8323669, -0.5166960, 0.5260789

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849675
time: 3.91 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849669
time: 4.13 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.6870532, -5.9477119, -6.6822243, -5.9462786, -0.5815014, 0.5806816
1: -8.6876612, -7.6381488, -8.6873417, -7.6602783, -0.6467595, 0.6478906
2: -3.7981577, -2.9660299, -3.7747014, -2.9700034, -0.5607114, 0.5604386
3: -6.9061952, -6.0001822, -6.9115129, -6.0029173, -0.6427014, 0.6549361
4: -4.1680398, -3.3461754, -4.1735587, -3.3626409, -0.5106931, 0.5027404
5: -0.9842093, -0.2748606, -0.9831415, -0.3106177, -0.5942214, 0.6008101
6: 4.7048125, 5.5853667, 4.7549605, 5.5892577, -0.6505675, 0.6434975
7: -11.8072662, -10.8479900, -11.8059626, -10.8328972, -0.7865453, 0.7766671
8: -2.4027939, -1.6438398, -2.3875840, -1.6275826, -0.5402100, 0.5396011
9: -10.5158262, -9.8227215, -10.5148449, -9.8236427, -0.5240777, 0.5211396

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851022
time: 3.64 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851022
time: 3.83 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.6997380, -5.9441409, -6.6822243, -5.9462786, -0.5936950, 0.5835147
1: -8.6900120, -7.6289992, -8.6873417, -7.6602783, -0.6522279, 0.6517389
2: -3.8050213, -2.9651279, -3.7747014, -2.9700034, -0.5644269, 0.5599461
3: -6.9168110, -5.9991355, -6.9115129, -6.0029173, -0.6568885, 0.6567003
4: -4.1737132, -3.3345041, -4.1735587, -3.3626409, -0.5191379, 0.5145450
5: -0.9870024, -0.2672167, -0.9831415, -0.3106177, -0.5995164, 0.6039152
6: 4.6979833, 5.5897322, 4.7549605, 5.5892577, -0.6583941, 0.6508379
7: -11.8098850, -10.8191185, -11.8059626, -10.8328972, -0.7909079, 0.7959805
8: -2.4091039, -1.6266813, -2.3875840, -1.6275826, -0.5494311, 0.5512588
9: -10.5166206, -9.8140364, -10.5148449, -9.8236427, -0.5255580, 0.5282748

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 4625

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849675
time: 3.74 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849669
time: 3.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 36.41 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864259
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864266
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864266
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849708, upper bound: 0.3864259
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3865278
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3865278
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3864266
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849695, upper bound: 0.3864260
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849675
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849676
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849675
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849669
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851022
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851022
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849675
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.41
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849669

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.6697659, -5.9498529, -6.6697659, -5.9498529, -0.5643573, 0.5643573
1: -8.6849394, -7.6694331, -8.6849394, -7.6694331, -0.6312656, 0.6312656
2: -3.7678154, -2.9709716, -3.7678154, -2.9709716, -0.5471897, 0.5471897
3: -6.9009385, -6.0039639, -6.9009385, -6.0039639, -0.6363418, 0.6363418
4: -4.1678691, -3.3743489, -4.1678691, -3.3743489, -0.4940348, 0.4940346
5: -0.9803085, -0.3182267, -0.9803085, -0.3182267, -0.5868564, 0.5868561
6: 4.7617369, 5.5848618, 4.7617369, 5.5848618, -0.6312914, 0.6312914
7: -11.8033180, -10.8618317, -11.8033180, -10.8618317, -0.7604041, 0.7604041
8: -2.3813396, -1.6447852, -2.3813396, -1.6447852, -0.5253730, 0.5253730
9: -10.5140104, -9.8323708, -10.5140104, -9.8323708, -0.5144963, 0.5144963

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.6697659, -5.9498529, -6.6870532, -5.9477119, -0.5664051, 0.5801618
1: -8.6849394, -7.6694331, -8.6876612, -7.6381488, -0.6428264, 0.6340857
2: -3.7678154, -2.9709716, -3.7981577, -2.9660299, -0.5495780, 0.5598159
3: -6.9009385, -6.0039639, -6.9061952, -6.0001822, -0.6410108, 0.6408746
4: -4.1678691, -3.3743489, -4.1680398, -3.3461754, -0.5010176, 0.4943602
5: -0.9803085, -0.3182267, -0.9842093, -0.2748606, -0.5975757, 0.5883012
6: 4.7617369, 5.5848618, 4.7048125, 5.5853667, -0.6317725, 0.6493702
7: -11.8033180, -10.8618317, -11.8072662, -10.8479900, -0.7723317, 0.7635169
8: -2.3813396, -1.6447852, -2.4027939, -1.6438398, -0.5269248, 0.5366025
9: -10.5140104, -9.8323708, -10.5158262, -9.8227215, -0.5203371, 0.5159883

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 119

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.6822157, -5.9462790, -6.6697659, -5.9498529, -0.5779216, 0.5692968
1: -8.6873112, -7.6602831, -8.6849394, -7.6694331, -0.6363063, 0.6402142
2: -3.7746983, -2.9700336, -3.7678154, -2.9709716, -0.5548451, 0.5480635
3: -6.9115047, -6.0029202, -6.9009385, -6.0039639, -0.6495805, 0.6381671
4: -4.1735411, -3.3626435, -4.1678691, -3.3743489, -0.5003009, 0.5054308
5: -0.9831100, -0.3106222, -0.9803085, -0.3182267, -0.5900559, 0.5926619
6: 4.7549644, 5.5892305, 4.7617369, 5.5848618, -0.6392064, 0.6360178
7: -11.8059473, -10.8329048, -11.8033180, -10.8618317, -0.7646785, 0.7833900
8: -2.3875799, -1.6275947, -2.3813396, -1.6447852, -0.5347862, 0.5359142
9: -10.5148373, -9.8236456, -10.5140104, -9.8323708, -0.5151789, 0.5225830

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851032, upper bound: 0.3862511
time: 4.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 119

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.6822157, -5.9462790, -6.6870532, -5.9477119, -0.5799694, 0.5814986
1: -8.6873112, -7.6602831, -8.6876612, -7.6381488, -0.6478595, 0.6430345
2: -3.7746983, -2.9700336, -3.7981577, -2.9660299, -0.5572333, 0.5606767
3: -6.9115047, -6.0029202, -6.9061952, -6.0001822, -0.6542494, 0.6427000
4: -4.1735411, -3.3626435, -4.1680398, -3.3461754, -0.5027213, 0.5057564
5: -0.9831100, -0.3106222, -0.9842093, -0.2748606, -0.6007779, 0.5942175
6: 4.7549644, 5.5892305, 4.7048125, 5.5853667, -0.6396875, 0.6505384
7: -11.8059473, -10.8329048, -11.8072662, -10.8479900, -0.7766361, 0.7865396
8: -2.3875799, -1.6275947, -2.4027939, -1.6438398, -0.5363381, 0.5401968
9: -10.5148373, -9.8236456, -10.5158262, -9.8227215, -0.5210109, 0.5240750

Time for backsubstitution: 22.38 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.37 + 555.25 = 611.62 seconds
