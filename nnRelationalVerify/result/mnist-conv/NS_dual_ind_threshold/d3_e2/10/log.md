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
execution time: IAR + RelationalAnalysis = 24.21 + 34.28 = 58.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3867555, upper bound: 0.3867563

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 119

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 524

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3866593, upper bound: 0.3867545
time: 3.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3867530, upper bound: 0.3867530
time: 3.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.50 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.50
Output dim: 6, lower bound: -0.3866593, upper bound: 0.3867545
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.50
Output dim: 6, lower bound: -0.3867530, upper bound: 0.3867530

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.6697774, -5.9498520, -6.6700492, -5.9498515, -0.5643804, 0.5647683
1: -8.6849709, -7.6694283, -8.6853065, -7.6693430, -0.6313417, 0.6316395
2: -3.7678173, -2.9709396, -3.7678936, -2.9706786, -0.5474818, 0.5472879
3: -6.9009461, -6.0039577, -6.9010534, -6.0038757, -0.6365995, 0.6365094
4: -4.1678867, -3.3743448, -4.1678934, -3.3740907, -0.4944179, 0.4940771
5: -0.9803408, -0.3182213, -0.9807291, -0.3181703, -0.5869555, 0.5872662
6: 4.7617316, 5.5848894, 4.7614808, 5.5848999, -0.6313822, 0.6317930
7: -11.8033333, -10.8618259, -11.8036003, -10.8616753, -0.7605743, 0.7606955
8: -2.3813438, -1.6447730, -2.3824112, -1.6447504, -0.5254529, 0.5265393
9: -10.5140190, -9.8323660, -10.5140667, -9.8322964, -0.5146751, 0.5146620

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3866593, upper bound: 0.3866600
time: 3.94 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3866593, upper bound: 0.3867535
time: 3.71 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.6822252, -5.9462790, -6.6700478, -5.9498510, -0.5806074, 0.5697150
1: -8.6873417, -7.6602798, -8.6853065, -7.6693444, -0.6364594, 0.6405876
2: -3.7747006, -2.9700031, -3.7678938, -2.9706795, -0.5551374, 0.5485058
3: -6.9115129, -6.0029163, -6.9010525, -6.0038753, -0.6498375, 0.6423852
4: -4.1735592, -3.3626404, -4.1678929, -3.3740911, -0.5006850, 0.5106580
5: -0.9831413, -0.3106172, -0.9807277, -0.3181703, -0.5907364, 0.5930748
6: 4.7549601, 5.5892587, 4.7614818, 5.5848999, -0.6433582, 0.6365190
7: -11.8059635, -10.8328972, -11.8036003, -10.8616753, -0.7652688, 0.7836857
8: -2.3875833, -1.6275840, -2.3824086, -1.6447492, -0.5418637, 0.5370923
9: -10.5148449, -9.8236427, -10.5140657, -9.8322973, -0.5153575, 0.5233250

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 524

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3867531, upper bound: 0.3866600
time: 3.51 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3867531, upper bound: 0.3867536
time: 3.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.31 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.31
Output dim: 6, lower bound: -0.3866593, upper bound: 0.3866600
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.31
Output dim: 6, lower bound: -0.3866593, upper bound: 0.3867535
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.31
Output dim: 6, lower bound: -0.3867531, upper bound: 0.3866600
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.31
Output dim: 6, lower bound: -0.3867531, upper bound: 0.3867536

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.6697774, -5.9498520, -6.6697774, -5.9498520, -0.5643668, 0.5643668
1: -8.6849709, -7.6694283, -8.6849709, -7.6694283, -0.6312971, 0.6312969
2: -3.7678173, -2.9709396, -3.7678173, -2.9709396, -0.5472245, 0.5472245
3: -6.9009461, -6.0039577, -6.9009461, -6.0039577, -0.6363525, 0.6363528
4: -4.1678867, -3.3743448, -4.1678867, -3.3743448, -0.4940591, 0.4940590
5: -0.9803408, -0.3182213, -0.9803408, -0.3182213, -0.5868914, 0.5868914
6: 4.7617316, 5.5848894, 4.7617316, 5.5848894, -0.6313241, 0.6313241
7: -11.8033333, -10.8618259, -11.8033333, -10.8618259, -0.7604308, 0.7604308
8: -2.3813438, -1.6447730, -2.3813438, -1.6447730, -0.5253880, 0.5253880
9: -10.5140190, -9.8323660, -10.5140190, -9.8323660, -0.5145102, 0.5145104

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3864280
time: 3.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849689
time: 3.94 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.6697774, -5.9498520, -6.6822252, -5.9462790, -0.5693073, 0.5779309
1: -8.6849709, -7.6694283, -8.6873417, -7.6602798, -0.6402464, 0.6363375
2: -3.7678173, -2.9709396, -3.7747006, -2.9700031, -0.5480983, 0.5548804
3: -6.9009461, -6.0039577, -6.9115129, -6.0029163, -0.6381779, 0.6495917
4: -4.1678867, -3.3743448, -4.1735592, -3.3626404, -0.5054555, 0.5003277
5: -0.9803408, -0.3182213, -0.9831413, -0.3106172, -0.5926983, 0.5900927
6: 4.7617316, 5.5848894, 4.7549601, 5.5892587, -0.6360505, 0.6392386
7: -11.8033333, -10.8618259, -11.8059635, -10.8328972, -0.7834177, 0.7647052
8: -2.3813438, -1.6447730, -2.3875833, -1.6275840, -0.5359323, 0.5348022
9: -10.5140190, -9.8323660, -10.5148449, -9.8236427, -0.5225973, 0.5151925

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3865282
time: 3.76 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851025
time: 4.13 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.6822252, -5.9462790, -6.6697774, -5.9498520, -0.5779312, 0.5693076
1: -8.6873417, -7.6602798, -8.6849709, -7.6694283, -0.6363378, 0.6402462
2: -3.7747006, -2.9700031, -3.7678173, -2.9709396, -0.5548804, 0.5480983
3: -6.9115129, -6.0029163, -6.9009461, -6.0039577, -0.6495914, 0.6381779
4: -4.1735592, -3.3626404, -4.1678867, -3.3743448, -0.5003278, 0.5054556
5: -0.9831413, -0.3106172, -0.9803408, -0.3182213, -0.5900929, 0.5926983
6: 4.7549601, 5.5892587, 4.7617316, 5.5848894, -0.6392384, 0.6360505
7: -11.8059635, -10.8328972, -11.8033333, -10.8618259, -0.7647052, 0.7834177
8: -2.3875833, -1.6275840, -2.3813438, -1.6447730, -0.5348022, 0.5359323
9: -10.5148449, -9.8236427, -10.5140190, -9.8323660, -0.5151925, 0.5225973

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 119

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851042, upper bound: 0.3864258
time: 3.86 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849668
time: 3.86 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.6822252, -5.9462790, -6.6822252, -5.9462790, -0.5816228, 0.5823793
1: -8.6873417, -7.6602798, -8.6873417, -7.6602798, -0.6449804, 0.6449804
2: -3.7747006, -2.9700031, -3.7747006, -2.9700031, -0.5543988, 0.5543988
3: -6.9115129, -6.0029163, -6.9115129, -6.0029163, -0.6522646, 0.6522646
4: -4.1735592, -3.3626404, -4.1735592, -3.3626404, -0.5129261, 0.5120687
5: -0.9831413, -0.3106172, -0.9831413, -0.3106172, -0.5959003, 0.5964811
6: 4.7549601, 5.5892587, 4.7549601, 5.5892587, -0.6441779, 0.6451714
7: -11.8059635, -10.8328972, -11.8059635, -10.8328972, -0.7877226, 0.7881436
8: -2.3875833, -1.6275840, -2.3875833, -1.6275840, -0.5451415, 0.5465329
9: -10.5148449, -9.8236427, -10.5148449, -9.8236427, -0.5240178, 0.5236905

Time for backsubstitution: 22.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4625
type: A, layer: 1, pos: 119

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 4625

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851043, upper bound: 0.3864260
time: 4.12 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849667
time: 4.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.29 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3864280
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3849689
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3849694, upper bound: 0.3865282
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851025
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3851042, upper bound: 0.3864258
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849668
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3851043, upper bound: 0.3864260
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.29
Output dim: 6, lower bound: -0.3851017, upper bound: 0.3849667

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

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849684, upper bound: 0.3849689
time: 4.02 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849684, upper bound: 0.3849689
time: 4.06 seconds

## BFS NS instance: NS_A1_B1_A2

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

Time for backsubstitution: 22.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849689
time: 3.92 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849689
time: 3.96 seconds

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

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851033
time: 4.16 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851031
time: 3.91 seconds

## BFS NS instance: NS_A1_B2_A2

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

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851033
time: 4.41 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851033
time: 4.33 seconds

## BFS NS instance: NS_A2_B1_A1

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

Time for backsubstitution: 22.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849675
time: 3.95 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849668
time: 4.17 seconds

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

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849675
time: 3.88 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849675
time: 3.83 seconds

## BFS NS instance: NS_A2_B2_A1

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

Time for backsubstitution: 22.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849675
time: 3.92 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849667
time: 4.03 seconds

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

Time for backsubstitution: 22.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4625
type: B, layer: 1, pos: 119

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 4625

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849675
time: 4.00 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849675
time: 3.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.96 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849684, upper bound: 0.3849689
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849684, upper bound: 0.3849689
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849689
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849683, upper bound: 0.3849689
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851033
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851031
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851033
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3849669, upper bound: 0.3851033
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849675
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849668
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849675
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851027, upper bound: 0.3849675
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849675
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849667
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849675
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.96
Output dim: 6, lower bound: -0.3851016, upper bound: 0.3849675

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

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

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

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.6870532, -5.9477119, -6.6697659, -5.9498529, -0.5801617, 0.5664051
1: -8.6876612, -7.6381488, -8.6849394, -7.6694331, -0.6340857, 0.6428263
2: -3.7981577, -2.9660299, -3.7678154, -2.9709716, -0.5598159, 0.5495780
3: -6.9061952, -6.0001822, -6.9009385, -6.0039639, -0.6408746, 0.6410108
4: -4.1680398, -3.3461754, -4.1678691, -3.3743489, -0.4943602, 0.5010177
5: -0.9842093, -0.2748606, -0.9803085, -0.3182267, -0.5883012, 0.5975757
6: 4.7048125, 5.5853667, 4.7617369, 5.5848618, -0.6493702, 0.6317725
7: -11.8072662, -10.8479900, -11.8033180, -10.8618317, -0.7635169, 0.7723317
8: -2.4027939, -1.6438398, -2.3813396, -1.6447852, -0.5366025, 0.5269248
9: -10.5158262, -9.8227215, -10.5140104, -9.8323708, -0.5159883, 0.5203371

Time for backsubstitution: 22.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 119

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 119

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.6870532, -5.9477119, -6.6870532, -5.9477119, -0.5812860, 0.5805569
1: -8.6876612, -7.6381488, -8.6876612, -7.6381488, -0.6458195, 0.6502571
2: -3.7981577, -2.9660299, -3.7981577, -2.9660299, -0.5654387, 0.5622334
3: -6.9061952, -6.0001822, -6.9061952, -6.0001822, -0.6463220, 0.6463220
4: -4.1680398, -3.3461754, -4.1680398, -3.3461754, -0.5013652, 0.5056051
5: -0.9842093, -0.2748606, -0.9842093, -0.2748606, -0.5991311, 0.6011789
6: 4.7048125, 5.5853667, 4.7048125, 5.5853667, -0.6555450, 0.6499293
7: -11.8072662, -10.8479900, -11.8072662, -10.8479900, -0.7760005, 0.7754841
8: -2.4027939, -1.6438398, -2.4027939, -1.6438398, -0.5359759, 0.5359759
9: -10.5158262, -9.8227215, -10.5158262, -9.8227215, -0.5232658, 0.5218379

Time for backsubstitution: 22.38 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.49 + 543.90 = 602.39 seconds
