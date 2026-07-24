## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0009916


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9949470, 0.9984001, 0.9949470, 0.9984001, -0.0022326, 0.0022326)
1: (-0.0025230, -0.0016626, -0.0025230, -0.0016626, -0.0005563, 0.0005563)
2: (-0.0012431, 0.0033167, -0.0012431, 0.0033167, -0.0029481, 0.0029481)
3: (-0.0027827, -0.0007073, -0.0027827, -0.0007073, -0.0013418, 0.0013418)
4: (0.0002873, 0.0011698, 0.0002873, 0.0011698, -0.0005706, 0.0005706)
5: (-0.0026040, 0.0031309, -0.0026040, 0.0031309, -0.0037079, 0.0037079)
6: (0.0007462, 0.0022018, 0.0007462, 0.0022018, -0.0009411, 0.0009411)
7: (-0.0012071, 0.0025590, -0.0012071, 0.0025590, -0.0024349, 0.0024349)
8: (-0.0001989, 0.0017816, -0.0001989, 0.0017816, -0.0012805, 0.0012805)
9: (-0.0039297, -0.0016332, -0.0039297, -0.0016332, -0.0014848, 0.0014848)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.75 + 2.07 = 3.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0012395, upper bound: 0.0012395

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011600, upper bound: 0.0011042
time: 1.18 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011661, upper bound: 0.0011661
time: 1.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.57 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 0, lower bound: -0.0011600, upper bound: 0.0011042
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.57
Output dim: 0, lower bound: -0.0011661, upper bound: 0.0011661

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.9948618, 0.9980378, 0.9949595, 0.9982629, -0.0019201, 0.0016741
1: -0.0025442, -0.0017529, -0.0025199, -0.0016968, -0.0004784, 0.0004171
2: -0.0007646, 0.0034291, -0.0010619, 0.0033002, -0.0022106, 0.0025354
3: -0.0028339, -0.0009251, -0.0027753, -0.0007898, -0.0011540, 0.0010062
4: 0.0003799, 0.0011916, 0.0003224, 0.0011666, -0.0004279, 0.0004907
5: -0.0020022, 0.0032724, -0.0023762, 0.0031103, -0.0027803, 0.0031889
6: 0.0007103, 0.0020490, 0.0007514, 0.0021439, -0.0008094, 0.0007057
7: -0.0013000, 0.0021638, -0.0011935, 0.0024094, -0.0020941, 0.0018258
8: -0.0002478, 0.0015738, -0.0001918, 0.0017029, -0.0011013, 0.0009602
9: -0.0036887, -0.0015765, -0.0038385, -0.0016415, -0.0011134, 0.0012770

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011041
time: 1.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011042
time: 1.19 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.9949558, 0.9983113, 0.9949479, 0.9983909, -0.0022221, 0.0018901
1: -0.0025208, -0.0016847, -0.0025228, -0.0016649, -0.0005537, 0.0004710
2: -0.0011258, 0.0033051, -0.0012308, 0.0033156, -0.0024958, 0.0029342
3: -0.0027774, -0.0007607, -0.0027822, -0.0007129, -0.0013355, 0.0011360
4: 0.0003100, 0.0011676, 0.0002897, 0.0011696, -0.0004831, 0.0005679
5: -0.0024565, 0.0031164, -0.0025886, 0.0031296, -0.0031391, 0.0036905
6: 0.0007499, 0.0021643, 0.0007465, 0.0021979, -0.0009367, 0.0007967
7: -0.0011975, 0.0024621, -0.0012062, 0.0025489, -0.0024235, 0.0020614
8: -0.0001939, 0.0017306, -0.0001985, 0.0017763, -0.0012745, 0.0010841
9: -0.0038706, -0.0016390, -0.0039235, -0.0016337, -0.0012570, 0.0014778

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011600
time: 1.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011661
time: 1.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.23 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011041
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011042
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011600
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.23
Output dim: 0, lower bound: -0.0011041, upper bound: 0.0011661

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.9948618, 0.9980378, 0.9948618, 0.9980378, -0.0016653, 0.0016653
1: -0.0025442, -0.0017529, -0.0025442, -0.0017529, -0.0004150, 0.0004150
2: -0.0007646, 0.0034291, -0.0007646, 0.0034291, -0.0021991, 0.0021991
3: -0.0028339, -0.0009251, -0.0028339, -0.0009251, -0.0010009, 0.0010009
4: 0.0003799, 0.0011916, 0.0003799, 0.0011916, -0.0004256, 0.0004256
5: -0.0020022, 0.0032724, -0.0020022, 0.0032724, -0.0027658, 0.0027658
6: 0.0007103, 0.0020490, 0.0007103, 0.0020490, -0.0007020, 0.0007020
7: -0.0013000, 0.0021638, -0.0013000, 0.0021638, -0.0018163, 0.0018163
8: -0.0002478, 0.0015738, -0.0002478, 0.0015738, -0.0009552, 0.0009552
9: -0.0036887, -0.0015765, -0.0036887, -0.0015765, -0.0011076, 0.0011076

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010708, upper bound: 0.0010666
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010710, upper bound: 0.0010706
time: 1.24 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.9948618, 0.9980378, 0.9949564, 0.9983113, -0.0020712, 0.0016754
1: -0.0025442, -0.0017529, -0.0025207, -0.0016847, -0.0005161, 0.0004175
2: -0.0007646, 0.0034291, -0.0011258, 0.0033043, -0.0022123, 0.0027349
3: -0.0028339, -0.0009251, -0.0027771, -0.0007607, -0.0012448, 0.0010069
4: 0.0003799, 0.0011916, 0.0003100, 0.0011674, -0.0004282, 0.0005293
5: -0.0020022, 0.0032724, -0.0024565, 0.0031154, -0.0027825, 0.0034398
6: 0.0007103, 0.0020490, 0.0007501, 0.0021643, -0.0008731, 0.0007062
7: -0.0013000, 0.0021638, -0.0011969, 0.0024621, -0.0022589, 0.0018272
8: -0.0002478, 0.0015738, -0.0001936, 0.0017306, -0.0011879, 0.0009609
9: -0.0036887, -0.0015765, -0.0038706, -0.0016394, -0.0011142, 0.0013775

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010708, upper bound: 0.0010666
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010710, upper bound: 0.0010706
time: 1.18 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.9949558, 0.9983113, 0.9948618, 0.9980378, -0.0018103, 0.0020712
1: -0.0025208, -0.0016847, -0.0025442, -0.0017529, -0.0004511, 0.0005161
2: -0.0011258, 0.0033051, -0.0007646, 0.0034291, -0.0027349, 0.0023905
3: -0.0027774, -0.0007607, -0.0028339, -0.0009251, -0.0010881, 0.0012448
4: 0.0003100, 0.0011676, 0.0003799, 0.0011916, -0.0005293, 0.0004627
5: -0.0024565, 0.0031164, -0.0020022, 0.0032724, -0.0034398, 0.0030066
6: 0.0007499, 0.0021643, 0.0007103, 0.0020490, -0.0007631, 0.0008731
7: -0.0011975, 0.0024621, -0.0013000, 0.0021638, -0.0019744, 0.0022589
8: -0.0001939, 0.0017306, -0.0002478, 0.0015738, -0.0010383, 0.0011879
9: -0.0038706, -0.0016390, -0.0036887, -0.0015765, -0.0013775, 0.0012040

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010666, upper bound: 0.0011246
time: 1.24 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010706, upper bound: 0.0011252
time: 1.21 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.9949558, 0.9983113, 0.9949558, 0.9983113, -0.0018848, 0.0018848
1: -0.0025208, -0.0016847, -0.0025208, -0.0016847, -0.0004696, 0.0004696
2: -0.0011258, 0.0033051, -0.0011258, 0.0033051, -0.0024889, 0.0024889
3: -0.0027774, -0.0007607, -0.0027774, -0.0007607, -0.0011328, 0.0011328
4: 0.0003100, 0.0011676, 0.0003100, 0.0011676, -0.0004817, 0.0004817
5: -0.0024565, 0.0031164, -0.0024565, 0.0031164, -0.0031303, 0.0031303
6: 0.0007499, 0.0021643, 0.0007499, 0.0021643, -0.0007945, 0.0007945
7: -0.0011975, 0.0024621, -0.0011975, 0.0024621, -0.0020556, 0.0020556
8: -0.0001939, 0.0017306, -0.0001939, 0.0017306, -0.0010810, 0.0010810
9: -0.0038706, -0.0016390, -0.0038706, -0.0016390, -0.0012535, 0.0012535

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 223
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010705, upper bound: 0.0011288
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010706, upper bound: 0.0011298
time: 1.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.27 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010708, upper bound: 0.0010666
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010710, upper bound: 0.0010706
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010708, upper bound: 0.0010666
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010710, upper bound: 0.0010706
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010666, upper bound: 0.0011246
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010706, upper bound: 0.0011252
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010705, upper bound: 0.0011288
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 0, lower bound: -0.0010706, upper bound: 0.0011298

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9948846, 0.9979826, 0.9948673, 0.9980245, -0.0016061, 0.0016078
1: -0.0025386, -0.0017667, -0.0025429, -0.0017562, -0.0004002, 0.0004006
2: -0.0006916, 0.0033992, -0.0007471, 0.0034220, -0.0021230, 0.0021209
3: -0.0028203, -0.0009583, -0.0028307, -0.0009331, -0.0009653, 0.0009663
4: 0.0003940, 0.0011858, 0.0003833, 0.0011902, -0.0004109, 0.0004105
5: -0.0019105, 0.0032348, -0.0019802, 0.0032635, -0.0026702, 0.0026675
6: 0.0007198, 0.0020257, 0.0007125, 0.0020434, -0.0006770, 0.0006777
7: -0.0012752, 0.0021035, -0.0012941, 0.0021494, -0.0017517, 0.0017535
8: -0.0002348, 0.0015421, -0.0002447, 0.0015662, -0.0009212, 0.0009221
9: -0.0036520, -0.0015916, -0.0036799, -0.0015801, -0.0010693, 0.0010682

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010670
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010670
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9948485, 0.9979604, 0.9948714, 0.9980088, -0.0016098, 0.0016072
1: -0.0025476, -0.0017722, -0.0025419, -0.0017601, -0.0004011, 0.0004005
2: -0.0006624, 0.0034467, -0.0007264, 0.0034166, -0.0021223, 0.0021258
3: -0.0028419, -0.0009716, -0.0028282, -0.0009425, -0.0009676, 0.0009660
4: 0.0003997, 0.0011950, 0.0003873, 0.0011892, -0.0004108, 0.0004114
5: -0.0018736, 0.0032945, -0.0019541, 0.0032566, -0.0026693, 0.0026736
6: 0.0007046, 0.0020164, 0.0007143, 0.0020368, -0.0006786, 0.0006775
7: -0.0013145, 0.0020794, -0.0012896, 0.0021322, -0.0017557, 0.0017529
8: -0.0002554, 0.0015294, -0.0002423, 0.0015572, -0.0009233, 0.0009218
9: -0.0036372, -0.0015677, -0.0036695, -0.0015829, -0.0010689, 0.0010706

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010709
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010710
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9948846, 0.9979826, 0.9949619, 0.9982989, -0.0020107, 0.0016184
1: -0.0025386, -0.0017667, -0.0025193, -0.0016878, -0.0005010, 0.0004033
2: -0.0006916, 0.0033992, -0.0011094, 0.0032972, -0.0021371, 0.0026551
3: -0.0028203, -0.0009583, -0.0027739, -0.0007682, -0.0012085, 0.0009727
4: 0.0003940, 0.0011858, 0.0003132, 0.0011661, -0.0004136, 0.0005139
5: -0.0019105, 0.0032348, -0.0024359, 0.0031064, -0.0026879, 0.0033394
6: 0.0007198, 0.0020257, 0.0007524, 0.0021591, -0.0008476, 0.0006822
7: -0.0012752, 0.0021035, -0.0011910, 0.0024486, -0.0021929, 0.0017651
8: -0.0002348, 0.0015421, -0.0001905, 0.0017235, -0.0011532, 0.0009283
9: -0.0036520, -0.0015916, -0.0038624, -0.0016430, -0.0010764, 0.0013372

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010666
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010666
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9948485, 0.9979604, 0.9949660, 0.9982802, -0.0020163, 0.0016168
1: -0.0025476, -0.0017722, -0.0025183, -0.0016925, -0.0005024, 0.0004029
2: -0.0006624, 0.0034467, -0.0010847, 0.0032918, -0.0021349, 0.0026625
3: -0.0028419, -0.0009716, -0.0027714, -0.0007794, -0.0012119, 0.0009717
4: 0.0003997, 0.0011950, 0.0003180, 0.0011650, -0.0004132, 0.0005153
5: -0.0018736, 0.0032945, -0.0024048, 0.0030996, -0.0026852, 0.0033487
6: 0.0007046, 0.0020164, 0.0007541, 0.0021512, -0.0008499, 0.0006815
7: -0.0013145, 0.0020794, -0.0011865, 0.0024282, -0.0021991, 0.0017633
8: -0.0002554, 0.0015294, -0.0001881, 0.0017128, -0.0011565, 0.0009273
9: -0.0036372, -0.0015677, -0.0038499, -0.0016457, -0.0010753, 0.0013410

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010705
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010706
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.9949614, 0.9982989, 0.9948846, 0.9979826, -0.0017542, 0.0020107
1: -0.0025195, -0.0016878, -0.0025386, -0.0017667, -0.0004371, 0.0005010
2: -0.0011094, 0.0032978, -0.0006916, 0.0033992, -0.0026551, 0.0023164
3: -0.0027741, -0.0007682, -0.0028203, -0.0009583, -0.0010543, 0.0012085
4: 0.0003132, 0.0011662, 0.0003940, 0.0011858, -0.0005139, 0.0004483
5: -0.0024359, 0.0031073, -0.0019105, 0.0032348, -0.0033394, 0.0029135
6: 0.0007522, 0.0021591, 0.0007198, 0.0020257, -0.0007395, 0.0008476
7: -0.0011915, 0.0024486, -0.0012752, 0.0021035, -0.0019132, 0.0021929
8: -0.0001907, 0.0017235, -0.0002348, 0.0015421, -0.0010061, 0.0011532
9: -0.0038624, -0.0016427, -0.0036520, -0.0015916, -0.0013372, 0.0011667

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010666, upper bound: 0.0011239
time: 1.33 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010666, upper bound: 0.0011246
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9949656, 0.9982802, 0.9948485, 0.9979604, -0.0017527, 0.0020163
1: -0.0025184, -0.0016925, -0.0025476, -0.0017722, -0.0004367, 0.0005024
2: -0.0010847, 0.0032922, -0.0006624, 0.0034467, -0.0026625, 0.0023144
3: -0.0027716, -0.0007794, -0.0028419, -0.0009716, -0.0010534, 0.0012119
4: 0.0003180, 0.0011651, 0.0003997, 0.0011950, -0.0005153, 0.0004480
5: -0.0024048, 0.0031002, -0.0018736, 0.0032945, -0.0033487, 0.0029109
6: 0.0007540, 0.0021512, 0.0007046, 0.0020164, -0.0007388, 0.0008499
7: -0.0011869, 0.0024282, -0.0013145, 0.0020794, -0.0019116, 0.0021991
8: -0.0001883, 0.0017128, -0.0002554, 0.0015294, -0.0010053, 0.0011565
9: -0.0038499, -0.0016455, -0.0036372, -0.0015677, -0.0013410, 0.0011657

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 223

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010705, upper bound: 0.0011239
time: 1.18 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010705, upper bound: 0.0011252
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9949788, 0.9982597, 0.9949614, 0.9982989, -0.0018287, 0.0018294
1: -0.0025151, -0.0016976, -0.0025195, -0.0016878, -0.0004557, 0.0004558
2: -0.0010577, 0.0032747, -0.0011094, 0.0032978, -0.0024156, 0.0024148
3: -0.0027636, -0.0007917, -0.0027741, -0.0007682, -0.0010991, 0.0010995
4: 0.0003232, 0.0011617, 0.0003132, 0.0011662, -0.0004675, 0.0004674
5: -0.0023708, 0.0030781, -0.0024359, 0.0031073, -0.0030382, 0.0030372
6: 0.0007596, 0.0021426, 0.0007522, 0.0021591, -0.0007709, 0.0007711
7: -0.0011724, 0.0024059, -0.0011915, 0.0024486, -0.0019945, 0.0019952
8: -0.0001807, 0.0017011, -0.0001907, 0.0017235, -0.0010489, 0.0010492
9: -0.0038363, -0.0016543, -0.0038624, -0.0016427, -0.0012166, 0.0012162

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011283
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011283
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9949513, 0.9982269, 0.9949656, 0.9982802, -0.0016681, 0.0018257
1: -0.0025220, -0.0017058, -0.0025184, -0.0016925, -0.0004156, 0.0004549
2: -0.0010144, 0.0033111, -0.0010847, 0.0032922, -0.0024109, 0.0022027
3: -0.0027802, -0.0008114, -0.0027716, -0.0007794, -0.0010026, 0.0010973
4: 0.0003316, 0.0011687, 0.0003180, 0.0011651, -0.0004666, 0.0004263
5: -0.0023164, 0.0031240, -0.0024048, 0.0031002, -0.0030322, 0.0027704
6: 0.0007479, 0.0021288, 0.0007540, 0.0021512, -0.0007032, 0.0007696
7: -0.0012025, 0.0023701, -0.0011869, 0.0024282, -0.0018193, 0.0019912
8: -0.0001965, 0.0016823, -0.0001883, 0.0017128, -0.0009567, 0.0010472
9: -0.0038145, -0.0016360, -0.0038499, -0.0016455, -0.0012142, 0.0011094

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 223
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 223

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011291
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011298
time: 1.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.23 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010670
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010670
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010709
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010670, upper bound: 0.0010710
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010666
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010666
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010705
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0011239, upper bound: 0.0010706
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010666, upper bound: 0.0011239
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010666, upper bound: 0.0011246
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010705, upper bound: 0.0011239
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010705, upper bound: 0.0011252
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011283
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011283
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011291
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.23
Output dim: 0, lower bound: -0.0010746, upper bound: 0.0011298

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9948846, 0.9979826, 0.9948846, 0.9979826, -0.0015722, 0.0015722
1: -0.0025386, -0.0017667, -0.0025386, -0.0017667, -0.0003917, 0.0003917
2: -0.0006916, 0.0033992, -0.0006916, 0.0033992, -0.0020760, 0.0020760
3: -0.0028203, -0.0009583, -0.0028203, -0.0009583, -0.0009449, 0.0009449
4: 0.0003940, 0.0011858, 0.0003940, 0.0011858, -0.0004018, 0.0004018
5: -0.0019105, 0.0032348, -0.0019105, 0.0032348, -0.0026111, 0.0026111
6: 0.0007198, 0.0020257, 0.0007198, 0.0020257, -0.0006627, 0.0006627
7: -0.0012752, 0.0021035, -0.0012752, 0.0021035, -0.0017147, 0.0017147
8: -0.0002348, 0.0015421, -0.0002348, 0.0015421, -0.0009017, 0.0009017
9: -0.0036520, -0.0015916, -0.0036520, -0.0015916, -0.0010456, 0.0010456

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006143, upper bound: 0.0007607
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010608, upper bound: 0.0010603
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9948846, 0.9979826, 0.9948485, 0.9979604, -0.0015447, 0.0015940
1: -0.0025386, -0.0017667, -0.0025476, -0.0017722, -0.0003849, 0.0003972
2: -0.0006916, 0.0033992, -0.0006624, 0.0034467, -0.0021048, 0.0020398
3: -0.0028203, -0.0009583, -0.0028419, -0.0009716, -0.0009284, 0.0009580
4: 0.0003940, 0.0011858, 0.0003997, 0.0011950, -0.0004074, 0.0003948
5: -0.0019105, 0.0032348, -0.0018736, 0.0032945, -0.0026473, 0.0025655
6: 0.0007198, 0.0020257, 0.0007046, 0.0020164, -0.0006512, 0.0006719
7: -0.0012752, 0.0021035, -0.0013145, 0.0020794, -0.0016847, 0.0017384
8: -0.0002348, 0.0015421, -0.0002554, 0.0015294, -0.0008860, 0.0009142
9: -0.0036520, -0.0015916, -0.0036372, -0.0015677, -0.0010601, 0.0010273

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9948485, 0.9979604, 0.9948846, 0.9979826, -0.0015940, 0.0015447
1: -0.0025476, -0.0017722, -0.0025386, -0.0017667, -0.0003972, 0.0003849
2: -0.0006624, 0.0034467, -0.0006916, 0.0033992, -0.0020398, 0.0021048
3: -0.0028419, -0.0009716, -0.0028203, -0.0009583, -0.0009580, 0.0009284
4: 0.0003997, 0.0011950, 0.0003940, 0.0011858, -0.0003948, 0.0004074
5: -0.0018736, 0.0032945, -0.0019105, 0.0032348, -0.0025655, 0.0026473
6: 0.0007046, 0.0020164, 0.0007198, 0.0020257, -0.0006719, 0.0006512
7: -0.0013145, 0.0020794, -0.0012752, 0.0021035, -0.0017384, 0.0016847
8: -0.0002554, 0.0015294, -0.0002348, 0.0015421, -0.0009142, 0.0008860
9: -0.0036372, -0.0015677, -0.0036520, -0.0015916, -0.0010273, 0.0010601

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9948485, 0.9979604, 0.9948485, 0.9979604, -0.0015810, 0.0015810
1: -0.0025476, -0.0017722, -0.0025476, -0.0017722, -0.0003939, 0.0003939
2: -0.0006624, 0.0034467, -0.0006624, 0.0034467, -0.0020877, 0.0020877
3: -0.0028419, -0.0009716, -0.0028419, -0.0009716, -0.0009502, 0.0009502
4: 0.0003997, 0.0011950, 0.0003997, 0.0011950, -0.0004041, 0.0004041
5: -0.0018736, 0.0032945, -0.0018736, 0.0032945, -0.0026258, 0.0026258
6: 0.0007046, 0.0020164, 0.0007046, 0.0020164, -0.0006664, 0.0006664
7: -0.0013145, 0.0020794, -0.0013145, 0.0020794, -0.0017243, 0.0017243
8: -0.0002554, 0.0015294, -0.0002554, 0.0015294, -0.0009068, 0.0009068
9: -0.0036372, -0.0015677, -0.0036372, -0.0015677, -0.0010515, 0.0010515

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9948846, 0.9979826, 0.9949791, 0.9982597, -0.0019719, 0.0015849
1: -0.0025386, -0.0017667, -0.0025150, -0.0016976, -0.0004913, 0.0003949
2: -0.0006916, 0.0033992, -0.0010577, 0.0032744, -0.0020928, 0.0026039
3: -0.0028203, -0.0009583, -0.0027635, -0.0007917, -0.0011852, 0.0009526
4: 0.0003940, 0.0011858, 0.0003232, 0.0011616, -0.0004051, 0.0005040
5: -0.0019105, 0.0032348, -0.0023708, 0.0030777, -0.0026322, 0.0032750
6: 0.0007198, 0.0020257, 0.0007597, 0.0021426, -0.0008312, 0.0006681
7: -0.0012752, 0.0021035, -0.0011721, 0.0024059, -0.0021506, 0.0017285
8: -0.0002348, 0.0015421, -0.0001806, 0.0017011, -0.0011310, 0.0009090
9: -0.0036520, -0.0015916, -0.0038363, -0.0016545, -0.0010541, 0.0013114

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9948846, 0.9979826, 0.9949513, 0.9982269, -0.0019476, 0.0016171
1: -0.0025386, -0.0017667, -0.0025220, -0.0017058, -0.0004853, 0.0004029
2: -0.0006916, 0.0033992, -0.0010144, 0.0033111, -0.0021354, 0.0025718
3: -0.0028203, -0.0009583, -0.0027802, -0.0008114, -0.0011706, 0.0009719
4: 0.0003940, 0.0011858, 0.0003316, 0.0011687, -0.0004133, 0.0004978
5: -0.0019105, 0.0032348, -0.0023164, 0.0031240, -0.0026858, 0.0032346
6: 0.0007198, 0.0020257, 0.0007479, 0.0021288, -0.0008210, 0.0006817
7: -0.0012752, 0.0021035, -0.0012025, 0.0023701, -0.0021241, 0.0017637
8: -0.0002348, 0.0015421, -0.0001965, 0.0016823, -0.0011171, 0.0009275
9: -0.0036520, -0.0015916, -0.0038145, -0.0016360, -0.0010755, 0.0012953

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9948485, 0.9979604, 0.9949791, 0.9982597, -0.0019937, 0.0015574
1: -0.0025476, -0.0017722, -0.0025150, -0.0016976, -0.0004968, 0.0003881
2: -0.0006624, 0.0034467, -0.0010577, 0.0032744, -0.0020566, 0.0026326
3: -0.0028419, -0.0009716, -0.0027635, -0.0007917, -0.0011983, 0.0009361
4: 0.0003997, 0.0011950, 0.0003232, 0.0011616, -0.0003980, 0.0005095
5: -0.0018736, 0.0032945, -0.0023708, 0.0030777, -0.0025866, 0.0033112
6: 0.0007046, 0.0020164, 0.0007597, 0.0021426, -0.0008404, 0.0006565
7: -0.0013145, 0.0020794, -0.0011721, 0.0024059, -0.0021744, 0.0016986
8: -0.0002554, 0.0015294, -0.0001806, 0.0017011, -0.0011435, 0.0008933
9: -0.0036372, -0.0015677, -0.0038363, -0.0016545, -0.0010358, 0.0013259

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9948485, 0.9979604, 0.9949513, 0.9982269, -0.0019810, 0.0015919
1: -0.0025476, -0.0017722, -0.0025220, -0.0017058, -0.0004936, 0.0003966
2: -0.0006624, 0.0034467, -0.0010144, 0.0033111, -0.0021020, 0.0026159
3: -0.0028419, -0.0009716, -0.0027802, -0.0008114, -0.0011907, 0.0009568
4: 0.0003997, 0.0011950, 0.0003316, 0.0011687, -0.0004068, 0.0005063
5: -0.0018736, 0.0032945, -0.0023164, 0.0031240, -0.0026438, 0.0032902
6: 0.0007046, 0.0020164, 0.0007479, 0.0021288, -0.0008351, 0.0006710
7: -0.0013145, 0.0020794, -0.0012025, 0.0023701, -0.0021606, 0.0017361
8: -0.0002554, 0.0015294, -0.0001965, 0.0016823, -0.0011362, 0.0009130
9: -0.0036372, -0.0015677, -0.0038145, -0.0016360, -0.0010587, 0.0013175

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9949788, 0.9982597, 0.9948846, 0.9979826, -0.0017215, 0.0019719
1: -0.0025151, -0.0016976, -0.0025386, -0.0017667, -0.0004289, 0.0004913
2: -0.0010577, 0.0032747, -0.0006916, 0.0033992, -0.0026039, 0.0022732
3: -0.0027636, -0.0007917, -0.0028203, -0.0009583, -0.0010346, 0.0011852
4: 0.0003232, 0.0011617, 0.0003940, 0.0011858, -0.0005040, 0.0004400
5: -0.0023708, 0.0030781, -0.0019105, 0.0032348, -0.0032750, 0.0028590
6: 0.0007596, 0.0021426, 0.0007198, 0.0020257, -0.0007257, 0.0008312
7: -0.0011724, 0.0024059, -0.0012752, 0.0021035, -0.0018775, 0.0021506
8: -0.0001807, 0.0017011, -0.0002348, 0.0015421, -0.0009874, 0.0011310
9: -0.0038363, -0.0016543, -0.0036520, -0.0015916, -0.0013114, 0.0011449

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9949513, 0.9982269, 0.9948846, 0.9979826, -0.0016171, 0.0019476
1: -0.0025220, -0.0017058, -0.0025386, -0.0017667, -0.0004029, 0.0004853
2: -0.0010144, 0.0033111, -0.0006916, 0.0033992, -0.0025718, 0.0021354
3: -0.0027802, -0.0008114, -0.0028203, -0.0009583, -0.0009719, 0.0011706
4: 0.0003316, 0.0011687, 0.0003940, 0.0011858, -0.0004978, 0.0004133
5: -0.0023164, 0.0031240, -0.0019105, 0.0032348, -0.0032346, 0.0026858
6: 0.0007479, 0.0021288, 0.0007198, 0.0020257, -0.0006817, 0.0008210
7: -0.0012025, 0.0023701, -0.0012752, 0.0021035, -0.0017637, 0.0021241
8: -0.0001965, 0.0016823, -0.0002348, 0.0015421, -0.0009275, 0.0011171
9: -0.0038145, -0.0016360, -0.0036520, -0.0015916, -0.0012953, 0.0010755

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9949788, 0.9982597, 0.9948485, 0.9979604, -0.0016823, 0.0019937
1: -0.0025151, -0.0016976, -0.0025476, -0.0017722, -0.0004192, 0.0004968
2: -0.0010577, 0.0032747, -0.0006624, 0.0034467, -0.0026326, 0.0022214
3: -0.0027636, -0.0007917, -0.0028419, -0.0009716, -0.0010111, 0.0011983
4: 0.0003232, 0.0011617, 0.0003997, 0.0011950, -0.0005095, 0.0004300
5: -0.0023708, 0.0030781, -0.0018736, 0.0032945, -0.0033112, 0.0027940
6: 0.0007596, 0.0021426, 0.0007046, 0.0020164, -0.0007091, 0.0008404
7: -0.0011724, 0.0024059, -0.0013145, 0.0020794, -0.0018348, 0.0021744
8: -0.0001807, 0.0017011, -0.0002554, 0.0015294, -0.0009649, 0.0011435
9: -0.0038363, -0.0016543, -0.0036372, -0.0015677, -0.0013259, 0.0011188

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: B, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9949513, 0.9982269, 0.9948485, 0.9979604, -0.0015919, 0.0019810
1: -0.0025220, -0.0017058, -0.0025476, -0.0017722, -0.0003966, 0.0004936
2: -0.0010144, 0.0033111, -0.0006624, 0.0034467, -0.0026159, 0.0021020
3: -0.0027802, -0.0008114, -0.0028419, -0.0009716, -0.0009568, 0.0011907
4: 0.0003316, 0.0011687, 0.0003997, 0.0011950, -0.0005063, 0.0004068
5: -0.0023164, 0.0031240, -0.0018736, 0.0032945, -0.0032902, 0.0026438
6: 0.0007479, 0.0021288, 0.0007046, 0.0020164, -0.0006710, 0.0008351
7: -0.0012025, 0.0023701, -0.0013145, 0.0020794, -0.0017361, 0.0021606
8: -0.0001965, 0.0016823, -0.0002554, 0.0015294, -0.0009130, 0.0011362
9: -0.0038145, -0.0016360, -0.0036372, -0.0015677, -0.0013175, 0.0010587

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 66

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9949788, 0.9982597, 0.9949788, 0.9982597, -0.0017958, 0.0017958
1: -0.0025151, -0.0016976, -0.0025151, -0.0016976, -0.0004475, 0.0004475
2: -0.0010577, 0.0032747, -0.0010577, 0.0032747, -0.0023713, 0.0023713
3: -0.0027636, -0.0007917, -0.0027636, -0.0007917, -0.0010793, 0.0010793
4: 0.0003232, 0.0011617, 0.0003232, 0.0011617, -0.0004590, 0.0004590
5: -0.0023708, 0.0030781, -0.0023708, 0.0030781, -0.0029825, 0.0029825
6: 0.0007596, 0.0021426, 0.0007596, 0.0021426, -0.0007570, 0.0007570
7: -0.0011724, 0.0024059, -0.0011724, 0.0024059, -0.0019586, 0.0019586
8: -0.0001807, 0.0017011, -0.0001807, 0.0017011, -0.0010300, 0.0010300
9: -0.0038363, -0.0016543, -0.0038363, -0.0016543, -0.0011943, 0.0011943

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006343, upper bound: 0.0007744
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010694, upper bound: 0.0011229
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9949788, 0.9982597, 0.9949513, 0.9982269, -0.0017591, 0.0016512
1: -0.0025151, -0.0016976, -0.0025220, -0.0017058, -0.0004383, 0.0004114
2: -0.0010577, 0.0032747, -0.0010144, 0.0033111, -0.0021804, 0.0023228
3: -0.0027636, -0.0007917, -0.0027802, -0.0008114, -0.0010572, 0.0009924
4: 0.0003232, 0.0011617, 0.0003316, 0.0011687, -0.0004220, 0.0004496
5: -0.0023708, 0.0030781, -0.0023164, 0.0031240, -0.0027424, 0.0029215
6: 0.0007596, 0.0021426, 0.0007479, 0.0021288, -0.0007415, 0.0006960
7: -0.0011724, 0.0024059, -0.0012025, 0.0023701, -0.0019185, 0.0018009
8: -0.0001807, 0.0017011, -0.0001965, 0.0016823, -0.0010089, 0.0009471
9: -0.0038363, -0.0016543, -0.0038145, -0.0016360, -0.0010982, 0.0011699

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 113

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008205, upper bound: 0.0009194
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006711, upper bound: 0.0007131
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9949513, 0.9982269, 0.9949788, 0.9982597, -0.0016512, 0.0017591
1: -0.0025220, -0.0017058, -0.0025151, -0.0016976, -0.0004114, 0.0004383
2: -0.0010144, 0.0033111, -0.0010577, 0.0032747, -0.0023228, 0.0021804
3: -0.0027802, -0.0008114, -0.0027636, -0.0007917, -0.0009924, 0.0010572
4: 0.0003316, 0.0011687, 0.0003232, 0.0011617, -0.0004496, 0.0004220
5: -0.0023164, 0.0031240, -0.0023708, 0.0030781, -0.0029215, 0.0027424
6: 0.0007479, 0.0021288, 0.0007596, 0.0021426, -0.0006960, 0.0007415
7: -0.0012025, 0.0023701, -0.0011724, 0.0024059, -0.0018009, 0.0019185
8: -0.0001965, 0.0016823, -0.0001807, 0.0017011, -0.0009471, 0.0010089
9: -0.0038145, -0.0016360, -0.0038363, -0.0016543, -0.0011699, 0.0010982

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008240, upper bound: 0.0008096
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006500, upper bound: 0.0006746
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9949513, 0.9982269, 0.9949513, 0.9982269, -0.0016321, 0.0016321
1: -0.0025220, -0.0017058, -0.0025220, -0.0017058, -0.0004067, 0.0004067
2: -0.0010144, 0.0033111, -0.0010144, 0.0033111, -0.0021552, 0.0021552
3: -0.0027802, -0.0008114, -0.0027802, -0.0008114, -0.0009809, 0.0009809
4: 0.0003316, 0.0011687, 0.0003316, 0.0011687, -0.0004171, 0.0004171
5: -0.0023164, 0.0031240, -0.0023164, 0.0031240, -0.0027106, 0.0027106
6: 0.0007479, 0.0021288, 0.0007479, 0.0021288, -0.0006880, 0.0006880
7: -0.0012025, 0.0023701, -0.0012025, 0.0023701, -0.0017800, 0.0017800
8: -0.0001965, 0.0016823, -0.0001965, 0.0016823, -0.0009361, 0.0009361
9: -0.0038145, -0.0016360, -0.0038145, -0.0016360, -0.0010855, 0.0010855

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 113

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008240, upper bound: 0.0008096
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006500, upper bound: 0.0006746
time: 1.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.90 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0006143, upper bound: 0.0007607
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0010608, upper bound: 0.0010603
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0006343, upper bound: 0.0007744
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0010694, upper bound: 0.0011229
NS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0008205, upper bound: 0.0009194
NS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0006711, upper bound: 0.0007131
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0008240, upper bound: 0.0008096
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0006500, upper bound: 0.0006746
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0008240, upper bound: 0.0008096
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.90
Output dim: 0, lower bound: -0.0006500, upper bound: 0.0006746

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9948903, 0.9979823, 0.9948846, 0.9979826, -0.0015377, 0.0015700
1: -0.0025371, -0.0017667, -0.0025386, -0.0017667, -0.0003831, 0.0003912
2: -0.0006914, 0.0033915, -0.0006916, 0.0033992, -0.0020732, 0.0020305
3: -0.0028168, -0.0009584, -0.0028203, -0.0009583, -0.0009242, 0.0009436
4: 0.0003941, 0.0011843, 0.0003940, 0.0011858, -0.0004013, 0.0003930
5: -0.0019101, 0.0032251, -0.0019105, 0.0032348, -0.0026076, 0.0025538
6: 0.0007223, 0.0020256, 0.0007198, 0.0020257, -0.0006482, 0.0006618
7: -0.0012689, 0.0021033, -0.0012752, 0.0021035, -0.0016770, 0.0017124
8: -0.0002315, 0.0015420, -0.0002348, 0.0015421, -0.0008819, 0.0009005
9: -0.0036518, -0.0015955, -0.0036520, -0.0015916, -0.0010442, 0.0010227

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009023, upper bound: 0.0007763
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009023, upper bound: 0.0010638
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9949852, 0.9982595, 0.9949788, 0.9982597, -0.0018209, 0.0017945
1: -0.0025135, -0.0016976, -0.0025151, -0.0016976, -0.0004537, 0.0004471
2: -0.0010574, 0.0032663, -0.0010577, 0.0032747, -0.0023696, 0.0024045
3: -0.0027598, -0.0007918, -0.0027636, -0.0007917, -0.0010944, 0.0010785
4: 0.0003232, 0.0011601, 0.0003232, 0.0011617, -0.0004586, 0.0004654
5: -0.0023705, 0.0030675, -0.0023708, 0.0030781, -0.0029804, 0.0030243
6: 0.0007623, 0.0021425, 0.0007596, 0.0021426, -0.0007676, 0.0007564
7: -0.0011654, 0.0024056, -0.0011724, 0.0024059, -0.0019860, 0.0019572
8: -0.0001770, 0.0017010, -0.0001807, 0.0017011, -0.0010444, 0.0010292
9: -0.0038362, -0.0016586, -0.0038363, -0.0016543, -0.0011935, 0.0012110

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009106, upper bound: 0.0008669
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009106, upper bound: 0.0011253
time: 1.08 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.01 seconds
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.01
Output dim: 0, lower bound: -0.0009023, upper bound: 0.0007763
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -0.0009023, upper bound: 0.0010638
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.01
Output dim: 0, lower bound: -0.0009106, upper bound: 0.0008669
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.01
Output dim: 0, lower bound: -0.0009106, upper bound: 0.0011253

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9948903, 0.9979823, 0.9948903, 0.9979823, -0.0015366, 0.0015366
1: -0.0025371, -0.0017667, -0.0025371, -0.0017667, -0.0003829, 0.0003829
2: -0.0006914, 0.0033915, -0.0006914, 0.0033915, -0.0020291, 0.0020291
3: -0.0028168, -0.0009584, -0.0028168, -0.0009584, -0.0009236, 0.0009236
4: 0.0003941, 0.0011843, 0.0003941, 0.0011843, -0.0003927, 0.0003927
5: -0.0019101, 0.0032251, -0.0019101, 0.0032251, -0.0025521, 0.0025521
6: 0.0007223, 0.0020256, 0.0007223, 0.0020256, -0.0006478, 0.0006478
7: -0.0012689, 0.0021033, -0.0012689, 0.0021033, -0.0016759, 0.0016759
8: -0.0002315, 0.0015420, -0.0002315, 0.0015420, -0.0008814, 0.0008814
9: -0.0036518, -0.0015955, -0.0036518, -0.0015955, -0.0010220, 0.0010220

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9949852, 0.9982595, 0.9949852, 0.9982595, -0.0018201, 0.0018201
1: -0.0025135, -0.0016976, -0.0025135, -0.0016976, -0.0004535, 0.0004535
2: -0.0010574, 0.0032663, -0.0010574, 0.0032663, -0.0024034, 0.0024034
3: -0.0027598, -0.0007918, -0.0027598, -0.0007918, -0.0010939, 0.0010939
4: 0.0003232, 0.0011601, 0.0003232, 0.0011601, -0.0004652, 0.0004652
5: -0.0023705, 0.0030675, -0.0023705, 0.0030675, -0.0030229, 0.0030229
6: 0.0007623, 0.0021425, 0.0007623, 0.0021425, -0.0007672, 0.0007672
7: -0.0011654, 0.0024056, -0.0011654, 0.0024056, -0.0019851, 0.0019851
8: -0.0001770, 0.0017010, -0.0001770, 0.0017010, -0.0010439, 0.0010439
9: -0.0038362, -0.0016586, -0.0038362, -0.0016586, -0.0012105, 0.0012105

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 251

### Candidate
type: A, layer: 1, pos: 204

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 204

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 66

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.81 + 218.13 = 221.95 seconds
