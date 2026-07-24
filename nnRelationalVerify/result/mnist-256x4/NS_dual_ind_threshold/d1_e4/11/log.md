## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.014896192999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0076237, 0.0055722, -0.0076237, 0.0055722, -0.0131959, 0.0131959)
1: (0.9907092, 1.0064780, 0.9907092, 1.0064780, -0.0157687, 0.0157687)
2: (-0.0219780, 0.0113110, -0.0219780, 0.0113110, -0.0327574, 0.0327574)
3: (-0.0027511, 0.0050303, -0.0027511, 0.0050303, -0.0077814, 0.0077814)
4: (-0.0121325, 0.0157801, -0.0121325, 0.0157801, -0.0279126, 0.0279126)
5: (-0.0040185, 0.0179425, -0.0040185, 0.0179425, -0.0219610, 0.0219610)
6: (-0.0067780, 0.0077860, -0.0067780, 0.0077860, -0.0145640, 0.0145640)
7: (-0.0124661, 0.0050048, -0.0124661, 0.0050048, -0.0174709, 0.0174709)
8: (-0.0125516, 0.0275235, -0.0125516, 0.0275235, -0.0398314, 0.0398314)
9: (-0.0122965, 0.0064546, -0.0122965, 0.0064546, -0.0187511, 0.0187511)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 3.58 = 5.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0153569, upper bound: 0.0153569

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 130

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153189, upper bound: 0.0153384
time: 2.39 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0153384, upper bound: 0.0153384
time: 2.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.26 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.26
Output dim: 1, lower bound: -0.0153189, upper bound: 0.0153384
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.26
Output dim: 1, lower bound: -0.0153384, upper bound: 0.0153384

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0074553, 0.0046652, -0.0071427, 0.0052024, -0.0126577, 0.0118078
1: 0.9904237, 1.0052683, 0.9909374, 1.0059847, -0.0155610, 0.0143309
2: -0.0205615, 0.0106047, -0.0213816, 0.0108700, -0.0309194, 0.0315101
3: -0.0027631, 0.0049361, -0.0024212, 0.0049912, -0.0077543, 0.0073573
4: -0.0111205, 0.0146627, -0.0116639, 0.0153103, -0.0264308, 0.0263266
5: -0.0030602, 0.0169155, -0.0036087, 0.0172191, -0.0202794, 0.0205242
6: -0.0062864, 0.0069538, -0.0065675, 0.0073883, -0.0136747, 0.0135213
7: -0.0120463, 0.0054753, -0.0122949, 0.0043070, -0.0163533, 0.0177703
8: -0.0118991, 0.0257418, -0.0122855, 0.0267127, -0.0383697, 0.0377899
9: -0.0109855, 0.0060923, -0.0116355, 0.0063069, -0.0172924, 0.0177278

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 130

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152579, upper bound: 0.0152611
time: 2.48 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152575, upper bound: 0.0152784
time: 2.74 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0071862, 0.0052366, -0.0074827, 0.0054604, -0.0126466, 0.0127193
1: 0.9909216, 1.0060304, 0.9907764, 1.0063289, -0.0154073, 0.0152540
2: -0.0214789, 0.0109097, -0.0218208, 0.0111790, -0.0320261, 0.0322063
3: -0.0024499, 0.0049976, -0.0026558, 0.0050200, -0.0074699, 0.0076534
4: -0.0117073, 0.0153870, -0.0119916, 0.0156563, -0.0273636, 0.0273786
5: -0.0036759, 0.0172837, -0.0039103, 0.0177265, -0.0214024, 0.0211941
6: -0.0066018, 0.0074520, -0.0067223, 0.0076800, -0.0142818, 0.0141743
7: -0.0123108, 0.0043678, -0.0124144, 0.0048024, -0.0171132, 0.0167822
8: -0.0123102, 0.0268471, -0.0124712, 0.0273101, -0.0393864, 0.0390506
9: -0.0116959, 0.0063206, -0.0120979, 0.0064100, -0.0181059, 0.0184185

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 130

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152795, upper bound: 0.0152615
time: 2.90 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152790, upper bound: 0.0152790
time: 2.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.20 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.20
Output dim: 1, lower bound: -0.0152579, upper bound: 0.0152611
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.20
Output dim: 1, lower bound: -0.0152575, upper bound: 0.0152784
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.20
Output dim: 1, lower bound: -0.0152795, upper bound: 0.0152615
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.20
Output dim: 1, lower bound: -0.0152790, upper bound: 0.0152790

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0045861, 0.0044980, -0.0042278, 0.0047713, -0.0093575, 0.0087258
1: 0.9921400, 1.0050454, 0.9922239, 1.0054100, -0.0132700, 0.0128216
2: -0.0204769, 0.0094036, -0.0212534, 0.0094333, -0.0290786, 0.0298620
3: -0.0007937, 0.0049344, -0.0009258, 0.0049867, -0.0057804, 0.0058602
4: -0.0105451, 0.0146001, -0.0107816, 0.0152145, -0.0257596, 0.0253817
5: -0.0029594, 0.0146401, -0.0034828, 0.0146282, -0.0175877, 0.0181229
6: -0.0062336, 0.0066065, -0.0064983, 0.0070476, -0.0132811, 0.0131048
7: -0.0119690, -0.0006318, -0.0120955, -0.0016867, -0.0102822, 0.0114637
8: -0.0117789, 0.0252378, -0.0119755, 0.0261759, -0.0377118, 0.0369740
9: -0.0098686, 0.0060255, -0.0101290, 0.0061347, -0.0160034, 0.0161545

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152238, upper bound: 0.0152433
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152365, upper bound: 0.0152418
time: 2.64 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0040172, 0.0044578, -0.0046580, 0.0054119, -0.0094291, 0.0091158
1: 0.9923000, 1.0049920, 0.9922132, 1.0062642, -0.0139642, 0.0127788
2: -0.0204512, 0.0091361, -0.0213653, 0.0100403, -0.0296523, 0.0296481
3: -0.0006353, 0.0049336, -0.0009663, 0.0049941, -0.0056293, 0.0058999
4: -0.0104142, 0.0145805, -0.0115321, 0.0153030, -0.0257171, 0.0261127
5: -0.0029325, 0.0141484, -0.0035596, 0.0156085, -0.0185409, 0.0177079
6: -0.0062199, 0.0065387, -0.0065372, 0.0071186, -0.0133385, 0.0130759
7: -0.0119504, -0.0018395, -0.0123919, -0.0013747, -0.0105757, 0.0105524
8: -0.0117499, 0.0251220, -0.0124363, 0.0263230, -0.0378295, 0.0373194
9: -0.0096200, 0.0060095, -0.0111687, 0.0063906, -0.0160106, 0.0171781

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152063, upper bound: 0.0152572
time: 2.38 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152053, upper bound: 0.0152249
time: 2.37 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0044379, 0.0050763, -0.0044032, 0.0050325, -0.0094703, 0.0094795
1: 0.9922092, 1.0058167, 0.9921826, 1.0057582, -0.0135490, 0.0136341
2: -0.0213926, 0.0097247, -0.0216873, 0.0096807, -0.0301193, 0.0305855
3: -0.0009774, 0.0049959, -0.0010829, 0.0050154, -0.0059928, 0.0060788
4: -0.0111398, 0.0153245, -0.0110876, 0.0155574, -0.0266972, 0.0264121
5: -0.0035784, 0.0150993, -0.0037805, 0.0150278, -0.0186062, 0.0188798
6: -0.0065467, 0.0071364, -0.0066489, 0.0073229, -0.0138696, 0.0137854
7: -0.0122366, -0.0015268, -0.0122163, -0.0015595, -0.0106771, 0.0106895
8: -0.0121949, 0.0263597, -0.0121633, 0.0267460, -0.0387065, 0.0382571
9: -0.0106258, 0.0062565, -0.0105528, 0.0062390, -0.0168649, 0.0168094

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152351, upper bound: 0.0152465
time: 2.49 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152066, upper bound: 0.0152184
time: 2.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0044084, 0.0050402, -0.0048346, 0.0056747, -0.0100831, 0.0098748
1: 0.9922119, 1.0057687, 0.9921721, 1.0066148, -0.0144029, 0.0135965
2: -0.0213791, 0.0096881, -0.0217978, 0.0102894, -0.0307137, 0.0306582
3: -0.0009713, 0.0049950, -0.0011230, 0.0050227, -0.0059940, 0.0061180
4: -0.0110966, 0.0153139, -0.0118402, 0.0156448, -0.0267414, 0.0271541
5: -0.0035690, 0.0150397, -0.0038563, 0.0160108, -0.0195798, 0.0188960
6: -0.0065420, 0.0071273, -0.0066873, 0.0073930, -0.0139350, 0.0138146
7: -0.0122199, -0.0015557, -0.0125135, -0.0012466, -0.0109733, 0.0109578
8: -0.0121689, 0.0263412, -0.0126254, 0.0268913, -0.0388251, 0.0387002
9: -0.0105654, 0.0062421, -0.0115954, 0.0064956, -0.0170610, 0.0178375

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152340, upper bound: 0.0152603
time: 2.59 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152340, upper bound: 0.0152340
time: 2.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.70 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152238, upper bound: 0.0152433
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152365, upper bound: 0.0152418
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152063, upper bound: 0.0152572
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152053, upper bound: 0.0152249
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152351, upper bound: 0.0152465
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152066, upper bound: 0.0152184
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152340, upper bound: 0.0152603
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.70
Output dim: 1, lower bound: -0.0152340, upper bound: 0.0152340

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0034095, 0.0035531, -0.0041192, 0.0046097, -0.0080192, 0.0076723
1: 0.9924093, 1.0037851, 0.9922672, 1.0051944, -0.0127851, 0.0115179
2: -0.0193011, 0.0082788, -0.0207968, 0.0092801, -0.0277447, 0.0282232
3: -0.0002186, 0.0048574, -0.0007604, 0.0049564, -0.0051751, 0.0056179
4: -0.0093540, 0.0136716, -0.0105922, 0.0148537, -0.0242076, 0.0242637
5: -0.0021434, 0.0127638, -0.0031696, 0.0143809, -0.0165243, 0.0159333
6: -0.0058207, 0.0058091, -0.0063398, 0.0067579, -0.0125786, 0.0121489
7: -0.0115317, -0.0022803, -0.0120207, -0.0017655, -0.0097662, 0.0097404
8: -0.0110991, 0.0236109, -0.0118592, 0.0255760, -0.0364362, 0.0352323
9: -0.0081515, 0.0056480, -0.0098666, 0.0060702, -0.0142217, 0.0155146

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151940, upper bound: 0.0151967
time: 2.41 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151808, upper bound: 0.0151968
time: 2.44 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0040578, 0.0042108, -0.0041616, 0.0046727, -0.0087305, 0.0083724
1: 0.9922804, 1.0046623, 0.9922392, 1.0052786, -0.0129982, 0.0124231
2: -0.0200516, 0.0089937, -0.0210919, 0.0093398, -0.0284637, 0.0292535
3: -0.0005436, 0.0049068, -0.0008673, 0.0049760, -0.0055196, 0.0057741
4: -0.0101581, 0.0142645, -0.0106660, 0.0150869, -0.0252450, 0.0249305
5: -0.0026623, 0.0139419, -0.0033720, 0.0144773, -0.0171396, 0.0173139
6: -0.0060830, 0.0063061, -0.0064423, 0.0069451, -0.0130281, 0.0127484
7: -0.0118361, -0.0015090, -0.0120498, -0.0017348, -0.0101013, 0.0105408
8: -0.0115722, 0.0246306, -0.0119046, 0.0259638, -0.0372939, 0.0362726
9: -0.0092927, 0.0059108, -0.0099689, 0.0060953, -0.0153881, 0.0158797

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152098, upper bound: 0.0151967
time: 2.72 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151927, upper bound: 0.0151968
time: 2.90 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0035718, 0.0037946, -0.0045077, 0.0051881, -0.0087598, 0.0083023
1: 0.9923697, 1.0041072, 0.9922386, 1.0059658, -0.0135961, 0.0118687
2: -0.0197175, 0.0085077, -0.0210984, 0.0098282, -0.0287087, 0.0287551
3: -0.0003695, 0.0048850, -0.0008696, 0.0049764, -0.0053459, 0.0057546
4: -0.0096370, 0.0140007, -0.0112699, 0.0150920, -0.0247290, 0.0252706
5: -0.0024291, 0.0131335, -0.0033764, 0.0152660, -0.0176951, 0.0165099
6: -0.0059652, 0.0060733, -0.0064445, 0.0069493, -0.0129145, 0.0125178
7: -0.0116435, -0.0021626, -0.0122883, -0.0014837, -0.0101598, 0.0101257
8: -0.0112729, 0.0241580, -0.0122753, 0.0259722, -0.0370014, 0.0361946
9: -0.0085436, 0.0057445, -0.0108054, 0.0063012, -0.0148448, 0.0165499

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151919, upper bound: 0.0152304
time: 2.87 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151919, upper bound: 0.0152414
time: 3.06 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0047036, 0.0054797, -0.0045191, 0.0052051, -0.0099087, 0.0099989
1: 0.9923511, 1.0063549, 0.9922346, 1.0059885, -0.0136374, 0.0141203
2: -0.0199148, 0.0101046, -0.0211398, 0.0098443, -0.0289161, 0.0303995
3: -0.0004409, 0.0048981, -0.0008847, 0.0049791, -0.0054201, 0.0057827
4: -0.0116117, 0.0141566, -0.0112898, 0.0151248, -0.0267365, 0.0254464
5: -0.0025645, 0.0157124, -0.0034049, 0.0152920, -0.0178565, 0.0191173
6: -0.0060337, 0.0061984, -0.0064589, 0.0069755, -0.0130092, 0.0126573
7: -0.0124233, -0.0013416, -0.0122962, -0.0014754, -0.0109479, 0.0109546
8: -0.0124851, 0.0244172, -0.0122875, 0.0260267, -0.0382695, 0.0364600
9: -0.0112789, 0.0064177, -0.0108330, 0.0063080, -0.0175869, 0.0172507

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151911, upper bound: 0.0152043
time: 2.46 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151910, upper bound: 0.0152122
time: 3.26 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0040009, 0.0044335, -0.0042544, 0.0048110, -0.0088119, 0.0086879
1: 0.9922852, 1.0049595, 0.9922084, 1.0054629, -0.0131777, 0.0127511
2: -0.0206077, 0.0091131, -0.0214166, 0.0094709, -0.0291273, 0.0297048
3: -0.0006919, 0.0049439, -0.0009849, 0.0049975, -0.0056894, 0.0059288
4: -0.0103857, 0.0147042, -0.0108280, 0.0153435, -0.0257292, 0.0255322
5: -0.0030398, 0.0141112, -0.0035947, 0.0146889, -0.0177287, 0.0177060
6: -0.0062742, 0.0066380, -0.0065550, 0.0071511, -0.0134253, 0.0131929
7: -0.0119391, -0.0018513, -0.0121138, -0.0016674, -0.0102717, 0.0102625
8: -0.0117325, 0.0253275, -0.0120040, 0.0263903, -0.0378889, 0.0370651
9: -0.0095806, 0.0059998, -0.0101934, 0.0061506, -0.0157312, 0.0161931

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0152202
time: 2.32 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0152308
time: 2.73 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0134516, 0.0063284, -0.0042711, 0.0048357, -0.0182874, 0.0105995
1: 0.9855510, 1.0074867, 0.9922036, 1.0054957, -0.0199447, 0.0152831
2: -0.0211227, 0.0140671, -0.0214669, 0.0094943, -0.0297798, 0.0355339
3: -0.0072354, 0.0049641, -0.0010031, 0.0050008, -0.0122361, 0.0059672
4: -0.0137157, 0.0150825, -0.0108570, 0.0153832, -0.0290989, 0.0259395
5: -0.0034968, 0.0230975, -0.0036292, 0.0147268, -0.0182236, 0.0267267
6: -0.0065489, 0.0081086, -0.0065724, 0.0071830, -0.0137319, 0.0146810
7: -0.0128161, 0.0170784, -0.0121253, -0.0016554, -0.0111607, 0.0292037
8: -0.0130956, 0.0271923, -0.0120218, 0.0264564, -0.0393187, 0.0389417
9: -0.0151063, 0.0067568, -0.0102335, 0.0061604, -0.0212668, 0.0169903

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0151984
time: 2.53 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0152054
time: 2.64 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0039771, 0.0043981, -0.0046828, 0.0054488, -0.0094259, 0.0090809
1: 0.9922865, 1.0049121, 0.9921978, 1.0063133, -0.0140268, 0.0127143
2: -0.0205945, 0.0090796, -0.0215278, 0.0100752, -0.0297185, 0.0297817
3: -0.0006871, 0.0049430, -0.0010252, 0.0050048, -0.0056920, 0.0059682
4: -0.0103442, 0.0146938, -0.0115754, 0.0154314, -0.0257755, 0.0262692
5: -0.0030308, 0.0140570, -0.0036710, 0.0156650, -0.0186957, 0.0177280
6: -0.0062696, 0.0066296, -0.0065936, 0.0072216, -0.0134913, 0.0132232
7: -0.0119227, -0.0018686, -0.0124090, -0.0013567, -0.0105661, 0.0105404
8: -0.0117070, 0.0253103, -0.0124628, 0.0265364, -0.0380085, 0.0375065
9: -0.0095231, 0.0059856, -0.0112285, 0.0064053, -0.0159285, 0.0172142

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152211, upper bound: 0.0152335
time: 2.85 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152209, upper bound: 0.0152442
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0052484, 0.0062909, -0.0046980, 0.0054714, -0.0107198, 0.0109889
1: 0.9922585, 1.0074366, 0.9921930, 1.0063436, -0.0140851, 0.0152436
2: -0.0208876, 0.0108733, -0.0215771, 0.0100967, -0.0300286, 0.0316304
3: -0.0007933, 0.0049624, -0.0010431, 0.0050081, -0.0058014, 0.0060055
4: -0.0125622, 0.0149254, -0.0116020, 0.0154704, -0.0280326, 0.0265273
5: -0.0032318, 0.0169537, -0.0037049, 0.0156996, -0.0189314, 0.0206586
6: -0.0063713, 0.0068155, -0.0066107, 0.0072530, -0.0136243, 0.0134262
7: -0.0127987, -0.0009464, -0.0124194, -0.0013457, -0.0114530, 0.0114731
8: -0.0130686, 0.0256953, -0.0124791, 0.0266013, -0.0394357, 0.0379068
9: -0.0125955, 0.0067417, -0.0112653, 0.0064144, -0.0190098, 0.0180071

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152211, upper bound: 0.0152145
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152209, upper bound: 0.0152209
time: 2.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.62 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0151940, upper bound: 0.0151967
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0151808, upper bound: 0.0151968
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152098, upper bound: 0.0151967
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0151927, upper bound: 0.0151968
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0151919, upper bound: 0.0152304
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0151919, upper bound: 0.0152414
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0151911, upper bound: 0.0152043
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0151910, upper bound: 0.0152122
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0152202
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0152308
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0151984
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152224, upper bound: 0.0152054
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152211, upper bound: 0.0152335
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152209, upper bound: 0.0152442
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152211, upper bound: 0.0152145
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.62
Output dim: 1, lower bound: -0.0152209, upper bound: 0.0152209

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0032389, 0.0032990, -0.0037105, 0.0040012, -0.0072401, 0.0070095
1: 0.9924342, 1.0034465, 0.9923397, 1.0043827, -0.0119485, 0.0111068
2: -0.0190391, 0.0080380, -0.0200347, 0.0087034, -0.0269127, 0.0272285
3: -0.0001238, 0.0048401, -0.0004844, 0.0049060, -0.0050297, 0.0053245
4: -0.0090563, 0.0134645, -0.0098791, 0.0142514, -0.0233077, 0.0233436
5: -0.0019637, 0.0123750, -0.0026467, 0.0134496, -0.0154133, 0.0150218
6: -0.0057297, 0.0056429, -0.0060753, 0.0062745, -0.0120042, 0.0117182
7: -0.0114141, -0.0024041, -0.0117391, -0.0020620, -0.0093521, 0.0093350
8: -0.0109163, 0.0232667, -0.0114214, 0.0245748, -0.0352531, 0.0344528
9: -0.0077392, 0.0055466, -0.0088789, 0.0058271, -0.0135662, 0.0144254

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150904
time: 2.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151546
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0032531, 0.0033201, -0.0049715, 0.0058786, -0.0091317, 0.0082917
1: 0.9924313, 1.0034746, 0.9923162, 1.0068868, -0.0144555, 0.0111583
2: -0.0190689, 0.0080580, -0.0202807, 0.0104826, -0.0287172, 0.0274879
3: -0.0001345, 0.0048421, -0.0005735, 0.0049223, -0.0050568, 0.0054155
4: -0.0090810, 0.0134880, -0.0120791, 0.0144458, -0.0235267, 0.0255672
5: -0.0019841, 0.0124073, -0.0028155, 0.0163228, -0.0183069, 0.0152228
6: -0.0057401, 0.0056618, -0.0061607, 0.0064305, -0.0121706, 0.0118224
7: -0.0114239, -0.0023938, -0.0126079, -0.0011473, -0.0102766, 0.0102141
8: -0.0109315, 0.0233058, -0.0127720, 0.0248979, -0.0355855, 0.0358378
9: -0.0077734, 0.0055550, -0.0119263, 0.0065771, -0.0143505, 0.0174813

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150904
time: 2.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0151546
time: 2.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0036831, 0.0039603, -0.0037529, 0.0040644, -0.0077474, 0.0077133
1: 0.9923630, 1.0043285, 0.9923110, 1.0044669, -0.0121039, 0.0120175
2: -0.0197894, 0.0086647, -0.0203364, 0.0087633, -0.0276225, 0.0281483
3: -0.0003955, 0.0048897, -0.0005937, 0.0049259, -0.0053215, 0.0054834
4: -0.0098312, 0.0140574, -0.0099531, 0.0144898, -0.0243210, 0.0240105
5: -0.0024784, 0.0133871, -0.0028537, 0.0135463, -0.0160246, 0.0162408
6: -0.0059902, 0.0061188, -0.0061800, 0.0064659, -0.0124560, 0.0122989
7: -0.0117201, -0.0020819, -0.0117683, -0.0020312, -0.0096890, 0.0096864
8: -0.0113921, 0.0242524, -0.0114669, 0.0249712, -0.0361208, 0.0354567
9: -0.0088126, 0.0058107, -0.0089814, 0.0058523, -0.0146649, 0.0147921

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150904
time: 2.95 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151545
time: 2.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0036985, 0.0039833, -0.0050165, 0.0059455, -0.0096440, 0.0089997
1: 0.9923610, 1.0043589, 0.9922882, 1.0069760, -0.0146150, 0.0120707
2: -0.0198093, 0.0086865, -0.0205751, 0.0105460, -0.0294307, 0.0284109
3: -0.0004027, 0.0048911, -0.0006801, 0.0049418, -0.0053445, 0.0055712
4: -0.0098581, 0.0140732, -0.0121575, 0.0146785, -0.0245366, 0.0262307
5: -0.0024920, 0.0134222, -0.0030175, 0.0164252, -0.0189172, 0.0164397
6: -0.0059970, 0.0061315, -0.0062629, 0.0066173, -0.0126144, 0.0123943
7: -0.0117308, -0.0020707, -0.0126389, -0.0011147, -0.0106161, 0.0105682
8: -0.0114086, 0.0242786, -0.0128202, 0.0252848, -0.0364488, 0.0368358
9: -0.0088498, 0.0058199, -0.0120349, 0.0066038, -0.0154536, 0.0178548

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150904
time: 2.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151545
time: 2.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0034530, 0.0036178, -0.0039313, 0.0043299, -0.0077829, 0.0075491
1: 0.9924146, 1.0038713, 0.9923474, 1.0048214, -0.0124068, 0.0115240
2: -0.0192453, 0.0083401, -0.0199530, 0.0090150, -0.0274308, 0.0274432
3: -0.0001985, 0.0048537, -0.0004548, 0.0049006, -0.0050990, 0.0053085
4: -0.0094298, 0.0136275, -0.0102643, 0.0141868, -0.0236165, 0.0238917
5: -0.0021051, 0.0128628, -0.0025907, 0.0139527, -0.0160578, 0.0154535
6: -0.0058013, 0.0057737, -0.0060469, 0.0062226, -0.0120239, 0.0118206
7: -0.0115616, -0.0022488, -0.0118912, -0.0019018, -0.0096598, 0.0096424
8: -0.0111456, 0.0235376, -0.0116579, 0.0244674, -0.0353711, 0.0349592
9: -0.0082565, 0.0056739, -0.0094125, 0.0059584, -0.0142149, 0.0150864

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151306
time: 2.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151863
time: 2.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0034986, 0.0036857, -0.0043222, 0.0049119, -0.0084105, 0.0080079
1: 0.9923842, 1.0039620, 0.9922795, 1.0055975, -0.0132133, 0.0116825
2: -0.0195653, 0.0084045, -0.0206678, 0.0095665, -0.0282920, 0.0281304
3: -0.0003144, 0.0048749, -0.0007137, 0.0049479, -0.0052623, 0.0055886
4: -0.0095094, 0.0138804, -0.0109462, 0.0147517, -0.0242610, 0.0248266
5: -0.0023247, 0.0129668, -0.0030810, 0.0148433, -0.0171680, 0.0160478
6: -0.0059124, 0.0059767, -0.0062950, 0.0066761, -0.0125885, 0.0122718
7: -0.0115931, -0.0022157, -0.0121605, -0.0016183, -0.0099748, 0.0099448
8: -0.0111945, 0.0239580, -0.0120766, 0.0254065, -0.0363341, 0.0357968
9: -0.0083668, 0.0057010, -0.0103571, 0.0061909, -0.0145577, 0.0160581

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 131

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151403
time: 2.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151501, upper bound: 0.0151961
time: 2.74 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0045861, 0.0053048, -0.0039431, 0.0043474, -0.0089335, 0.0092479
1: 0.9923949, 1.0061215, 0.9923440, 1.0048445, -0.0124497, 0.0137776
2: -0.0194531, 0.0099389, -0.0199890, 0.0090315, -0.0276427, 0.0290842
3: -0.0002737, 0.0048675, -0.0004678, 0.0049030, -0.0051767, 0.0053353
4: -0.0114067, 0.0137917, -0.0102848, 0.0142153, -0.0256220, 0.0240765
5: -0.0022477, 0.0154447, -0.0026154, 0.0139795, -0.0162272, 0.0180600
6: -0.0058734, 0.0059055, -0.0060595, 0.0062455, -0.0121189, 0.0119650
7: -0.0123423, -0.0014268, -0.0118993, -0.0018933, -0.0104490, 0.0104725
8: -0.0123593, 0.0238106, -0.0116705, 0.0245148, -0.0366322, 0.0352374
9: -0.0109949, 0.0063478, -0.0094409, 0.0059654, -0.0169603, 0.0157887

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151621, upper bound: 0.0151676
time: 2.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151763, upper bound: 0.0151893
time: 2.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0046346, 0.0053770, -0.0043362, 0.0049327, -0.0095673, 0.0097132
1: 0.9923661, 1.0062177, 0.9922763, 1.0056251, -0.0132591, 0.0139415
2: -0.0197561, 0.0100073, -0.0207015, 0.0095862, -0.0285010, 0.0297797
3: -0.0003835, 0.0048875, -0.0007259, 0.0049501, -0.0053336, 0.0056134
4: -0.0114913, 0.0140311, -0.0109706, 0.0147783, -0.0262697, 0.0250017
5: -0.0024555, 0.0155552, -0.0031042, 0.0148751, -0.0173307, 0.0186594
6: -0.0059786, 0.0060977, -0.0063068, 0.0066975, -0.0126761, 0.0124045
7: -0.0123758, -0.0013916, -0.0121701, -0.0016081, -0.0107676, 0.0107785
8: -0.0124112, 0.0242087, -0.0120915, 0.0254509, -0.0375972, 0.0360569
9: -0.0111121, 0.0063767, -0.0103909, 0.0061992, -0.0173113, 0.0167675

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 173

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151618, upper bound: 0.0151736
time: 2.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151762, upper bound: 0.0151971
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038906, 0.0042693, -0.0036981, 0.0039827, -0.0078733, 0.0079674
1: 0.9923272, 1.0047406, 0.9923165, 1.0043582, -0.0120310, 0.0124241
2: -0.0201659, 0.0089575, -0.0202791, 0.0086859, -0.0278912, 0.0284135
3: -0.0005319, 0.0049147, -0.0005729, 0.0049222, -0.0054541, 0.0054875
4: -0.0101933, 0.0143551, -0.0098574, 0.0144445, -0.0246378, 0.0242125
5: -0.0027367, 0.0138600, -0.0028143, 0.0134213, -0.0161581, 0.0166743
6: -0.0061209, 0.0063577, -0.0061601, 0.0064295, -0.0125503, 0.0125179
7: -0.0118631, -0.0019313, -0.0117305, -0.0020710, -0.0097922, 0.0097992
8: -0.0116144, 0.0247472, -0.0114082, 0.0248958, -0.0362779, 0.0358872
9: -0.0093141, 0.0059342, -0.0088489, 0.0058197, -0.0151338, 0.0147831

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151207, upper bound: 0.0151173
time: 2.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151804, upper bound: 0.0151774
time: 2.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0039323, 0.0043314, -0.0040739, 0.0045421, -0.0084744, 0.0084052
1: 0.9923013, 1.0048232, 0.9922507, 1.0051043, -0.0128030, 0.0125725
2: -0.0204386, 0.0090163, -0.0209706, 0.0092161, -0.0287059, 0.0290789
3: -0.0006307, 0.0049327, -0.0008234, 0.0049680, -0.0055986, 0.0057561
4: -0.0102660, 0.0145705, -0.0105130, 0.0149910, -0.0252570, 0.0250835
5: -0.0029238, 0.0139549, -0.0032888, 0.0142775, -0.0172012, 0.0172437
6: -0.0062155, 0.0065307, -0.0064002, 0.0068682, -0.0130837, 0.0129309
7: -0.0118919, -0.0019011, -0.0119894, -0.0017984, -0.0100934, 0.0100883
8: -0.0116590, 0.0251054, -0.0118106, 0.0258044, -0.0372051, 0.0366508
9: -0.0094148, 0.0059589, -0.0097570, 0.0060432, -0.0154580, 0.0157159

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150996, upper bound: 0.0151264
time: 2.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151796, upper bound: 0.0151874
time: 2.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0130800, 0.0061657, -0.0037155, 0.0040087, -0.0170887, 0.0098813
1: 0.9858088, 1.0072696, 0.9923117, 1.0043929, -0.0185840, 0.0149579
2: -0.0206722, 0.0138050, -0.0203286, 0.0087105, -0.0285285, 0.0341332
3: -0.0069516, 0.0049347, -0.0005908, 0.0049254, -0.0118770, 0.0055255
4: -0.0134780, 0.0147286, -0.0098879, 0.0144836, -0.0279616, 0.0246165
5: -0.0031890, 0.0226661, -0.0028483, 0.0134611, -0.0166501, 0.0255144
6: -0.0063884, 0.0078241, -0.0061773, 0.0064609, -0.0128493, 0.0140014
7: -0.0127408, 0.0164324, -0.0117425, -0.0020583, -0.0106825, 0.0281750
8: -0.0129786, 0.0265638, -0.0114269, 0.0249609, -0.0377056, 0.0377166
9: -0.0147727, 0.0066917, -0.0088911, 0.0058300, -0.0206028, 0.0155828

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148178, upper bound: 0.0147911
time: 2.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151925, upper bound: 0.0151694
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0132291, 0.0062302, -0.0040933, 0.0045712, -0.0178003, 0.0103236
1: 0.9857104, 1.0073557, 0.9922459, 1.0051429, -0.0194325, 0.0151098
2: -0.0209508, 0.0139104, -0.0210211, 0.0092436, -0.0293607, 0.0348667
3: -0.0070715, 0.0049529, -0.0008416, 0.0049713, -0.0120427, 0.0057946
4: -0.0135759, 0.0149478, -0.0105470, 0.0150309, -0.0286068, 0.0254948
5: -0.0033806, 0.0228366, -0.0033234, 0.0143219, -0.0177024, 0.0261600
6: -0.0064873, 0.0079854, -0.0064177, 0.0069002, -0.0133875, 0.0144031
7: -0.0127706, 0.0166863, -0.0120028, -0.0017843, -0.0109863, 0.0286892
8: -0.0130250, 0.0269527, -0.0118315, 0.0258707, -0.0386394, 0.0385131
9: -0.0149061, 0.0067175, -0.0098041, 0.0060547, -0.0209609, 0.0165216

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0148168, upper bound: 0.0147953
time: 2.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151923, upper bound: 0.0151756
time: 2.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038670, 0.0042342, -0.0041182, 0.0046082, -0.0084752, 0.0083524
1: 0.9923285, 1.0046935, 0.9923056, 1.0051924, -0.0128639, 0.0123879
2: -0.0201518, 0.0089243, -0.0203919, 0.0092787, -0.0284727, 0.0284933
3: -0.0005268, 0.0049137, -0.0006138, 0.0049296, -0.0054565, 0.0055275
4: -0.0101521, 0.0143439, -0.0105904, 0.0145337, -0.0246858, 0.0249343
5: -0.0027271, 0.0138062, -0.0028918, 0.0143786, -0.0171056, 0.0166980
6: -0.0061160, 0.0063488, -0.0061993, 0.0065011, -0.0126170, 0.0125481
7: -0.0118469, -0.0019484, -0.0120200, -0.0017662, -0.0100807, 0.0100715
8: -0.0115891, 0.0247286, -0.0118581, 0.0250441, -0.0363980, 0.0363195
9: -0.0092571, 0.0059201, -0.0098642, 0.0060695, -0.0153267, 0.0157843

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151148, upper bound: 0.0151310
time: 2.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151800, upper bound: 0.0151901
time: 2.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0039086, 0.0042961, -0.0044990, 0.0051752, -0.0090838, 0.0087951
1: 0.9923025, 1.0047761, 0.9922403, 1.0059484, -0.0136459, 0.0125359
2: -0.0204253, 0.0089830, -0.0210804, 0.0098160, -0.0292919, 0.0291541
3: -0.0006258, 0.0049318, -0.0008631, 0.0049752, -0.0056010, 0.0057950
4: -0.0102247, 0.0145600, -0.0112548, 0.0150778, -0.0253025, 0.0258148
5: -0.0029147, 0.0139010, -0.0033641, 0.0152463, -0.0181609, 0.0172651
6: -0.0062109, 0.0065222, -0.0064383, 0.0069378, -0.0131487, 0.0129605
7: -0.0118755, -0.0019183, -0.0122824, -0.0014900, -0.0103856, 0.0103641
8: -0.0116336, 0.0250879, -0.0122660, 0.0259486, -0.0373223, 0.0370879
9: -0.0093576, 0.0059449, -0.0107845, 0.0062960, -0.0156537, 0.0167294

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151146, upper bound: 0.0151410
time: 3.07 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151790, upper bound: 0.0152004
time: 3.09 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0051391, 0.0061281, -0.0041342, 0.0046319, -0.0097710, 0.0102623
1: 0.9923009, 1.0072194, 0.9923011, 1.0052241, -0.0129232, 0.0149183
2: -0.0204427, 0.0107190, -0.0204409, 0.0093012, -0.0287754, 0.0303368
3: -0.0006322, 0.0049330, -0.0006315, 0.0049329, -0.0055651, 0.0055645
4: -0.0123715, 0.0145738, -0.0106182, 0.0145723, -0.0269438, 0.0251920
5: -0.0029266, 0.0167046, -0.0029254, 0.0144149, -0.0173415, 0.0196300
6: -0.0062169, 0.0065333, -0.0062163, 0.0065321, -0.0127491, 0.0127496
7: -0.0127234, -0.0010257, -0.0120310, -0.0017547, -0.0109687, 0.0110053
8: -0.0129515, 0.0251108, -0.0118752, 0.0251084, -0.0378237, 0.0367156
9: -0.0123313, 0.0066767, -0.0099027, 0.0060790, -0.0184103, 0.0165794

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 196

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151935, upper bound: 0.0151798
time: 3.08 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152058, upper bound: 0.0151991
time: 2.98 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0051825, 0.0061927, -0.0045167, 0.0052014, -0.0103839, 0.0107094
1: 0.9922747, 1.0073055, 0.9922357, 1.0059834, -0.0137087, 0.0150698
2: -0.0207180, 0.0107802, -0.0211293, 0.0098409, -0.0296075, 0.0310120
3: -0.0007319, 0.0049512, -0.0008808, 0.0049785, -0.0057103, 0.0058321
4: -0.0124472, 0.0147914, -0.0112856, 0.0151164, -0.0275636, 0.0260770
5: -0.0031155, 0.0168035, -0.0033977, 0.0152865, -0.0184019, 0.0202011
6: -0.0063125, 0.0067079, -0.0064552, 0.0069689, -0.0132813, 0.0131632
7: -0.0127532, -0.0009942, -0.0122945, -0.0014772, -0.0112761, 0.0113003
8: -0.0129980, 0.0254725, -0.0122849, 0.0260130, -0.0387534, 0.0374927
9: -0.0124361, 0.0067025, -0.0108271, 0.0063065, -0.0187427, 0.0175296

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 173

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151927, upper bound: 0.0151857
time: 2.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0152056, upper bound: 0.0152056
time: 2.97 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.75 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150904
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151546
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150904
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0151546
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150904
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151545
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150904
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151545
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151306
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151863
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151403
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151501, upper bound: 0.0151961
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151621, upper bound: 0.0151676
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151763, upper bound: 0.0151893
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151618, upper bound: 0.0151736
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151762, upper bound: 0.0151971
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151207, upper bound: 0.0151173
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151804, upper bound: 0.0151774
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0150996, upper bound: 0.0151264
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151796, upper bound: 0.0151874
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0148178, upper bound: 0.0147911
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151925, upper bound: 0.0151694
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0148168, upper bound: 0.0147953
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151923, upper bound: 0.0151756
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151148, upper bound: 0.0151310
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151800, upper bound: 0.0151901
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151146, upper bound: 0.0151410
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151790, upper bound: 0.0152004
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151935, upper bound: 0.0151798
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0152058, upper bound: 0.0151991
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0151927, upper bound: 0.0151857
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.75
Output dim: 1, lower bound: -0.0152056, upper bound: 0.0152056

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0022773, 0.0018674, -0.0033250, 0.0034273, -0.0057046, 0.0051924
1: 0.9924330, 1.0015370, 0.9923454, 1.0036175, -0.0111846, 0.0091916
2: -0.0190525, 0.0066814, -0.0199748, 0.0081596, -0.0263615, 0.0257886
3: -0.0001286, 0.0048410, -0.0004627, 0.0049020, -0.0050306, 0.0053036
4: -0.0073786, 0.0134751, -0.0092065, 0.0142040, -0.0215827, 0.0226816
5: -0.0019729, 0.0101841, -0.0026056, 0.0125713, -0.0145442, 0.0127897
6: -0.0057344, 0.0056514, -0.0060545, 0.0062365, -0.0119708, 0.0117059
7: -0.0107516, -0.0031016, -0.0114735, -0.0023416, -0.0084100, 0.0083719
8: -0.0098865, 0.0232843, -0.0110086, 0.0244961, -0.0341367, 0.0340493
9: -0.0054154, 0.0049746, -0.0079473, 0.0055978, -0.0110132, 0.0129220

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150768
time: 2.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150904
time: 2.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0025804, 0.0023187, -0.0034818, 0.0036607, -0.0062411, 0.0058005
1: 0.9924417, 1.0021390, 0.9923422, 1.0039287, -0.0114870, 0.0097968
2: -0.0189613, 0.0071090, -0.0200073, 0.0083808, -0.0265131, 0.0262218
3: -0.0000956, 0.0048349, -0.0004745, 0.0049042, -0.0049998, 0.0053094
4: -0.0079074, 0.0134030, -0.0094801, 0.0142297, -0.0221372, 0.0228831
5: -0.0019103, 0.0108747, -0.0026279, 0.0129285, -0.0148388, 0.0135026
6: -0.0057027, 0.0055935, -0.0060658, 0.0062571, -0.0119598, 0.0116593
7: -0.0109604, -0.0028817, -0.0115815, -0.0022278, -0.0087326, 0.0086998
8: -0.0102111, 0.0231644, -0.0111765, 0.0245388, -0.0344987, 0.0341059
9: -0.0061479, 0.0051549, -0.0083263, 0.0056910, -0.0118389, 0.0134812

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151385
time: 3.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151546
time: 3.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0022903, 0.0018867, -0.0045930, 0.0053150, -0.0076053, 0.0064797
1: 0.9924297, 1.0015628, 0.9923220, 1.0061350, -0.0137053, 0.0092409
2: -0.0190866, 0.0066996, -0.0202206, 0.0099485, -0.0281817, 0.0260431
3: -0.0001410, 0.0048432, -0.0005517, 0.0049183, -0.0050592, 0.0053949
4: -0.0074012, 0.0135020, -0.0114187, 0.0143982, -0.0217995, 0.0249207
5: -0.0019962, 0.0102136, -0.0027742, 0.0154603, -0.0174566, 0.0129878
6: -0.0057462, 0.0056730, -0.0061398, 0.0063924, -0.0121386, 0.0118128
7: -0.0107605, -0.0030922, -0.0123471, -0.0014218, -0.0093387, 0.0092549
8: -0.0099004, 0.0233290, -0.0123666, 0.0248190, -0.0344668, 0.0354503
9: -0.0054467, 0.0049824, -0.0110115, 0.0063519, -0.0117986, 0.0159938

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150756
time: 3.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150904
time: 2.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0025915, 0.0023353, -0.0047396, 0.0055334, -0.0081249, 0.0070749
1: 0.9924388, 1.0021610, 0.9923189, 1.0064263, -0.0139875, 0.0098421
2: -0.0189908, 0.0071247, -0.0202529, 0.0101554, -0.0283109, 0.0264723
3: -0.0001062, 0.0048369, -0.0005634, 0.0049204, -0.0050267, 0.0054003
4: -0.0079269, 0.0134263, -0.0116746, 0.0144238, -0.0223506, 0.0251009
5: -0.0019305, 0.0109001, -0.0027964, 0.0157945, -0.0177250, 0.0136965
6: -0.0057130, 0.0056122, -0.0061510, 0.0064129, -0.0121258, 0.0117632
7: -0.0109681, -0.0028736, -0.0124481, -0.0013154, -0.0096527, 0.0095745
8: -0.0102230, 0.0232032, -0.0125237, 0.0248614, -0.0348271, 0.0354863
9: -0.0061748, 0.0051615, -0.0113659, 0.0064391, -0.0126140, 0.0165275

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151379
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151546
time: 2.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0027549, 0.0025785, -0.0033681, 0.0034913, -0.0062463, 0.0059466
1: 0.9923592, 1.0024855, 0.9923165, 1.0037029, -0.0113437, 0.0101690
2: -0.0198301, 0.0073553, -0.0202779, 0.0082203, -0.0271041, 0.0267616
3: -0.0004102, 0.0048924, -0.0005725, 0.0049221, -0.0053323, 0.0054649
4: -0.0082119, 0.0140896, -0.0092816, 0.0144436, -0.0226555, 0.0233712
5: -0.0025063, 0.0112723, -0.0028136, 0.0126693, -0.0151756, 0.0140859
6: -0.0060043, 0.0061446, -0.0061597, 0.0064288, -0.0124330, 0.0123043
7: -0.0110807, -0.0027551, -0.0115031, -0.0023104, -0.0087703, 0.0087480
8: -0.0103980, 0.0243059, -0.0110547, 0.0248943, -0.0350458, 0.0350960
9: -0.0065696, 0.0052587, -0.0080513, 0.0056234, -0.0121930, 0.0133100

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150768
time: 2.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150904
time: 2.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0030208, 0.0029743, -0.0035246, 0.0037244, -0.0067452, 0.0064989
1: 0.9923700, 1.0030134, 0.9923136, 1.0040137, -0.0116436, 0.0106998
2: -0.0197148, 0.0077303, -0.0203096, 0.0084412, -0.0272263, 0.0271426
3: -0.0003685, 0.0048848, -0.0005839, 0.0049242, -0.0052927, 0.0054687
4: -0.0086757, 0.0139985, -0.0095548, 0.0144686, -0.0231443, 0.0235532
5: -0.0024272, 0.0118781, -0.0028353, 0.0130260, -0.0154533, 0.0147134
6: -0.0059643, 0.0060715, -0.0061707, 0.0064489, -0.0124131, 0.0122422
7: -0.0112638, -0.0025623, -0.0116110, -0.0021968, -0.0090670, 0.0090487
8: -0.0106828, 0.0241544, -0.0112224, 0.0249359, -0.0353645, 0.0351143
9: -0.0072121, 0.0054168, -0.0084297, 0.0057165, -0.0129286, 0.0138465

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151384
time: 2.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151545
time: 2.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0027696, 0.0026003, -0.0046381, 0.0053822, -0.0081518, 0.0072384
1: 0.9923566, 1.0025146, 0.9922939, 1.0062246, -0.0138680, 0.0102207
2: -0.0198556, 0.0073759, -0.0205156, 0.0100122, -0.0289296, 0.0270184
3: -0.0004195, 0.0048941, -0.0006586, 0.0049378, -0.0053573, 0.0055527
4: -0.0082375, 0.0141098, -0.0114974, 0.0146314, -0.0228688, 0.0256072
5: -0.0025238, 0.0113057, -0.0029766, 0.0155631, -0.0180869, 0.0142823
6: -0.0060131, 0.0061608, -0.0062422, 0.0065795, -0.0125926, 0.0124031
7: -0.0110908, -0.0027445, -0.0123781, -0.0013891, -0.0097016, 0.0096336
8: -0.0104137, 0.0243394, -0.0124149, 0.0252065, -0.0353709, 0.0364882
9: -0.0066050, 0.0052674, -0.0111205, 0.0063787, -0.0129838, 0.0163879

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150756
time: 2.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150904
time: 2.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0030336, 0.0029935, -0.0047851, 0.0056010, -0.0086346, 0.0077785
1: 0.9923682, 1.0030389, 0.9922909, 1.0065165, -0.0141482, 0.0107480
2: -0.0197340, 0.0077485, -0.0205476, 0.0102195, -0.0290278, 0.0273926
3: -0.0003755, 0.0048861, -0.0006701, 0.0049399, -0.0053154, 0.0055562
4: -0.0086982, 0.0140137, -0.0117538, 0.0146567, -0.0233549, 0.0257675
5: -0.0024404, 0.0119074, -0.0029986, 0.0158979, -0.0183383, 0.0149059
6: -0.0059709, 0.0060837, -0.0062533, 0.0065998, -0.0125708, 0.0123370
7: -0.0112727, -0.0025530, -0.0124794, -0.0012825, -0.0099902, 0.0099265
8: -0.0106965, 0.0241797, -0.0125723, 0.0252485, -0.0356881, 0.0364879
9: -0.0072432, 0.0054245, -0.0114756, 0.0064661, -0.0137093, 0.0169001

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151378
time: 2.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151545
time: 2.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0025337, 0.0022491, -0.0035444, 0.0037538, -0.0062875, 0.0057935
1: 0.9924122, 1.0020462, 0.9923538, 1.0040530, -0.0116408, 0.0096924
2: -0.0192709, 0.0070431, -0.0198856, 0.0084690, -0.0268941, 0.0260632
3: -0.0002077, 0.0048554, -0.0004304, 0.0048961, -0.0051038, 0.0052858
4: -0.0078259, 0.0136476, -0.0095892, 0.0141335, -0.0219594, 0.0232368
5: -0.0021227, 0.0107682, -0.0025444, 0.0130710, -0.0151937, 0.0133126
6: -0.0058102, 0.0057899, -0.0060235, 0.0061799, -0.0119900, 0.0118134
7: -0.0109282, -0.0029156, -0.0116246, -0.0021825, -0.0087457, 0.0087090
8: -0.0101610, 0.0235711, -0.0112435, 0.0243788, -0.0342956, 0.0345743
9: -0.0060349, 0.0051271, -0.0084774, 0.0057282, -0.0117632, 0.0136045

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0150967
time: 2.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151306
time: 2.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0028032, 0.0026504, -0.0037095, 0.0039996, -0.0068028, 0.0063599
1: 0.9924216, 1.0025814, 0.9923501, 1.0043808, -0.0119592, 0.0102313
2: -0.0191716, 0.0074234, -0.0199253, 0.0087020, -0.0270416, 0.0264581
3: -0.0001718, 0.0048489, -0.0004447, 0.0048987, -0.0050705, 0.0052936
4: -0.0082962, 0.0135692, -0.0098773, 0.0141649, -0.0224610, 0.0234465
5: -0.0020546, 0.0113824, -0.0025717, 0.0134472, -0.0155018, 0.0139540
6: -0.0057757, 0.0057269, -0.0060373, 0.0062051, -0.0119808, 0.0117643
7: -0.0111140, -0.0027201, -0.0117383, -0.0020627, -0.0090512, 0.0090182
8: -0.0104497, 0.0234408, -0.0114204, 0.0244310, -0.0346292, 0.0346244
9: -0.0066863, 0.0052874, -0.0088764, 0.0058264, -0.0125128, 0.0141638

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151504, upper bound: 0.0151523
time: 2.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151504, upper bound: 0.0151863
time: 2.38 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0025815, 0.0023203, -0.0039417, 0.0043454, -0.0069269, 0.0062620
1: 0.9923817, 1.0021411, 0.9922855, 1.0048420, -0.0124604, 0.0098556
2: -0.0195916, 0.0071105, -0.0206048, 0.0090296, -0.0277695, 0.0267527
3: -0.0003239, 0.0048767, -0.0006909, 0.0049437, -0.0052676, 0.0055676
4: -0.0079093, 0.0139012, -0.0102824, 0.0147020, -0.0226113, 0.0241836
5: -0.0023427, 0.0108771, -0.0030379, 0.0139764, -0.0163191, 0.0139150
6: -0.0059215, 0.0059934, -0.0062732, 0.0066362, -0.0125577, 0.0122666
7: -0.0109612, -0.0028809, -0.0118983, -0.0018943, -0.0090669, 0.0090174
8: -0.0102123, 0.0239926, -0.0116691, 0.0253239, -0.0352654, 0.0354219
9: -0.0061505, 0.0051556, -0.0094376, 0.0059646, -0.0121150, 0.0145932

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151074
time: 2.38 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151403
time: 2.65 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0028487, 0.0027181, -0.0040929, 0.0045705, -0.0074192, 0.0068110
1: 0.9923911, 1.0026716, 0.9922820, 1.0051421, -0.0127510, 0.0103896
2: -0.0194930, 0.0074876, -0.0206416, 0.0092429, -0.0278976, 0.0271409
3: -0.0002882, 0.0048701, -0.0007042, 0.0049462, -0.0052343, 0.0055743
4: -0.0083756, 0.0138232, -0.0105462, 0.0147310, -0.0231065, 0.0243694
5: -0.0022751, 0.0114860, -0.0030630, 0.0143208, -0.0165959, 0.0145490
6: -0.0058873, 0.0059308, -0.0062859, 0.0066595, -0.0125467, 0.0122167
7: -0.0111453, -0.0026871, -0.0120025, -0.0017846, -0.0093607, 0.0093154
8: -0.0104985, 0.0238630, -0.0118310, 0.0253721, -0.0355928, 0.0354577
9: -0.0067963, 0.0053145, -0.0098029, 0.0060545, -0.0128507, 0.0151174

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151501, upper bound: 0.0151640
time: 2.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151501, upper bound: 0.0151961
time: 2.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0037895, 0.0041188, -0.0035859, 0.0038157, -0.0076051, 0.0077047
1: 0.9923880, 1.0045397, 0.9923456, 1.0041355, -0.0117475, 0.0121941
2: -0.0195260, 0.0088149, -0.0199709, 0.0085276, -0.0271876, 0.0279169
3: -0.0003001, 0.0048723, -0.0004613, 0.0049018, -0.0052019, 0.0053336
4: -0.0100169, 0.0138493, -0.0096617, 0.0142009, -0.0242178, 0.0235109
5: -0.0022977, 0.0136296, -0.0026029, 0.0131656, -0.0154633, 0.0162325
6: -0.0058987, 0.0059517, -0.0060532, 0.0062340, -0.0121327, 0.0120049
7: -0.0117935, -0.0020047, -0.0116532, -0.0021524, -0.0096411, 0.0096485
8: -0.0115061, 0.0239063, -0.0112880, 0.0244909, -0.0357506, 0.0349461
9: -0.0090698, 0.0058740, -0.0085777, 0.0057529, -0.0148227, 0.0144518

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150648
time: 2.27 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151190, upper bound: 0.0151235
time: 2.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0041545, 0.0046623, -0.0038078, 0.0041460, -0.0083006, 0.0084700
1: 0.9923974, 1.0052644, 0.9923447, 1.0045761, -0.0121787, 0.0129197
2: -0.0194278, 0.0093299, -0.0199812, 0.0088407, -0.0274238, 0.0284385
3: -0.0002646, 0.0048658, -0.0004650, 0.0049024, -0.0051670, 0.0053308
4: -0.0106538, 0.0137717, -0.0100488, 0.0142091, -0.0248628, 0.0238205
5: -0.0022304, 0.0144613, -0.0026100, 0.0136713, -0.0159016, 0.0170713
6: -0.0058646, 0.0058895, -0.0060567, 0.0062405, -0.0121052, 0.0119462
7: -0.0120450, -0.0017399, -0.0118061, -0.0019914, -0.0100536, 0.0100662
8: -0.0118970, 0.0237774, -0.0115257, 0.0245045, -0.0361545, 0.0350595
9: -0.0099519, 0.0060911, -0.0091140, 0.0058849, -0.0158369, 0.0152052

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150892
time: 2.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150719, upper bound: 0.0151481
time: 2.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038382, 0.0041913, -0.0039794, 0.0044016, -0.0082397, 0.0081707
1: 0.9923590, 1.0046364, 0.9922780, 1.0049170, -0.0125580, 0.0123584
2: -0.0198316, 0.0088836, -0.0206831, 0.0090829, -0.0280508, 0.0286099
3: -0.0004108, 0.0048925, -0.0007192, 0.0049489, -0.0053597, 0.0056117
4: -0.0101018, 0.0140908, -0.0103483, 0.0147638, -0.0248656, 0.0244391
5: -0.0025073, 0.0137405, -0.0030915, 0.0140623, -0.0165697, 0.0168320
6: -0.0060048, 0.0061456, -0.0063004, 0.0066858, -0.0126906, 0.0124460
7: -0.0118270, -0.0019694, -0.0119243, -0.0018669, -0.0099601, 0.0099550
8: -0.0115582, 0.0243078, -0.0117095, 0.0254266, -0.0367148, 0.0357698
9: -0.0091874, 0.0059030, -0.0095288, 0.0059870, -0.0151744, 0.0154318

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150713
time: 2.41 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0151300
time: 2.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042028, 0.0047341, -0.0041992, 0.0047288, -0.0089316, 0.0089333
1: 0.9923685, 1.0053602, 0.9922771, 1.0053533, -0.0129848, 0.0130832
2: -0.0197308, 0.0093980, -0.0206936, 0.0093930, -0.0282812, 0.0291305
3: -0.0003743, 0.0048859, -0.0007230, 0.0049496, -0.0053239, 0.0056089
4: -0.0107379, 0.0140112, -0.0107318, 0.0147721, -0.0255100, 0.0247430
5: -0.0024383, 0.0145712, -0.0030987, 0.0145632, -0.0170014, 0.0176699
6: -0.0059698, 0.0060817, -0.0063040, 0.0066925, -0.0126623, 0.0123857
7: -0.0120782, -0.0017049, -0.0120758, -0.0017074, -0.0103708, 0.0103709
8: -0.0119487, 0.0241755, -0.0119449, 0.0254404, -0.0371194, 0.0358771
9: -0.0100685, 0.0061198, -0.0100600, 0.0061177, -0.0161862, 0.0161798

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150719, upper bound: 0.0150951
time: 2.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151345, upper bound: 0.0151536
time: 2.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0030033, 0.0029483, -0.0033038, 0.0033957, -0.0063990, 0.0062521
1: 0.9923235, 1.0029787, 0.9923224, 1.0035752, -0.0112517, 0.0106563
2: -0.0202052, 0.0077057, -0.0202155, 0.0081296, -0.0273545, 0.0270766
3: -0.0005461, 0.0049173, -0.0005498, 0.0049180, -0.0054641, 0.0054671
4: -0.0086453, 0.0143861, -0.0091695, 0.0143942, -0.0230395, 0.0235556
5: -0.0027637, 0.0118383, -0.0027707, 0.0125229, -0.0152866, 0.0146090
6: -0.0061345, 0.0063826, -0.0061380, 0.0063891, -0.0125236, 0.0125207
7: -0.0112518, -0.0025749, -0.0114588, -0.0023570, -0.0088948, 0.0088839
8: -0.0106641, 0.0247988, -0.0109859, 0.0248123, -0.0352379, 0.0355116
9: -0.0071699, 0.0054065, -0.0078960, 0.0055851, -0.0127551, 0.0133025

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151165, upper bound: 0.0150880
time: 2.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151165, upper bound: 0.0150942
time: 2.48 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032431, 0.0033053, -0.0034760, 0.0036521, -0.0068952, 0.0067814
1: 0.9923345, 1.0034549, 0.9923192, 1.0039173, -0.0115829, 0.0111357
2: -0.0200883, 0.0080440, -0.0202507, 0.0083727, -0.0275001, 0.0274280
3: -0.0005038, 0.0049095, -0.0005626, 0.0049203, -0.0054240, 0.0054721
4: -0.0090636, 0.0142937, -0.0094701, 0.0144220, -0.0234856, 0.0237637
5: -0.0026834, 0.0123846, -0.0027949, 0.0129154, -0.0155989, 0.0151795
6: -0.0060939, 0.0063085, -0.0061503, 0.0064115, -0.0125054, 0.0124587
7: -0.0114170, -0.0024010, -0.0115775, -0.0022321, -0.0091850, 0.0091765
8: -0.0109209, 0.0246451, -0.0111704, 0.0248585, -0.0355377, 0.0355463
9: -0.0077493, 0.0055491, -0.0083123, 0.0056876, -0.0134370, 0.0138614

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151705, upper bound: 0.0151421
time: 2.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151705, upper bound: 0.0151567
time: 2.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0030456, 0.0030112, -0.0036880, 0.0039677, -0.0070133, 0.0066992
1: 0.9922957, 1.0030625, 0.9922563, 1.0043381, -0.0120424, 0.0108061
2: -0.0204965, 0.0077653, -0.0209114, 0.0086717, -0.0281979, 0.0277437
3: -0.0006516, 0.0049365, -0.0008019, 0.0049640, -0.0056157, 0.0057385
4: -0.0087190, 0.0146163, -0.0098399, 0.0149442, -0.0236632, 0.0244562
5: -0.0029635, 0.0119346, -0.0032481, 0.0133983, -0.0163619, 0.0151827
6: -0.0062356, 0.0065674, -0.0063796, 0.0068306, -0.0130662, 0.0129470
7: -0.0112809, -0.0025443, -0.0117236, -0.0020783, -0.0092026, 0.0091792
8: -0.0107093, 0.0251815, -0.0113974, 0.0257265, -0.0361708, 0.0363092
9: -0.0072720, 0.0054316, -0.0088246, 0.0058137, -0.0130857, 0.0142561

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151164, upper bound: 0.0150976
time: 2.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151164, upper bound: 0.0151033
time: 2.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032844, 0.0033668, -0.0038469, 0.0042042, -0.0074886, 0.0072137
1: 0.9923085, 1.0035368, 0.9922533, 1.0046536, -0.0123451, 0.0112835
2: -0.0203622, 0.0081023, -0.0209435, 0.0088958, -0.0283109, 0.0280940
3: -0.0006030, 0.0049277, -0.0008135, 0.0049661, -0.0055691, 0.0057412
4: -0.0091357, 0.0145102, -0.0101170, 0.0149696, -0.0241053, 0.0246272
5: -0.0028714, 0.0124788, -0.0032702, 0.0137603, -0.0166317, 0.0157490
6: -0.0061890, 0.0064823, -0.0063908, 0.0068510, -0.0130400, 0.0128730
7: -0.0114455, -0.0023710, -0.0118330, -0.0019631, -0.0094824, 0.0094620
8: -0.0109651, 0.0250051, -0.0115675, 0.0257688, -0.0364653, 0.0363070
9: -0.0078492, 0.0055736, -0.0092084, 0.0059082, -0.0137574, 0.0147821

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151164, upper bound: 0.0151551
time: 2.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151702, upper bound: 0.0151680
time: 2.88 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0085039, 0.0057600, -0.0037155, 0.0040087, -0.0125126, 0.0094755
1: 0.9900883, 1.0067285, 0.9923117, 1.0043929, -0.0143046, 0.0144168
2: -0.0205389, 0.0117712, -0.0203286, 0.0087105, -0.0283349, 0.0316026
3: -0.0032191, 0.0049333, -0.0005908, 0.0049254, -0.0081446, 0.0055241
4: -0.0124351, 0.0146392, -0.0098879, 0.0144836, -0.0269187, 0.0245270
5: -0.0030443, 0.0188116, -0.0028483, 0.0134611, -0.0165053, 0.0216599
6: -0.0062904, 0.0069809, -0.0061773, 0.0064609, -0.0127513, 0.0131582
7: -0.0125530, 0.0068159, -0.0117425, -0.0020583, -0.0104947, 0.0185584
8: -0.0126867, 0.0257478, -0.0114269, 0.0249609, -0.0374069, 0.0369006
9: -0.0128232, 0.0065297, -0.0088911, 0.0058300, -0.0186532, 0.0154208

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150885, upper bound: 0.0150748
time: 2.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151500, upper bound: 0.0151280
time: 2.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0086482, 0.0058242, -0.0040933, 0.0045712, -0.0132194, 0.0099175
1: 0.9899952, 1.0068140, 0.9922459, 1.0051429, -0.0151477, 0.0145681
2: -0.0208165, 0.0118700, -0.0210211, 0.0092436, -0.0291668, 0.0323300
3: -0.0033395, 0.0049515, -0.0008416, 0.0049713, -0.0083108, 0.0057931
4: -0.0125268, 0.0148580, -0.0105470, 0.0150309, -0.0275577, 0.0254050
5: -0.0032353, 0.0189782, -0.0033234, 0.0143219, -0.0175572, 0.0223016
6: -0.0063882, 0.0071654, -0.0064177, 0.0069002, -0.0132884, 0.0135831
7: -0.0125827, 0.0070616, -0.0120028, -0.0017843, -0.0107984, 0.0190644
8: -0.0127329, 0.0261308, -0.0118315, 0.0258707, -0.0383406, 0.0376928
9: -0.0129547, 0.0065553, -0.0098041, 0.0060547, -0.0190095, 0.0163594

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150884, upper bound: 0.0150795
time: 2.50 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151495, upper bound: 0.0151338
time: 2.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0029801, 0.0029137, -0.0037310, 0.0040317, -0.0070117, 0.0066447
1: 0.9923248, 1.0029324, 0.9923123, 1.0044236, -0.0120988, 0.0106202
2: -0.0201905, 0.0076729, -0.0203232, 0.0087324, -0.0279451, 0.0271564
3: -0.0005408, 0.0049163, -0.0005889, 0.0049251, -0.0054659, 0.0055052
4: -0.0086047, 0.0143745, -0.0099148, 0.0144793, -0.0230840, 0.0242893
5: -0.0027536, 0.0117853, -0.0028446, 0.0134963, -0.0162499, 0.0146299
6: -0.0061294, 0.0063733, -0.0061754, 0.0064575, -0.0125869, 0.0125487
7: -0.0112358, -0.0025918, -0.0117532, -0.0020471, -0.0091887, 0.0091614
8: -0.0106391, 0.0247795, -0.0114434, 0.0249538, -0.0353541, 0.0359519
9: -0.0071137, 0.0053926, -0.0089284, 0.0058392, -0.0129530, 0.0143210

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0150967
time: 2.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151044
time: 4.01 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0032204, 0.0032715, -0.0038962, 0.0042776, -0.0074980, 0.0071677
1: 0.9923357, 1.0034095, 0.9923083, 1.0047514, -0.0124157, 0.0111012
2: -0.0200763, 0.0080119, -0.0203638, 0.0089654, -0.0280816, 0.0275120
3: -0.0004994, 0.0049087, -0.0006036, 0.0049278, -0.0054272, 0.0055124
4: -0.0090240, 0.0142842, -0.0102030, 0.0145115, -0.0235355, 0.0244873
5: -0.0026753, 0.0123328, -0.0028725, 0.0138727, -0.0165479, 0.0152054
6: -0.0060898, 0.0063009, -0.0061896, 0.0064833, -0.0125730, 0.0124904
7: -0.0114014, -0.0024175, -0.0118670, -0.0019273, -0.0094741, 0.0094495
8: -0.0108965, 0.0246294, -0.0116203, 0.0250072, -0.0356609, 0.0359812
9: -0.0076944, 0.0055355, -0.0093276, 0.0059375, -0.0136319, 0.0148632

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151523
time: 2.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151699, upper bound: 0.0151690
time: 3.03 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0030224, 0.0029766, -0.0041206, 0.0046117, -0.0076341, 0.0070972
1: 0.9922971, 1.0030166, 0.9922464, 1.0051970, -0.0129000, 0.0107702
2: -0.0204825, 0.0077325, -0.0210163, 0.0092820, -0.0287930, 0.0278195
3: -0.0006466, 0.0049356, -0.0008399, 0.0049710, -0.0056175, 0.0057755
4: -0.0086785, 0.0146052, -0.0105945, 0.0150271, -0.0237056, 0.0251998
5: -0.0029539, 0.0118817, -0.0033201, 0.0143840, -0.0173379, 0.0152018
6: -0.0062307, 0.0065585, -0.0064160, 0.0068972, -0.0131279, 0.0129745
7: -0.0112649, -0.0025611, -0.0120216, -0.0017645, -0.0095004, 0.0094605
8: -0.0106844, 0.0251631, -0.0118607, 0.0258644, -0.0362847, 0.0367535
9: -0.0072159, 0.0054178, -0.0098699, 0.0060710, -0.0132869, 0.0152877

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151074
time: 2.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151074
time: 2.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032620, 0.0033334, -0.0042704, 0.0048347, -0.0080967, 0.0076038
1: 0.9923096, 1.0034924, 0.9922429, 1.0054945, -0.0131848, 0.0112495
2: -0.0203511, 0.0080706, -0.0210537, 0.0094933, -0.0288954, 0.0281744
3: -0.0005990, 0.0049269, -0.0008535, 0.0049734, -0.0055724, 0.0057804
4: -0.0090966, 0.0145014, -0.0108558, 0.0150567, -0.0241533, 0.0253572
5: -0.0028637, 0.0124276, -0.0033458, 0.0147252, -0.0175889, 0.0157735
6: -0.0061851, 0.0064752, -0.0064290, 0.0069209, -0.0131060, 0.0129042
7: -0.0114300, -0.0023873, -0.0121248, -0.0016559, -0.0097741, 0.0097375
8: -0.0109411, 0.0249904, -0.0120211, 0.0259136, -0.0365861, 0.0367460
9: -0.0077950, 0.0055603, -0.0102318, 0.0061600, -0.0139550, 0.0157921

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151688, upper bound: 0.0151641
time: 2.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151688, upper bound: 0.0151784
time: 2.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0043384, 0.0049360, -0.0037778, 0.0041014, -0.0084398, 0.0087138
1: 0.9922944, 1.0056298, 0.9923028, 1.0045164, -0.0122220, 0.0133270
2: -0.0205119, 0.0095894, -0.0204226, 0.0087984, -0.0283275, 0.0291634
3: -0.0006572, 0.0049376, -0.0006249, 0.0049317, -0.0055889, 0.0055625
4: -0.0109746, 0.0146285, -0.0099964, 0.0145579, -0.0255325, 0.0246249
5: -0.0029741, 0.0148803, -0.0029128, 0.0136029, -0.0165770, 0.0177931
6: -0.0062409, 0.0065772, -0.0062100, 0.0065205, -0.0127615, 0.0127871
7: -0.0121717, -0.0016065, -0.0117854, -0.0020132, -0.0101585, 0.0101789
8: -0.0120940, 0.0252017, -0.0114935, 0.0250844, -0.0369365, 0.0364200
9: -0.0103963, 0.0062005, -0.0090415, 0.0058671, -0.0162634, 0.0152420

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150801, upper bound: 0.0150730
time: 2.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151514, upper bound: 0.0151367
time: 2.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0047116, 0.0054916, -0.0039996, 0.0044316, -0.0091433, 0.0094913
1: 0.9923033, 1.0063705, 0.9923018, 1.0049570, -0.0126536, 0.0140688
2: -0.0204171, 0.0101159, -0.0204330, 0.0091114, -0.0285575, 0.0296912
3: -0.0006229, 0.0049313, -0.0006287, 0.0049323, -0.0055552, 0.0055600
4: -0.0116257, 0.0145536, -0.0103835, 0.0145661, -0.0261918, 0.0249371
5: -0.0029091, 0.0157306, -0.0029200, 0.0141083, -0.0170174, 0.0186506
6: -0.0062080, 0.0065170, -0.0062136, 0.0065271, -0.0127352, 0.0127306
7: -0.0124288, -0.0013358, -0.0119383, -0.0018522, -0.0105766, 0.0106025
8: -0.0124937, 0.0250772, -0.0117311, 0.0250981, -0.0373472, 0.0365378
9: -0.0112982, 0.0064225, -0.0095776, 0.0059990, -0.0172972, 0.0160001

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150719, upper bound: 0.0150940
time: 2.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151647, upper bound: 0.0151583
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0043811, 0.0049996, -0.0041619, 0.0046732, -0.0090543, 0.0091615
1: 0.9922683, 1.0057143, 0.9922374, 1.0052792, -0.0130109, 0.0134769
2: -0.0207854, 0.0096495, -0.0211108, 0.0093403, -0.0291591, 0.0298366
3: -0.0007563, 0.0049557, -0.0008741, 0.0049772, -0.0057335, 0.0058298
4: -0.0110490, 0.0148446, -0.0106666, 0.0151019, -0.0261508, 0.0255112
5: -0.0031617, 0.0149775, -0.0033850, 0.0144781, -0.0176398, 0.0183624
6: -0.0063359, 0.0067507, -0.0064488, 0.0069572, -0.0132930, 0.0131995
7: -0.0122011, -0.0015756, -0.0120501, -0.0017346, -0.0104665, 0.0104745
8: -0.0121397, 0.0255611, -0.0119049, 0.0259886, -0.0378653, 0.0371932
9: -0.0104994, 0.0062259, -0.0099697, 0.0060955, -0.0165949, 0.0161956

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150789
time: 2.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151507, upper bound: 0.0151430
time: 2.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0047547, 0.0055558, -0.0043805, 0.0049987, -0.0097534, 0.0099363
1: 0.9922771, 1.0064563, 0.9922364, 1.0057132, -0.0134361, 0.0142199
2: -0.0206923, 0.0101767, -0.0211213, 0.0096487, -0.0293885, 0.0303642
3: -0.0007226, 0.0049495, -0.0008780, 0.0049779, -0.0057005, 0.0058275
4: -0.0117009, 0.0147711, -0.0110480, 0.0151101, -0.0268110, 0.0258191
5: -0.0030979, 0.0158288, -0.0033922, 0.0149762, -0.0180741, 0.0192210
6: -0.0063036, 0.0066917, -0.0064525, 0.0069638, -0.0132674, 0.0131441
7: -0.0124585, -0.0013045, -0.0122007, -0.0015760, -0.0108825, 0.0108962
8: -0.0125398, 0.0254388, -0.0121391, 0.0260024, -0.0382771, 0.0373131
9: -0.0114024, 0.0064481, -0.0104980, 0.0062255, -0.0176279, 0.0169462

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150944, upper bound: 0.0150999
time: 2.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151639, upper bound: 0.0151639
time: 2.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.04 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150768
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150904
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151385
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151546
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150756
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150904
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151379
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151546
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150768
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150904
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151384
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151545
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150756
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150904
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151378
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151545
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0150967
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151306
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151504, upper bound: 0.0151523
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151504, upper bound: 0.0151863
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151074
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150937, upper bound: 0.0151403
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151501, upper bound: 0.0151640
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151501, upper bound: 0.0151961
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150648
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151190, upper bound: 0.0151235
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150892
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150719, upper bound: 0.0151481
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150713
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0151300
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150719, upper bound: 0.0150951
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151345, upper bound: 0.0151536
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151165, upper bound: 0.0150880
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151165, upper bound: 0.0150942
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151705, upper bound: 0.0151421
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151705, upper bound: 0.0151567
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151164, upper bound: 0.0150976
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151164, upper bound: 0.0151033
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151164, upper bound: 0.0151551
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151702, upper bound: 0.0151680
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150885, upper bound: 0.0150748
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151500, upper bound: 0.0151280
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150884, upper bound: 0.0150795
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151495, upper bound: 0.0151338
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0150967
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151044
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151523
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151699, upper bound: 0.0151690
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151074
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151097, upper bound: 0.0151074
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151688, upper bound: 0.0151641
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151688, upper bound: 0.0151784
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150801, upper bound: 0.0150730
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151514, upper bound: 0.0151367
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150719, upper bound: 0.0150940
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151647, upper bound: 0.0151583
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150569, upper bound: 0.0150789
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151507, upper bound: 0.0151430
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0150944, upper bound: 0.0150999
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.04
Output dim: 1, lower bound: -0.0151639, upper bound: 0.0151639

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0022773, 0.0018674, -0.0028958, 0.0027883, -0.0050655, 0.0047632
1: 0.9924330, 1.0015370, 0.9924232, 1.0027651, -0.0103321, 0.0091137
2: -0.0190525, 0.0066814, -0.0191543, 0.0075540, -0.0257327, 0.0249753
3: -0.0001286, 0.0048410, -0.0001655, 0.0048477, -0.0049763, 0.0050065
4: -0.0073786, 0.0134751, -0.0084577, 0.0135555, -0.0209342, 0.0219328
5: -0.0019729, 0.0101841, -0.0020427, 0.0115933, -0.0135662, 0.0122268
6: -0.0057344, 0.0056514, -0.0057697, 0.0057160, -0.0114503, 0.0114211
7: -0.0107516, -0.0031016, -0.0111777, -0.0026529, -0.0080987, 0.0080761
8: -0.0098865, 0.0232843, -0.0105489, 0.0234180, -0.0330603, 0.0335839
9: -0.0054154, 0.0049746, -0.0069101, 0.0053425, -0.0107579, 0.0118848

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150768
time: 2.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150768
time: 2.56 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0022773, 0.0018674, -0.0033421, 0.0034527, -0.0057300, 0.0052095
1: 0.9924330, 1.0015370, 0.9923375, 1.0036513, -0.0112183, 0.0091994
2: -0.0190525, 0.0066814, -0.0200577, 0.0081837, -0.0263852, 0.0258724
3: -0.0001286, 0.0048410, -0.0004927, 0.0049075, -0.0050361, 0.0053337
4: -0.0073786, 0.0134751, -0.0092363, 0.0142695, -0.0216481, 0.0227114
5: -0.0019729, 0.0101841, -0.0026624, 0.0126102, -0.0145831, 0.0128465
6: -0.0057344, 0.0056514, -0.0060833, 0.0062890, -0.0120234, 0.0117346
7: -0.0107516, -0.0031016, -0.0114852, -0.0023292, -0.0084224, 0.0083836
8: -0.0098865, 0.0232843, -0.0110269, 0.0246049, -0.0342469, 0.0340675
9: -0.0054154, 0.0049746, -0.0079886, 0.0056080, -0.0110234, 0.0129633

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150904
time: 2.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151040, upper bound: 0.0150904
time: 2.46 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0025804, 0.0023187, -0.0030669, 0.0030430, -0.0056234, 0.0053856
1: 0.9924417, 1.0021390, 0.9924202, 1.0031048, -0.0106632, 0.0097188
2: -0.0189613, 0.0071090, -0.0191866, 0.0077954, -0.0259065, 0.0254084
3: -0.0000956, 0.0048349, -0.0001772, 0.0048498, -0.0049454, 0.0050121
4: -0.0079074, 0.0134030, -0.0087562, 0.0135810, -0.0214885, 0.0221592
5: -0.0019103, 0.0108747, -0.0020649, 0.0119832, -0.0138935, 0.0129395
6: -0.0057027, 0.0055935, -0.0057809, 0.0057364, -0.0114392, 0.0113744
7: -0.0109604, -0.0028817, -0.0112956, -0.0025288, -0.0084316, 0.0084139
8: -0.0102111, 0.0231644, -0.0107321, 0.0234605, -0.0334219, 0.0336548
9: -0.0061479, 0.0051549, -0.0073236, 0.0054443, -0.0115921, 0.0124785

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151385
time: 3.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151385
time: 2.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0025804, 0.0023187, -0.0035000, 0.0036878, -0.0062681, 0.0058186
1: 0.9924417, 1.0021390, 0.9923342, 1.0039648, -0.0115231, 0.0098047
2: -0.0189613, 0.0071090, -0.0200910, 0.0084064, -0.0265381, 0.0263083
3: -0.0000956, 0.0048349, -0.0005048, 0.0049097, -0.0050053, 0.0053397
4: -0.0079074, 0.0134030, -0.0095118, 0.0142958, -0.0222032, 0.0229147
5: -0.0019103, 0.0108747, -0.0026854, 0.0129699, -0.0148802, 0.0135600
6: -0.0057027, 0.0055935, -0.0060948, 0.0063102, -0.0120129, 0.0116883
7: -0.0109604, -0.0028817, -0.0115940, -0.0022147, -0.0087457, 0.0087123
8: -0.0102111, 0.0231644, -0.0111960, 0.0246487, -0.0346103, 0.0341252
9: -0.0061479, 0.0051549, -0.0083701, 0.0057018, -0.0118497, 0.0135250

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151546
time: 3.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151527, upper bound: 0.0151546
time: 3.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0022903, 0.0018867, -0.0040501, 0.0045068, -0.0067971, 0.0059368
1: 0.9924297, 1.0015628, 0.9924036, 1.0050572, -0.0126275, 0.0091592
2: -0.0190866, 0.0066996, -0.0193614, 0.0091826, -0.0273961, 0.0251876
3: -0.0001410, 0.0048432, -0.0002405, 0.0048614, -0.0050024, 0.0050838
4: -0.0074012, 0.0135020, -0.0104716, 0.0137192, -0.0211204, 0.0239736
5: -0.0019962, 0.0102136, -0.0021848, 0.0142234, -0.0162196, 0.0123984
6: -0.0057462, 0.0056730, -0.0058416, 0.0058473, -0.0115935, 0.0115146
7: -0.0107605, -0.0030922, -0.0119730, -0.0018156, -0.0089449, 0.0088808
8: -0.0099004, 0.0233290, -0.0117852, 0.0236901, -0.0333381, 0.0348650
9: -0.0054467, 0.0049824, -0.0096996, 0.0060291, -0.0114758, 0.0146820

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150756
time: 2.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150756
time: 2.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0022903, 0.0018867, -0.0046154, 0.0053484, -0.0076387, 0.0065021
1: 0.9924297, 1.0015628, 0.9923098, 1.0061796, -0.0137498, 0.0092530
2: -0.0190866, 0.0066996, -0.0203482, 0.0099801, -0.0282124, 0.0261759
3: -0.0001410, 0.0048432, -0.0005980, 0.0049267, -0.0050677, 0.0054412
4: -0.0074012, 0.0135020, -0.0114578, 0.0144991, -0.0219003, 0.0249598
5: -0.0019962, 0.0102136, -0.0028618, 0.0155114, -0.0175076, 0.0130754
6: -0.0057462, 0.0056730, -0.0061841, 0.0064734, -0.0122196, 0.0118571
7: -0.0107605, -0.0030922, -0.0123625, -0.0014056, -0.0093549, 0.0092703
8: -0.0099004, 0.0233290, -0.0123906, 0.0249867, -0.0346387, 0.0354742
9: -0.0054467, 0.0049824, -0.0110657, 0.0063652, -0.0118120, 0.0160480

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150904
time: 2.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150902, upper bound: 0.0150904
time: 2.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0025915, 0.0023353, -0.0042161, 0.0047539, -0.0073454, 0.0065513
1: 0.9924388, 1.0021610, 0.9924005, 1.0053867, -0.0129479, 0.0097605
2: -0.0189908, 0.0071247, -0.0193937, 0.0094167, -0.0275571, 0.0256177
3: -0.0001062, 0.0048369, -0.0002522, 0.0048635, -0.0049698, 0.0050891
4: -0.0079269, 0.0134263, -0.0107611, 0.0137447, -0.0216716, 0.0241874
5: -0.0019305, 0.0109001, -0.0022069, 0.0146015, -0.0165320, 0.0131070
6: -0.0057130, 0.0056122, -0.0058528, 0.0058678, -0.0115808, 0.0114650
7: -0.0109681, -0.0028736, -0.0120874, -0.0016952, -0.0092729, 0.0092138
8: -0.0102230, 0.0232032, -0.0119629, 0.0237326, -0.0336985, 0.0349230
9: -0.0061748, 0.0051615, -0.0101006, 0.0061277, -0.0123026, 0.0152622

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151379
time: 2.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151379
time: 2.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0025915, 0.0023353, -0.0047635, 0.0055689, -0.0081604, 0.0070987
1: 0.9924388, 1.0021610, 0.9923067, 1.0064737, -0.0140349, 0.0098543
2: -0.0189908, 0.0071247, -0.0203814, 0.0101891, -0.0283439, 0.0266135
3: -0.0001062, 0.0048369, -0.0006100, 0.0049289, -0.0050352, 0.0054469
4: -0.0079269, 0.0134263, -0.0117162, 0.0145254, -0.0224522, 0.0251424
5: -0.0019305, 0.0109001, -0.0028846, 0.0158488, -0.0177793, 0.0137846
6: -0.0057130, 0.0056122, -0.0061957, 0.0064944, -0.0122074, 0.0118079
7: -0.0109681, -0.0028736, -0.0124646, -0.0012982, -0.0096700, 0.0095909
8: -0.0102230, 0.0232032, -0.0125492, 0.0250303, -0.0350018, 0.0355118
9: -0.0061748, 0.0051615, -0.0114235, 0.0064533, -0.0126281, 0.0165851

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151546
time: 2.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151405, upper bound: 0.0151546
time: 2.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0027549, 0.0025785, -0.0029428, 0.0028582, -0.0056132, 0.0055213
1: 0.9923592, 1.0024855, 0.9923925, 1.0028584, -0.0104992, 0.0100930
2: -0.0198301, 0.0073553, -0.0194773, 0.0076203, -0.0264781, 0.0259642
3: -0.0004102, 0.0048924, -0.0002825, 0.0048691, -0.0052793, 0.0051749
4: -0.0082119, 0.0140896, -0.0085397, 0.0138108, -0.0220228, 0.0226293
5: -0.0025063, 0.0112723, -0.0022643, 0.0117004, -0.0142067, 0.0135366
6: -0.0060043, 0.0061446, -0.0058818, 0.0059209, -0.0119252, 0.0120265
7: -0.0110807, -0.0027551, -0.0112101, -0.0026188, -0.0084618, 0.0084550
8: -0.0103980, 0.0243059, -0.0105992, 0.0238425, -0.0339964, 0.0346341
9: -0.0065696, 0.0052587, -0.0070237, 0.0053705, -0.0119401, 0.0122824

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150768
time: 2.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150768
time: 2.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0027549, 0.0025785, -0.0033841, 0.0035152, -0.0062702, 0.0059626
1: 0.9923592, 1.0024855, 0.9923112, 1.0037348, -0.0113757, 0.0101743
2: -0.0198301, 0.0073553, -0.0203342, 0.0082429, -0.0271263, 0.0268276
3: -0.0004102, 0.0048924, -0.0005929, 0.0049258, -0.0053361, 0.0054853
4: -0.0082119, 0.0140896, -0.0093096, 0.0144881, -0.0227000, 0.0233992
5: -0.0025063, 0.0112723, -0.0028522, 0.0127059, -0.0152122, 0.0141245
6: -0.0060043, 0.0061446, -0.0061793, 0.0064645, -0.0124687, 0.0123239
7: -0.0110807, -0.0027551, -0.0115142, -0.0022987, -0.0087819, 0.0087590
8: -0.0103980, 0.0243059, -0.0110719, 0.0249683, -0.0351221, 0.0351131
9: -0.0065696, 0.0052587, -0.0080901, 0.0056329, -0.0122026, 0.0133488

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150904
time: 2.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151159, upper bound: 0.0150904
time: 2.92 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0030208, 0.0029743, -0.0031134, 0.0031122, -0.0061330, 0.0060877
1: 0.9923700, 1.0030134, 0.9923896, 1.0031971, -0.0108271, 0.0106238
2: -0.0197148, 0.0077303, -0.0195090, 0.0078610, -0.0266208, 0.0263456
3: -0.0003685, 0.0048848, -0.0002940, 0.0048712, -0.0052397, 0.0051788
4: -0.0086757, 0.0139985, -0.0088373, 0.0138359, -0.0225116, 0.0228358
5: -0.0024272, 0.0118781, -0.0022861, 0.0120891, -0.0145163, 0.0141642
6: -0.0059643, 0.0060715, -0.0058928, 0.0059410, -0.0119053, 0.0119644
7: -0.0112638, -0.0025623, -0.0113276, -0.0024951, -0.0087687, 0.0087654
8: -0.0106828, 0.0241544, -0.0107819, 0.0238841, -0.0343154, 0.0346664
9: -0.0072121, 0.0054168, -0.0074359, 0.0054719, -0.0126840, 0.0128527

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151384
time: 2.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151384
time: 2.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0030208, 0.0029743, -0.0035418, 0.0037501, -0.0067708, 0.0065161
1: 0.9923700, 1.0030134, 0.9923080, 1.0040479, -0.0116779, 0.0107054
2: -0.0197148, 0.0077303, -0.0203669, 0.0084655, -0.0272502, 0.0272125
3: -0.0003685, 0.0048848, -0.0006047, 0.0049280, -0.0052965, 0.0054895
4: -0.0086757, 0.0139985, -0.0095848, 0.0145139, -0.0231896, 0.0235833
5: -0.0024272, 0.0118781, -0.0028746, 0.0130653, -0.0154925, 0.0147526
6: -0.0059643, 0.0060715, -0.0061906, 0.0064852, -0.0124494, 0.0122621
7: -0.0112638, -0.0025623, -0.0116228, -0.0021843, -0.0090795, 0.0090606
8: -0.0106828, 0.0241544, -0.0112408, 0.0250112, -0.0354436, 0.0351325
9: -0.0072121, 0.0054168, -0.0084713, 0.0057267, -0.0129388, 0.0138881

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151545
time: 2.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151663, upper bound: 0.0151545
time: 2.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0027696, 0.0026003, -0.0041001, 0.0045812, -0.0073508, 0.0067004
1: 0.9923566, 1.0025146, 0.9923745, 1.0051564, -0.0127998, 0.0101401
2: -0.0198556, 0.0073759, -0.0196672, 0.0092531, -0.0281465, 0.0261736
3: -0.0004195, 0.0048941, -0.0003513, 0.0048816, -0.0053011, 0.0052454
4: -0.0082375, 0.0141098, -0.0105587, 0.0139609, -0.0221983, 0.0246685
5: -0.0025238, 0.0113057, -0.0023946, 0.0143372, -0.0168611, 0.0137003
6: -0.0060131, 0.0061608, -0.0059478, 0.0060413, -0.0120544, 0.0121086
7: -0.0110908, -0.0027445, -0.0120075, -0.0017794, -0.0093114, 0.0092630
8: -0.0104137, 0.0243394, -0.0118387, 0.0240919, -0.0342558, 0.0359073
9: -0.0066050, 0.0052674, -0.0098203, 0.0060588, -0.0126638, 0.0150878

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150756
time: 2.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150756
time: 2.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0027696, 0.0026003, -0.0046588, 0.0054130, -0.0081825, 0.0072591
1: 0.9923566, 1.0025146, 0.9922835, 1.0062658, -0.0139092, 0.0102311
2: -0.0198556, 0.0073759, -0.0206260, 0.0100413, -0.0289578, 0.0271440
3: -0.0004195, 0.0048941, -0.0006986, 0.0049451, -0.0053646, 0.0055927
4: -0.0082375, 0.0141098, -0.0115335, 0.0147186, -0.0229561, 0.0256433
5: -0.0025238, 0.0113057, -0.0030524, 0.0156102, -0.0181340, 0.0143581
6: -0.0060131, 0.0061608, -0.0062806, 0.0066496, -0.0126627, 0.0124414
7: -0.0110908, -0.0027445, -0.0123924, -0.0013741, -0.0097166, 0.0096479
8: -0.0104137, 0.0243394, -0.0124371, 0.0253516, -0.0355240, 0.0365102
9: -0.0066050, 0.0052674, -0.0111705, 0.0063910, -0.0129961, 0.0164379

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150904
time: 2.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150978, upper bound: 0.0150904
time: 2.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0030336, 0.0029935, -0.0042654, 0.0048273, -0.0078610, 0.0072589
1: 0.9923682, 1.0030389, 0.9923715, 1.0054848, -0.0131166, 0.0106674
2: -0.0197340, 0.0077485, -0.0196991, 0.0094863, -0.0282740, 0.0265515
3: -0.0003755, 0.0048861, -0.0003628, 0.0048838, -0.0052592, 0.0052489
4: -0.0086982, 0.0140137, -0.0108472, 0.0139861, -0.0226843, 0.0248609
5: -0.0024404, 0.0119074, -0.0024165, 0.0147139, -0.0171543, 0.0143238
6: -0.0059709, 0.0060837, -0.0059588, 0.0060615, -0.0120325, 0.0120425
7: -0.0112727, -0.0025530, -0.0121214, -0.0016595, -0.0096132, 0.0095684
8: -0.0106965, 0.0241797, -0.0120158, 0.0241338, -0.0345741, 0.0359276
9: -0.0072432, 0.0054245, -0.0102198, 0.0061571, -0.0134003, 0.0156443

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151378
time: 2.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151378
time: 2.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0030336, 0.0029935, -0.0048070, 0.0056336, -0.0086673, 0.0078005
1: 0.9923682, 1.0030389, 0.9922802, 1.0065600, -0.0141917, 0.0107586
2: -0.0197340, 0.0077485, -0.0206590, 0.0102504, -0.0290580, 0.0275283
3: -0.0003755, 0.0048861, -0.0007105, 0.0049473, -0.0053228, 0.0055966
4: -0.0086982, 0.0140137, -0.0117920, 0.0147447, -0.0234429, 0.0258057
5: -0.0024404, 0.0119074, -0.0030750, 0.0159479, -0.0183883, 0.0149824
6: -0.0059709, 0.0060837, -0.0062920, 0.0066705, -0.0126414, 0.0123757
7: -0.0112727, -0.0025530, -0.0124945, -0.0012666, -0.0100061, 0.0099415
8: -0.0106965, 0.0241797, -0.0125958, 0.0253949, -0.0358440, 0.0365111
9: -0.0072432, 0.0054245, -0.0115286, 0.0064792, -0.0137224, 0.0169531

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151545
time: 2.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151505, upper bound: 0.0151545
time: 2.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0025337, 0.0022491, -0.0030517, 0.0030204, -0.0055540, 0.0053008
1: 0.9924122, 1.0020462, 0.9924345, 1.0030746, -0.0106624, 0.0096117
2: -0.0192709, 0.0070431, -0.0190353, 0.0077740, -0.0261685, 0.0252270
3: -0.0002077, 0.0048554, -0.0001224, 0.0048398, -0.0050475, 0.0049778
4: -0.0078259, 0.0136476, -0.0087297, 0.0134615, -0.0212874, 0.0223773
5: -0.0021227, 0.0107682, -0.0019611, 0.0119486, -0.0140713, 0.0127292
6: -0.0058102, 0.0057899, -0.0057284, 0.0056405, -0.0114506, 0.0115183
7: -0.0109282, -0.0029156, -0.0112852, -0.0025398, -0.0083884, 0.0083695
8: -0.0101610, 0.0235711, -0.0107159, 0.0232617, -0.0331809, 0.0340394
9: -0.0060349, 0.0051271, -0.0072869, 0.0054352, -0.0114702, 0.0124140

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150907, upper bound: 0.0150967
time: 2.16 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150907, upper bound: 0.0150967
time: 2.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0025337, 0.0022491, -0.0035655, 0.0037853, -0.0063190, 0.0058146
1: 0.9924122, 1.0020462, 0.9923437, 1.0040950, -0.0116827, 0.0097026
2: -0.0192709, 0.0070431, -0.0199920, 0.0084989, -0.0269233, 0.0261758
3: -0.0002077, 0.0048554, -0.0004689, 0.0049032, -0.0051109, 0.0053243
4: -0.0078259, 0.0136476, -0.0096261, 0.0142176, -0.0220435, 0.0232737
5: -0.0021227, 0.0107682, -0.0026174, 0.0131192, -0.0152419, 0.0133856
6: -0.0058102, 0.0057899, -0.0060605, 0.0062474, -0.0120576, 0.0118504
7: -0.0109282, -0.0029156, -0.0116392, -0.0021671, -0.0087611, 0.0087235
8: -0.0101610, 0.0235711, -0.0112662, 0.0245187, -0.0344361, 0.0345968
9: -0.0060349, 0.0051271, -0.0085285, 0.0057408, -0.0117758, 0.0136556

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150907, upper bound: 0.0151306
time: 2.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150907, upper bound: 0.0151306
time: 2.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0028032, 0.0026504, -0.0032263, 0.0032802, -0.0060834, 0.0058767
1: 0.9924216, 1.0025814, 0.9924307, 1.0034214, -0.0109998, 0.0101506
2: -0.0191716, 0.0074234, -0.0190753, 0.0080202, -0.0263364, 0.0256204
3: -0.0001718, 0.0048489, -0.0001369, 0.0048425, -0.0050142, 0.0049857
4: -0.0082962, 0.0135692, -0.0090342, 0.0134931, -0.0217893, 0.0226034
5: -0.0020546, 0.0113824, -0.0019885, 0.0123463, -0.0144008, 0.0133709
6: -0.0057757, 0.0057269, -0.0057423, 0.0056658, -0.0114416, 0.0114692
7: -0.0111140, -0.0027201, -0.0114054, -0.0024132, -0.0087007, 0.0086853
8: -0.0104497, 0.0234408, -0.0109028, 0.0233142, -0.0335149, 0.0341025
9: -0.0066863, 0.0052874, -0.0077087, 0.0055391, -0.0122254, 0.0129961

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149139, upper bound: 0.0149532
time: 2.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148954, upper bound: 0.0148999
time: 2.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0028032, 0.0026504, -0.0037309, 0.0040316, -0.0068348, 0.0063813
1: 0.9924216, 1.0025814, 0.9923398, 1.0044233, -0.0120016, 0.0102415
2: -0.0191716, 0.0074234, -0.0200330, 0.0087322, -0.0270714, 0.0265730
3: -0.0001718, 0.0048489, -0.0004837, 0.0049059, -0.0050776, 0.0053326
4: -0.0082962, 0.0135692, -0.0099147, 0.0142499, -0.0225461, 0.0234839
5: -0.0020546, 0.0113824, -0.0026455, 0.0134961, -0.0155507, 0.0140279
6: -0.0057757, 0.0057269, -0.0060747, 0.0062734, -0.0120491, 0.0118016
7: -0.0111140, -0.0027201, -0.0117531, -0.0020472, -0.0090668, 0.0090330
8: -0.0104497, 0.0234408, -0.0114433, 0.0245724, -0.0347722, 0.0346472
9: -0.0066863, 0.0052874, -0.0089282, 0.0058392, -0.0125255, 0.0142157

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0149139, upper bound: 0.0149532
time: 2.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0148954, upper bound: 0.0149364
time: 2.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0025815, 0.0023203, -0.0035270, 0.0037279, -0.0063094, 0.0058473
1: 0.9923817, 1.0021411, 0.9923626, 1.0040184, -0.0116367, 0.0097786
2: -0.0195916, 0.0071105, -0.0197940, 0.0084445, -0.0271591, 0.0259472
3: -0.0003239, 0.0048767, -0.0003972, 0.0048900, -0.0052140, 0.0052739
4: -0.0079093, 0.0139012, -0.0095589, 0.0140611, -0.0219704, 0.0234601
5: -0.0023427, 0.0108771, -0.0024816, 0.0130314, -0.0153742, 0.0133587
6: -0.0059215, 0.0059934, -0.0059918, 0.0061218, -0.0120433, 0.0119852
7: -0.0109612, -0.0028809, -0.0116126, -0.0021951, -0.0087661, 0.0087317
8: -0.0102123, 0.0239926, -0.0112249, 0.0242585, -0.0341997, 0.0349724
9: -0.0061505, 0.0051556, -0.0084354, 0.0057179, -0.0118684, 0.0135909

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150882, upper bound: 0.0151074
time: 2.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150882, upper bound: 0.0150967
time: 2.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0025815, 0.0023203, -0.0039631, 0.0043773, -0.0069587, 0.0062834
1: 0.9923817, 1.0021411, 0.9922804, 1.0048844, -0.0125027, 0.0098608
2: -0.0195916, 0.0071105, -0.0206586, 0.0090599, -0.0277991, 0.0268298
3: -0.0003239, 0.0048767, -0.0007104, 0.0049473, -0.0052712, 0.0055870
4: -0.0079093, 0.0139012, -0.0103198, 0.0147444, -0.0226537, 0.0242210
5: -0.0023427, 0.0108771, -0.0030747, 0.0140252, -0.0163679, 0.0139519
6: -0.0059215, 0.0059934, -0.0062919, 0.0066703, -0.0125918, 0.0122852
7: -0.0109612, -0.0028809, -0.0119131, -0.0018787, -0.0090824, 0.0090322
8: -0.0102123, 0.0239926, -0.0116920, 0.0253945, -0.0353433, 0.0354447
9: -0.0061505, 0.0051556, -0.0094894, 0.0059773, -0.0121278, 0.0146449

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150882, upper bound: 0.0151403
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0150882, upper bound: 0.0151306
time: 2.54 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0028487, 0.0027181, -0.0036828, 0.0039599, -0.0068086, 0.0064009
1: 0.9923911, 1.0026716, 0.9923590, 1.0043277, -0.0119365, 0.0103126
2: -0.0194930, 0.0074876, -0.0198308, 0.0086643, -0.0272923, 0.0263350
3: -0.0002882, 0.0048701, -0.0004105, 0.0048925, -0.0051806, 0.0052806
4: -0.0083756, 0.0138232, -0.0098307, 0.0140902, -0.0224658, 0.0236539
5: -0.0022751, 0.0114860, -0.0025069, 0.0133864, -0.0156615, 0.0139929
6: -0.0058873, 0.0059308, -0.0060045, 0.0061451, -0.0120324, 0.0119353
7: -0.0111453, -0.0026871, -0.0117199, -0.0020821, -0.0090632, 0.0090329
8: -0.0104985, 0.0238630, -0.0113917, 0.0243069, -0.0345275, 0.0350125
9: -0.0067963, 0.0053145, -0.0088119, 0.0058105, -0.0126068, 0.0141264

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 221

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151416, upper bound: 0.0151641
time: 2.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0151416, upper bound: 0.0151523
time: 3.06 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.27 + 595.58 = 600.85 seconds
