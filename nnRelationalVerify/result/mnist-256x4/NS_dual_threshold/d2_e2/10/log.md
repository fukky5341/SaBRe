## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13558995000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793)
1: (-0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627)
2: (-0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203)
3: (-0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837)
4: (-0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710)
5: (-0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747)
6: (-0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562)
7: (-0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183)
8: (0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069)
9: (-0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.80 + 2.69 = 4.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1506555, upper bound: 0.1506555

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1443145, upper bound: 0.1330585
time: 2.03 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1495534, upper bound: 0.1495534
time: 1.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.53 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 3.53
Output dim: 8, lower bound: -0.1443145, upper bound: 0.1330585
NS_B2, status: Status.UNKNOWN, split count: 1, time: 3.53
Output dim: 8, lower bound: -0.1495534, upper bound: 0.1495534

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -0.0478219, 0.0511951, -0.0019119, 0.0277417, -0.0755637, 0.0531070
1: -0.0373440, 0.0367610, -0.0111849, 0.0254320, -0.0627760, 0.0479459
2: -0.0243897, 0.1169914, 0.0010098, 0.0422721, -0.0666618, 0.1159816
3: -0.0615586, 0.0470138, -0.0364435, 0.0234478, -0.0850064, 0.0834573
4: -0.0690749, 0.0184173, -0.0209180, 0.0126847, -0.0817596, 0.0393353
5: -0.0262166, 0.0311483, -0.0044232, 0.0238883, -0.0501049, 0.0355714
6: -0.0218710, 0.0639239, -0.0160163, 0.0386838, -0.0605549, 0.0799402
7: -0.0561894, 0.0327967, -0.0265311, 0.0099074, -0.0660967, 0.0593279
8: 0.8201787, 0.9919229, 0.9059174, 0.9909688, -0.1707901, 0.0860055
9: -0.0293109, 0.1344257, -0.0189111, 0.0356451, -0.0649560, 0.1533367

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1330874, upper bound: 0.1293841
time: 1.98 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1433198, upper bound: 0.1317660
time: 1.35 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -0.0649462, 0.0711331, -0.0545596, 0.0590740, -0.1240202, 0.1256926
1: -0.0514351, 0.0405276, -0.0416306, 0.0377633, -0.0891984, 0.0821582
2: -0.0369548, 0.1439655, -0.0291157, 0.1281760, -0.1651307, 0.1730812
3: -0.0710570, 0.0654267, -0.0648195, 0.0518410, -0.1228979, 0.1302463
4: -0.0885529, 0.0217181, -0.0767542, 0.0194023, -0.1079551, 0.0984723
5: -0.0391050, 0.0332697, -0.0305920, 0.0315645, -0.0706695, 0.0638617
6: -0.0304398, 0.0748164, -0.0251009, 0.0672517, -0.0976915, 0.0999173
7: -0.0706228, 0.0447955, -0.0620919, 0.0371661, -0.1077889, 0.1068874
8: 0.7881831, 0.9921900, 0.8074871, 0.9919812, -0.2037981, 0.1847029
9: -0.0352478, 0.1706189, -0.0311402, 0.1496067, -0.1848545, 0.2017591

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1330585, upper bound: 0.1443145
time: 1.86 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1330585, upper bound: 0.1495534
time: 1.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.41 seconds
NS_B1_A1, status: Status.VERIFIED, split count: 2, time: 5.41
Output dim: 8, lower bound: -0.1330874, upper bound: 0.1293841
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.41
Output dim: 8, lower bound: -0.1433198, upper bound: 0.1317660
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 5.41
Output dim: 8, lower bound: -0.1330585, upper bound: 0.1443145
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 5.41
Output dim: 8, lower bound: -0.1330585, upper bound: 0.1495534

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -0.0324517, 0.0387565, -0.0019119, 0.0277417, -0.0601935, 0.0406684
1: -0.0268791, 0.0314468, -0.0111849, 0.0254320, -0.0523111, 0.0426317
2: -0.0131081, 0.0897795, 0.0010098, 0.0422721, -0.0553801, 0.0887696
3: -0.0505909, 0.0377685, -0.0364435, 0.0234478, -0.0740387, 0.0742120
4: -0.0494978, 0.0160478, -0.0209180, 0.0126847, -0.0621825, 0.0369658
5: -0.0168407, 0.0281426, -0.0044232, 0.0238883, -0.0407290, 0.0325658
6: -0.0175819, 0.0541344, -0.0160163, 0.0386838, -0.0562658, 0.0701507
7: -0.0457369, 0.0232424, -0.0265311, 0.0099074, -0.0556443, 0.0497736
8: 0.8497119, 0.9915081, 0.9059174, 0.9909688, -0.1412569, 0.0855907
9: -0.0235396, 0.1020685, -0.0189111, 0.0356451, -0.0591847, 0.1209795

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1317660
time: 1.93 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1317660
time: 1.55 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -0.0019119, 0.0277417, -0.0545596, 0.0590740, -0.0609859, 0.0823013
1: -0.0111849, 0.0254320, -0.0416306, 0.0377633, -0.0489482, 0.0670626
2: 0.0010098, 0.0422721, -0.0291157, 0.1281760, -0.1271661, 0.0713878
3: -0.0364435, 0.0234478, -0.0648195, 0.0518410, -0.0882845, 0.0882673
4: -0.0209180, 0.0126847, -0.0767542, 0.0194023, -0.0403203, 0.0894389
5: -0.0044232, 0.0238883, -0.0305920, 0.0315645, -0.0359877, 0.0544803
6: -0.0160163, 0.0386838, -0.0251009, 0.0672517, -0.0832681, 0.0637847
7: -0.0265311, 0.0099074, -0.0620919, 0.0371661, -0.0636973, 0.0719993
8: 0.9059174, 0.9909688, 0.8074871, 0.9919812, -0.0860638, 0.1834817
9: -0.0189111, 0.0356451, -0.0311402, 0.1496067, -0.1685178, 0.0667854

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1293841, upper bound: 0.1330874
time: 1.46 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1433198
time: 1.55 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -0.0545596, 0.0590740, -0.0545596, 0.0590740, -0.1136335, 0.1136335
1: -0.0416306, 0.0377633, -0.0416306, 0.0377633, -0.0793939, 0.0793939
2: -0.0291157, 0.1281760, -0.0291157, 0.1281760, -0.1572917, 0.1572917
3: -0.0648195, 0.0518410, -0.0648195, 0.0518410, -0.1166605, 0.1166605
4: -0.0767542, 0.0194023, -0.0767542, 0.0194023, -0.0961565, 0.0961565
5: -0.0305920, 0.0315645, -0.0305920, 0.0315645, -0.0621565, 0.0621565
6: -0.0251009, 0.0672517, -0.0251009, 0.0672517, -0.0923526, 0.0923526
7: -0.0620919, 0.0371661, -0.0620919, 0.0371661, -0.0992580, 0.0992580
8: 0.8074871, 0.9919812, 0.8074871, 0.9919812, -0.1844941, 0.1844941
9: -0.0311402, 0.1496067, -0.0311402, 0.1496067, -0.1807469, 0.1807469

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1422156
time: 1.53 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1470386
time: 1.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.98 seconds
NS_B1_A2_A1, status: Status.VERIFIED, split count: 3, time: 4.98
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1317660
NS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 4.98
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1317660
NS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 4.98
Output dim: 8, lower bound: -0.1293841, upper bound: 0.1330874
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.98
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1433198
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.98
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1422156
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.98
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1470386

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0019119, 0.0277417, -0.0391726, 0.0412229, -0.0431348, 0.0669144
1: -0.0111849, 0.0254320, -0.0310785, 0.0320921, -0.0432770, 0.0565105
2: 0.0010098, 0.0422721, -0.0164209, 0.1010427, -0.1000329, 0.0586930
3: -0.0364435, 0.0234478, -0.0538648, 0.0401076, -0.0765511, 0.0773126
4: -0.0209180, 0.0126847, -0.0571271, 0.0163730, -0.0372910, 0.0698118
5: -0.0044232, 0.0238883, -0.0199711, 0.0285410, -0.0329642, 0.0438594
6: -0.0160163, 0.0386838, -0.0182337, 0.0574382, -0.0734546, 0.0569175
7: -0.0265311, 0.0099074, -0.0493294, 0.0271645, -0.0536957, 0.0592368
8: 0.9059174, 0.9909688, 0.8386118, 0.9915689, -0.0856515, 0.1523570
9: -0.0189111, 0.0356451, -0.0250701, 0.1156761, -0.1345871, 0.0607153

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1310405, upper bound: 0.1426717
time: 1.63 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1315190, upper bound: 0.1430544
time: 1.84 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0020333, 0.0230930, -0.0395627, 0.0426403, -0.0446736, 0.0626557
1: -0.0101848, 0.0206137, -0.0316853, 0.0334027, -0.0435875, 0.0522990
2: 0.0036332, 0.0388584, -0.0175493, 0.1022856, -0.0986525, 0.0564077
3: -0.0315598, 0.0190998, -0.0553271, 0.0409586, -0.0725184, 0.0744269
4: -0.0182569, 0.0110922, -0.0584040, 0.0168007, -0.0350576, 0.0694962
5: -0.0027434, 0.0206050, -0.0207076, 0.0292908, -0.0320342, 0.0413127
6: -0.0162188, 0.0352391, -0.0187003, 0.0584700, -0.0746889, 0.0539394
7: -0.0259530, 0.0089382, -0.0498008, 0.0274050, -0.0533580, 0.0587390
8: 0.9123200, 0.9906856, 0.8368430, 0.9916730, -0.0793530, 0.1538426
9: -0.0171296, 0.0297734, -0.0259267, 0.1163149, -0.1334445, 0.0557000

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1352364
time: 1.69 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1422156
time: 2.51 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0391726, 0.0412229, -0.0545596, 0.0590740, -0.0982466, 0.0957825
1: -0.0310785, 0.0320921, -0.0416306, 0.0377633, -0.0688418, 0.0737227
2: -0.0164209, 0.1010427, -0.0291157, 0.1281760, -0.1445969, 0.1301584
3: -0.0538648, 0.0401076, -0.0648195, 0.0518410, -0.1057058, 0.1049271
4: -0.0571271, 0.0163730, -0.0767542, 0.0194023, -0.0765294, 0.0931272
5: -0.0199711, 0.0285410, -0.0305920, 0.0315645, -0.0515356, 0.0591330
6: -0.0182337, 0.0574382, -0.0251009, 0.0672517, -0.0854854, 0.0825391
7: -0.0493294, 0.0271645, -0.0620919, 0.0371661, -0.0864955, 0.0892564
8: 0.8386118, 0.9915689, 0.8074871, 0.9919812, -0.1533694, 0.1840818
9: -0.0250701, 0.1156761, -0.0311402, 0.1496067, -0.1746769, 0.1468163

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519
time: 2.00 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519
time: 1.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.61 seconds
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 8, lower bound: -0.1310405, upper bound: 0.1426717
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 8, lower bound: -0.1315190, upper bound: 0.1430544
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1352364
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1422156
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.61
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519

## BFS NS instance: NS_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0018934, 0.0274415, -0.0540257, 0.0484446, -0.0503380, 0.0814672
1: -0.0111002, 0.0251368, -0.0408645, 0.0351376, -0.0462378, 0.0660013
2: 0.0011619, 0.0420268, -0.0250656, 0.1267675, -0.1256055, 0.0670924
3: -0.0361827, 0.0231324, -0.0628321, 0.0462464, -0.0824291, 0.0859645
4: -0.0207572, 0.0125499, -0.0751459, 0.0176258, -0.0383830, 0.0876957
5: -0.0043182, 0.0236500, -0.0276357, 0.0303247, -0.0346429, 0.0512857
6: -0.0158045, 0.0384322, -0.0203846, 0.0658701, -0.0816746, 0.0588168
7: -0.0263947, 0.0096511, -0.0576240, 0.0363257, -0.0627204, 0.0672752
8: 0.9065452, 0.9908300, 0.8125090, 0.9912007, -0.0846555, 0.1783210
9: -0.0188051, 0.0351859, -0.0291410, 0.1456623, -0.1644673, 0.0643270

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1384570
time: 1.81 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1415083
time: 1.70 seconds

## BFS NS instance: NS_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0019119, 0.0277417, -0.0376675, 0.0402609, -0.0421728, 0.0654092
1: -0.0111849, 0.0254320, -0.0300462, 0.0315486, -0.0427335, 0.0554782
2: 0.0010098, 0.0422721, -0.0153962, 0.0983359, -0.0973261, 0.0576683
3: -0.0364435, 0.0234478, -0.0527286, 0.0393295, -0.0757730, 0.0761764
4: -0.0209180, 0.0126847, -0.0551888, 0.0161649, -0.0370829, 0.0678735
5: -0.0044232, 0.0238883, -0.0190961, 0.0282199, -0.0326430, 0.0429844
6: -0.0160163, 0.0386838, -0.0177170, 0.0564468, -0.0724631, 0.0564008
7: -0.0265311, 0.0099074, -0.0484326, 0.0261334, -0.0526646, 0.0583400
8: 0.9059174, 0.9909688, 0.8414712, 0.9913020, -0.0853846, 0.1494976
9: -0.0189111, 0.0356451, -0.0244890, 0.1125363, -0.1314473, 0.0601342

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1231885, upper bound: 0.1389203
time: 3.67 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1231885, upper bound: 0.1418500
time: 1.53 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0020333, 0.0230930, -0.0020333, 0.0230930, -0.0251263, 0.0251263
1: -0.0101848, 0.0206137, -0.0101848, 0.0206137, -0.0307985, 0.0307985
2: 0.0036332, 0.0388584, 0.0036332, 0.0388584, -0.0352253, 0.0352253
3: -0.0315598, 0.0190998, -0.0315598, 0.0190998, -0.0506596, 0.0506596
4: -0.0182569, 0.0110922, -0.0182569, 0.0110922, -0.0293491, 0.0293491
5: -0.0027434, 0.0206050, -0.0027434, 0.0206050, -0.0233484, 0.0233484
6: -0.0162188, 0.0352391, -0.0162188, 0.0352391, -0.0514579, 0.0514579
7: -0.0259530, 0.0089382, -0.0259530, 0.0089382, -0.0348912, 0.0348912
8: 0.9123200, 0.9906856, 0.9123200, 0.9906856, -0.0783656, 0.0783656
9: -0.0171296, 0.0297734, -0.0171296, 0.0297734, -0.0469030, 0.0469030

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1341569
time: 1.41 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1349809
time: 1.54 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0020333, 0.0230930, -0.0391726, 0.0412229, -0.0432562, 0.0622656
1: -0.0101848, 0.0206137, -0.0310785, 0.0320921, -0.0422768, 0.0516922
2: 0.0036332, 0.0388584, -0.0164209, 0.1010427, -0.0974095, 0.0552793
3: -0.0315598, 0.0190998, -0.0538648, 0.0401076, -0.0716674, 0.0729646
4: -0.0182569, 0.0110922, -0.0571271, 0.0163730, -0.0346299, 0.0682193
5: -0.0027434, 0.0206050, -0.0199711, 0.0285410, -0.0312844, 0.0405761
6: -0.0162188, 0.0352391, -0.0182337, 0.0574382, -0.0736571, 0.0534728
7: -0.0259530, 0.0089382, -0.0493294, 0.0271645, -0.0531176, 0.0582676
8: 0.9123200, 0.9906856, 0.8386118, 0.9915689, -0.0792490, 0.1520737
9: -0.0171296, 0.0297734, -0.0250701, 0.1156761, -0.1328057, 0.0548435

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1413976
time: 1.60 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1419607
time: 1.70 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0391726, 0.0412229, -0.0020333, 0.0230930, -0.0622656, 0.0432562
1: -0.0310785, 0.0320921, -0.0101848, 0.0206137, -0.0516922, 0.0422768
2: -0.0164209, 0.1010427, 0.0036332, 0.0388584, -0.0552793, 0.0974095
3: -0.0538648, 0.0401076, -0.0315598, 0.0190998, -0.0729646, 0.0716674
4: -0.0571271, 0.0163730, -0.0182569, 0.0110922, -0.0682193, 0.0346299
5: -0.0199711, 0.0285410, -0.0027434, 0.0206050, -0.0405761, 0.0312844
6: -0.0182337, 0.0574382, -0.0162188, 0.0352391, -0.0534728, 0.0736571
7: -0.0493294, 0.0271645, -0.0259530, 0.0089382, -0.0582676, 0.0531176
8: 0.8386118, 0.9915689, 0.9123200, 0.9906856, -0.1520737, 0.0792490
9: -0.0250701, 0.1156761, -0.0171296, 0.0297734, -0.0548435, 0.1328057

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1442675, upper bound: 0.1342772
time: 1.49 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1349960
time: 1.39 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0391726, 0.0412229, -0.0391726, 0.0412229, -0.0803955, 0.0803955
1: -0.0310785, 0.0320921, -0.0310785, 0.0320921, -0.0631706, 0.0631706
2: -0.0164209, 0.1010427, -0.0164209, 0.1010427, -0.1174636, 0.1174636
3: -0.0538648, 0.0401076, -0.0538648, 0.0401076, -0.0939724, 0.0939724
4: -0.0571271, 0.0163730, -0.0571271, 0.0163730, -0.0735001, 0.0735001
5: -0.0199711, 0.0285410, -0.0199711, 0.0285410, -0.0485121, 0.0485121
6: -0.0182337, 0.0574382, -0.0182337, 0.0574382, -0.0756719, 0.0756719
7: -0.0493294, 0.0271645, -0.0493294, 0.0271645, -0.0764939, 0.0764939
8: 0.8386118, 0.9915689, 0.8386118, 0.9915689, -0.1529571, 0.1529571
9: -0.0250701, 0.1156761, -0.0250701, 0.1156761, -0.1407462, 0.1407462

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1456012
time: 2.94 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1462186
time: 1.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.46 seconds
NS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1384570
NS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1415083
NS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1231885, upper bound: 0.1389203
NS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1231885, upper bound: 0.1418500
NS_B2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1341569
NS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1349809
NS_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1413976
NS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1419607
NS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1442675, upper bound: 0.1342772
NS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1349960
NS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1456012
NS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1462186

## BFS NS instance: NS_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0006837, 0.0188431, -0.0540257, 0.0484446, -0.0491283, 0.0728688
1: -0.0074740, 0.0164229, -0.0408645, 0.0351376, -0.0426116, 0.0572874
2: 0.0060170, 0.0335689, -0.0250656, 0.1267675, -0.1207505, 0.0586345
3: -0.0278730, 0.0116460, -0.0628321, 0.0462464, -0.0741194, 0.0744781
4: -0.0155916, 0.0069257, -0.0751459, 0.0176258, -0.0332175, 0.0820716
5: -0.0013184, 0.0154896, -0.0276357, 0.0303247, -0.0316431, 0.0431253
6: -0.0154259, 0.0281984, -0.0203846, 0.0658701, -0.0812959, 0.0485831
7: -0.0197852, 0.0077672, -0.0576240, 0.0363257, -0.0561110, 0.0653913
8: 0.9340367, 0.9902968, 0.8125090, 0.9912007, -0.0571640, 0.1777878
9: -0.0169929, 0.0205112, -0.0291410, 0.1456623, -0.1626552, 0.0496522

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A1_B2_B1_A1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381247
time: 1.52 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1384570
time: 1.41 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0016533, 0.0241489, -0.0540257, 0.0484446, -0.0500979, 0.0781747
1: -0.0100820, 0.0218138, -0.0408645, 0.0351376, -0.0452196, 0.0626783
2: 0.0028107, 0.0391824, -0.0250656, 0.1267675, -0.1239568, 0.0642480
3: -0.0333673, 0.0194011, -0.0628321, 0.0462464, -0.0796138, 0.0822332
4: -0.0189572, 0.0109465, -0.0751459, 0.0176258, -0.0365831, 0.0860924
5: -0.0033052, 0.0208381, -0.0276357, 0.0303247, -0.0336299, 0.0484738
6: -0.0154685, 0.0353978, -0.0203846, 0.0658701, -0.0813386, 0.0557824
7: -0.0246929, 0.0088278, -0.0576240, 0.0363257, -0.0610186, 0.0664518
8: 0.9143675, 0.9906009, 0.8125090, 0.9912007, -0.0768332, 0.1780919
9: -0.0181882, 0.0299198, -0.0291410, 0.1456623, -0.1638505, 0.0590609

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1413475
time: 2.42 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1415083
time: 1.68 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0006995, 0.0190184, -0.0376675, 0.0402609, -0.0409604, 0.0566859
1: -0.0075436, 0.0165633, -0.0300462, 0.0315486, -0.0390922, 0.0466096
2: 0.0059481, 0.0337382, -0.0153962, 0.0983359, -0.0923878, 0.0491344
3: -0.0280911, 0.0118530, -0.0527286, 0.0393295, -0.0674206, 0.0645816
4: -0.0156933, 0.0070357, -0.0551888, 0.0161649, -0.0318581, 0.0622245
5: -0.0014174, 0.0155574, -0.0190961, 0.0282199, -0.0296373, 0.0346535
6: -0.0156307, 0.0284073, -0.0177170, 0.0564468, -0.0720775, 0.0461243
7: -0.0198765, 0.0079921, -0.0484326, 0.0261334, -0.0460099, 0.0564247
8: 0.9335532, 0.9904189, 0.8414712, 0.9913020, -0.0577488, 0.1489477
9: -0.0171030, 0.0207279, -0.0244890, 0.1125363, -0.1296393, 0.0452169

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A1_B2_B2_A1_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381548
time: 1.48 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1389203
time: 1.48 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0016698, 0.0243654, -0.0376675, 0.0402609, -0.0419307, 0.0620329
1: -0.0101601, 0.0220349, -0.0300462, 0.0315486, -0.0417087, 0.0520811
2: 0.0026673, 0.0393722, -0.0153962, 0.0983359, -0.0956686, 0.0547684
3: -0.0336137, 0.0196589, -0.0527286, 0.0393295, -0.0729432, 0.0723875
4: -0.0190942, 0.0110712, -0.0551888, 0.0161649, -0.0352591, 0.0662600
5: -0.0034083, 0.0210243, -0.0190961, 0.0282199, -0.0316282, 0.0401204
6: -0.0156797, 0.0356317, -0.0177170, 0.0564468, -0.0721265, 0.0533487
7: -0.0247940, 0.0090796, -0.0484326, 0.0261334, -0.0509274, 0.0575122
8: 0.9138293, 0.9907394, 0.8414712, 0.9913020, -0.0774727, 0.1492682
9: -0.0182961, 0.0302396, -0.0244890, 0.1125363, -0.1308323, 0.0547286

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1413901
time: 1.85 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1418500
time: 1.45 seconds

## BFS NS instance: NS_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0020111, 0.0228301, -0.0020333, 0.0230930, -0.0251041, 0.0248634
1: -0.0100880, 0.0203515, -0.0101848, 0.0206137, -0.0307018, 0.0305363
2: 0.0038061, 0.0386262, 0.0036332, 0.0388584, -0.0350523, 0.0349930
3: -0.0312670, 0.0187905, -0.0315598, 0.0190998, -0.0503668, 0.0503503
4: -0.0180915, 0.0109365, -0.0182569, 0.0110922, -0.0291837, 0.0291934
5: -0.0026025, 0.0203822, -0.0027434, 0.0206050, -0.0232075, 0.0231256
6: -0.0157730, 0.0349574, -0.0162188, 0.0352391, -0.0510121, 0.0511763
7: -0.0258241, 0.0084808, -0.0259530, 0.0089382, -0.0347622, 0.0344339
8: 0.9129899, 0.9904315, 0.9123200, 0.9906856, -0.0776957, 0.0781115
9: -0.0169642, 0.0293804, -0.0171296, 0.0297734, -0.0467375, 0.0465101

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1344610
time: 1.45 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1349809
time: 1.66 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0020170, 0.0228963, -0.0540257, 0.0484446, -0.0504616, 0.0769220
1: -0.0101121, 0.0204129, -0.0408645, 0.0351376, -0.0452497, 0.0612773
2: 0.0037626, 0.0386826, -0.0250656, 0.1267675, -0.1230049, 0.0637482
3: -0.0313342, 0.0188666, -0.0628321, 0.0462464, -0.0775806, 0.0816987
4: -0.0181312, 0.0109772, -0.0751459, 0.0176258, -0.0357570, 0.0861231
5: -0.0026452, 0.0204362, -0.0276357, 0.0303247, -0.0329699, 0.0480718
6: -0.0160173, 0.0350269, -0.0203846, 0.0658701, -0.0818874, 0.0554116
7: -0.0258570, 0.0087034, -0.0576240, 0.0363257, -0.0621827, 0.0663274
8: 0.9128178, 0.9905547, 0.8125090, 0.9912007, -0.0783828, 0.1780457
9: -0.0170181, 0.0294795, -0.0291410, 0.1456623, -0.1626803, 0.0586206

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A1_B2_B1_A1

### Relational analysis result of NS_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1406302
time: 1.42 seconds

## Relational analysis of NS_B2_A2_A1_B2_B1_A2

### Relational analysis result of NS_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1413976
time: 1.42 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0020333, 0.0230930, -0.0376675, 0.0402609, -0.0422942, 0.0607605
1: -0.0101848, 0.0206137, -0.0300462, 0.0315486, -0.0417333, 0.0506599
2: 0.0036332, 0.0388584, -0.0153962, 0.0983359, -0.0947027, 0.0542547
3: -0.0315598, 0.0190998, -0.0527286, 0.0393295, -0.0708893, 0.0718284
4: -0.0182569, 0.0110922, -0.0551888, 0.0161649, -0.0344218, 0.0662810
5: -0.0027434, 0.0206050, -0.0190961, 0.0282199, -0.0309633, 0.0397011
6: -0.0162188, 0.0352391, -0.0177170, 0.0564468, -0.0726656, 0.0529561
7: -0.0259530, 0.0089382, -0.0484326, 0.0261334, -0.0520865, 0.0573708
8: 0.9123200, 0.9906856, 0.8414712, 0.9913020, -0.0789821, 0.1492144
9: -0.0171296, 0.0297734, -0.0244890, 0.1125363, -0.1296659, 0.0542624

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1406437
time: 1.69 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1419607
time: 1.86 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0020170, 0.0228963, -0.0769220, 0.0504616
1: -0.0408645, 0.0351376, -0.0101121, 0.0204129, -0.0612773, 0.0452497
2: -0.0250656, 0.1267675, 0.0037626, 0.0386826, -0.0637482, 0.1230049
3: -0.0628321, 0.0462464, -0.0313342, 0.0188666, -0.0816987, 0.0775806
4: -0.0751459, 0.0176258, -0.0181312, 0.0109772, -0.0861231, 0.0357570
5: -0.0276357, 0.0303247, -0.0026452, 0.0204362, -0.0480718, 0.0329699
6: -0.0203846, 0.0658701, -0.0160173, 0.0350269, -0.0554116, 0.0818874
7: -0.0576240, 0.0363257, -0.0258570, 0.0087034, -0.0663274, 0.0621827
8: 0.8125090, 0.9912007, 0.9128178, 0.9905547, -0.1780457, 0.0783828
9: -0.0291410, 0.1456623, -0.0170181, 0.0294795, -0.0586206, 0.1626803

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1340690
time: 1.58 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1342772
time: 1.36 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0020333, 0.0230930, -0.0607605, 0.0422942
1: -0.0300462, 0.0315486, -0.0101848, 0.0206137, -0.0506599, 0.0417333
2: -0.0153962, 0.0983359, 0.0036332, 0.0388584, -0.0542547, 0.0947027
3: -0.0527286, 0.0393295, -0.0315598, 0.0190998, -0.0718284, 0.0708893
4: -0.0551888, 0.0161649, -0.0182569, 0.0110922, -0.0662810, 0.0344218
5: -0.0190961, 0.0282199, -0.0027434, 0.0206050, -0.0397011, 0.0309633
6: -0.0177170, 0.0564468, -0.0162188, 0.0352391, -0.0529561, 0.0726656
7: -0.0484326, 0.0261334, -0.0259530, 0.0089382, -0.0573708, 0.0520865
8: 0.8414712, 0.9913020, 0.9123200, 0.9906856, -0.1492144, 0.0789821
9: -0.0244890, 0.1125363, -0.0171296, 0.0297734, -0.0542624, 0.1296659

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1344760
time: 1.51 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1349960
time: 1.63 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0381503, 0.0405585, -0.0540257, 0.0484446, -0.0865949, 0.0945842
1: -0.0303768, 0.0317087, -0.0408645, 0.0351376, -0.0655144, 0.0725732
2: -0.0157193, 0.0991971, -0.0250656, 0.1267675, -0.1424868, 0.1242627
3: -0.0530817, 0.0395705, -0.0628321, 0.0462464, -0.0993282, 0.1024026
4: -0.0558087, 0.0162281, -0.0751459, 0.0176258, -0.0734345, 0.0913740
5: -0.0193728, 0.0283141, -0.0276357, 0.0303247, -0.0496975, 0.0559498
6: -0.0179263, 0.0567597, -0.0203846, 0.0658701, -0.0837964, 0.0771444
7: -0.0487196, 0.0264727, -0.0576240, 0.0363257, -0.0850453, 0.0840968
8: 0.8405618, 0.9914264, 0.8125090, 0.9912007, -0.1506389, 0.1789174
9: -0.0246719, 0.1135405, -0.0291410, 0.1456623, -0.1703342, 0.1426815

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1451275
time: 1.29 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1456012
time: 1.56 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0391726, 0.0412229, -0.0376675, 0.0402609, -0.0794335, 0.0788904
1: -0.0310785, 0.0320921, -0.0300462, 0.0315486, -0.0626271, 0.0621383
2: -0.0164209, 0.1010427, -0.0153962, 0.0983359, -0.1147568, 0.1164389
3: -0.0538648, 0.0401076, -0.0527286, 0.0393295, -0.0931943, 0.0928362
4: -0.0571271, 0.0163730, -0.0551888, 0.0161649, -0.0732920, 0.0715618
5: -0.0199711, 0.0285410, -0.0190961, 0.0282199, -0.0481910, 0.0476371
6: -0.0182337, 0.0574382, -0.0177170, 0.0564468, -0.0746804, 0.0751552
7: -0.0493294, 0.0271645, -0.0484326, 0.0261334, -0.0754628, 0.0755971
8: 0.8386118, 0.9915689, 0.8414712, 0.9913020, -0.1526902, 0.1500977
9: -0.0250701, 0.1156761, -0.0244890, 0.1125363, -0.1376064, 0.1401651

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1475741, upper bound: 0.1452948
time: 1.50 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1475741, upper bound: 0.1462186
time: 1.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.86 seconds
NS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381247
NS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1384570
NS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1413475
NS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1415083
NS_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381548
NS_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1389203
NS_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1413901
NS_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1418500
NS_B2_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1344610
NS_B2_A2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1349809
NS_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1406302
NS_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1413976
NS_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1406437
NS_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1419607
NS_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1340690
NS_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1342772
NS_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1344760
NS_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1349960
NS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1451275
NS_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1456012
NS_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1475741, upper bound: 0.1452948
NS_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.86
Output dim: 8, lower bound: -0.1475741, upper bound: 0.1462186

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0008707, 0.0198998, -0.0540257, 0.0484446, -0.0493153, 0.0739256
1: -0.0080284, 0.0173411, -0.0408645, 0.0351376, -0.0431659, 0.0582056
2: 0.0056564, 0.0347844, -0.0250656, 0.1267675, -0.1211111, 0.0598500
3: -0.0291138, 0.0132635, -0.0628321, 0.0462464, -0.0753603, 0.0760956
4: -0.0162181, 0.0077863, -0.0751459, 0.0176258, -0.0338440, 0.0829322
5: -0.0016230, 0.0160990, -0.0276357, 0.0303247, -0.0319478, 0.0437347
6: -0.0146490, 0.0297791, -0.0203846, 0.0658701, -0.0805191, 0.0501637
7: -0.0208000, 0.0070211, -0.0576240, 0.0363257, -0.0571257, 0.0646451
8: 0.9298151, 0.9897382, 0.8125090, 0.9912007, -0.0613856, 0.1772292
9: -0.0169754, 0.0221925, -0.0291410, 0.1456623, -0.1626377, 0.0513335

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222177, upper bound: 0.1380821
time: 1.76 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379473
time: 1.60 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0006787, 0.0187909, -0.0540257, 0.0484446, -0.0491234, 0.0728166
1: -0.0074538, 0.0163851, -0.0408645, 0.0351376, -0.0425914, 0.0572496
2: 0.0060388, 0.0335214, -0.0250656, 0.1267675, -0.1207287, 0.0585870
3: -0.0278158, 0.0115854, -0.0628321, 0.0462464, -0.0740623, 0.0744175
4: -0.0155632, 0.0068907, -0.0751459, 0.0176258, -0.0331890, 0.0820366
5: -0.0012751, 0.0154698, -0.0276357, 0.0303247, -0.0315998, 0.0431055
6: -0.0152177, 0.0281398, -0.0203846, 0.0658701, -0.0810878, 0.0485244
7: -0.0197580, 0.0076004, -0.0576240, 0.0363257, -0.0560837, 0.0652244
8: 0.9341831, 0.9902020, 0.8125090, 0.9912007, -0.0570176, 0.1776930
9: -0.0169410, 0.0204402, -0.0291410, 0.1456623, -0.1626032, 0.0495812

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222177, upper bound: 0.1384036
time: 1.46 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1382508
time: 1.51 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0018666, 0.0255524, -0.0540257, 0.0484446, -0.0503112, 0.0795781
1: -0.0107909, 0.0233222, -0.0408645, 0.0351376, -0.0459285, 0.0641867
2: 0.0019298, 0.0407174, -0.0250656, 0.1267675, -0.1248377, 0.0657830
3: -0.0348904, 0.0214866, -0.0628321, 0.0462464, -0.0811369, 0.0843187
4: -0.0198965, 0.0120266, -0.0751459, 0.0176258, -0.0375224, 0.0871725
5: -0.0037908, 0.0223690, -0.0276357, 0.0303247, -0.0341155, 0.0500047
6: -0.0147785, 0.0373392, -0.0203846, 0.0658701, -0.0806486, 0.0577238
7: -0.0258353, 0.0084135, -0.0576240, 0.0363257, -0.0621610, 0.0660376
8: 0.9090434, 0.9901649, 0.8125090, 0.9912007, -0.0821573, 0.1776559
9: -0.0182422, 0.0325506, -0.0291410, 0.1456623, -0.1639045, 0.0616917

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1412404
time: 1.55 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411386
time: 1.51 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0016469, 0.0240765, -0.0540257, 0.0484446, -0.0500915, 0.0781022
1: -0.0100552, 0.0217439, -0.0408645, 0.0351376, -0.0451928, 0.0626084
2: 0.0028583, 0.0391192, -0.0250656, 0.1267675, -0.1239092, 0.0641848
3: -0.0332908, 0.0193161, -0.0628321, 0.0462464, -0.0795372, 0.0821482
4: -0.0189128, 0.0109020, -0.0751459, 0.0176258, -0.0365386, 0.0860478
5: -0.0032642, 0.0207770, -0.0276357, 0.0303247, -0.0335889, 0.0484127
6: -0.0152594, 0.0353205, -0.0203846, 0.0658701, -0.0811295, 0.0557051
7: -0.0246563, 0.0086383, -0.0576240, 0.0363257, -0.0609820, 0.0662624
8: 0.9145586, 0.9904979, 0.8125090, 0.9912007, -0.0766421, 0.1779889
9: -0.0181429, 0.0298088, -0.0291410, 0.1456623, -0.1638051, 0.0589498

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1414092
time: 1.58 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1412839
time: 1.83 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0008707, 0.0198998, -0.0376675, 0.0402609, -0.0411316, 0.0575673
1: -0.0080284, 0.0173411, -0.0300462, 0.0315486, -0.0395770, 0.0473873
2: 0.0056564, 0.0347844, -0.0153962, 0.0983359, -0.0926795, 0.0501806
3: -0.0291138, 0.0132635, -0.0527286, 0.0393295, -0.0684433, 0.0659922
4: -0.0162181, 0.0077863, -0.0551888, 0.0161649, -0.0323830, 0.0629751
5: -0.0016230, 0.0160990, -0.0190961, 0.0282199, -0.0298429, 0.0351951
6: -0.0146490, 0.0297791, -0.0177170, 0.0564468, -0.0710958, 0.0474961
7: -0.0208000, 0.0070211, -0.0484326, 0.0261334, -0.0469334, 0.0554537
8: 0.9298151, 0.9897382, 0.8414712, 0.9913020, -0.0614870, 0.1482670
9: -0.0169754, 0.0221925, -0.0244890, 0.1125363, -0.1295117, 0.0466815

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1379746
time: 3.28 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379788
time: 1.33 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0006787, 0.0187909, -0.0376675, 0.0402609, -0.0409396, 0.0564584
1: -0.0074538, 0.0163851, -0.0300462, 0.0315486, -0.0390024, 0.0464314
2: 0.0060388, 0.0335214, -0.0153962, 0.0983359, -0.0922971, 0.0489176
3: -0.0278158, 0.0115854, -0.0527286, 0.0393295, -0.0671453, 0.0643141
4: -0.0155632, 0.0068907, -0.0551888, 0.0161649, -0.0317280, 0.0620796
5: -0.0012751, 0.0154698, -0.0190961, 0.0282199, -0.0294950, 0.0345659
6: -0.0152177, 0.0281398, -0.0177170, 0.0564468, -0.0716645, 0.0458568
7: -0.0197580, 0.0076004, -0.0484326, 0.0261334, -0.0458914, 0.0560330
8: 0.9341831, 0.9902020, 0.8414712, 0.9913020, -0.0571190, 0.1487308
9: -0.0169410, 0.0204402, -0.0244890, 0.1125363, -0.1294773, 0.0449292

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1387187
time: 1.79 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1387198
time: 1.96 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0018666, 0.0255524, -0.0376675, 0.0402609, -0.0421275, 0.0632199
1: -0.0107909, 0.0233222, -0.0300462, 0.0315486, -0.0423395, 0.0533685
2: 0.0019298, 0.0407174, -0.0153962, 0.0983359, -0.0964061, 0.0561136
3: -0.0348904, 0.0214866, -0.0527286, 0.0393295, -0.0742199, 0.0742152
4: -0.0198965, 0.0120266, -0.0551888, 0.0161649, -0.0360614, 0.0672154
5: -0.0037908, 0.0223690, -0.0190961, 0.0282199, -0.0320106, 0.0414651
6: -0.0147785, 0.0373392, -0.0177170, 0.0564468, -0.0712253, 0.0550562
7: -0.0258353, 0.0084135, -0.0484326, 0.0261334, -0.0519687, 0.0568461
8: 0.9090434, 0.9901649, 0.8414712, 0.9913020, -0.0822586, 0.1486937
9: -0.0182422, 0.0325506, -0.0244890, 0.1125363, -0.1307785, 0.0570397

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1412404
time: 1.71 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411806
time: 1.59 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0016469, 0.0240765, -0.0376675, 0.0402609, -0.0419078, 0.0617439
1: -0.0100552, 0.0217439, -0.0300462, 0.0315486, -0.0416038, 0.0517901
2: 0.0028583, 0.0391192, -0.0153962, 0.0983359, -0.0954776, 0.0545154
3: -0.0332908, 0.0193161, -0.0527286, 0.0393295, -0.0726203, 0.0720447
4: -0.0189128, 0.0109020, -0.0551888, 0.0161649, -0.0350777, 0.0660908
5: -0.0032642, 0.0207770, -0.0190961, 0.0282199, -0.0314841, 0.0398731
6: -0.0152594, 0.0353205, -0.0177170, 0.0564468, -0.0717062, 0.0530375
7: -0.0246563, 0.0086383, -0.0484326, 0.0261334, -0.0507897, 0.0570709
8: 0.9145586, 0.9904979, 0.8414712, 0.9913020, -0.0767434, 0.1490267
9: -0.0181429, 0.0298088, -0.0244890, 0.1125363, -0.1306792, 0.0542978

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1417490
time: 1.54 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1416182
time: 1.56 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0022585, 0.0241961, -0.0540257, 0.0484446, -0.0507031, 0.0782218
1: -0.0107930, 0.0218282, -0.0408645, 0.0351376, -0.0459306, 0.0626927
2: 0.0029539, 0.0401321, -0.0250656, 0.1267675, -0.1238136, 0.0651977
3: -0.0327061, 0.0208967, -0.0628321, 0.0462464, -0.0789525, 0.0837288
4: -0.0190003, 0.0120081, -0.0751459, 0.0176258, -0.0366262, 0.0871540
5: -0.0030589, 0.0219027, -0.0276357, 0.0303247, -0.0333836, 0.0495383
6: -0.0153214, 0.0369249, -0.0203846, 0.0658701, -0.0811915, 0.0573096
7: -0.0270843, 0.0082864, -0.0576240, 0.0363257, -0.0634100, 0.0659105
8: 0.9076468, 0.9901348, 0.8125090, 0.9912007, -0.0835539, 0.1776258
9: -0.0170160, 0.0319851, -0.0291410, 0.1456623, -0.1626782, 0.0611262

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A1_B2_B1_A1_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1406143
time: 1.46 seconds

## Relational analysis of NS_B2_A2_A1_B2_B1_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405008
time: 1.45 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0020111, 0.0228301, -0.0540257, 0.0484446, -0.0504557, 0.0768558
1: -0.0100880, 0.0203515, -0.0408645, 0.0351376, -0.0452256, 0.0612160
2: 0.0038061, 0.0386262, -0.0250656, 0.1267675, -0.1229613, 0.0636918
3: -0.0312670, 0.0187905, -0.0628321, 0.0462464, -0.0775135, 0.0816226
4: -0.0180915, 0.0109365, -0.0751459, 0.0176258, -0.0357174, 0.0860824
5: -0.0026025, 0.0203822, -0.0276357, 0.0303247, -0.0329272, 0.0480179
6: -0.0157730, 0.0349574, -0.0203846, 0.0658701, -0.0816431, 0.0553421
7: -0.0258241, 0.0084808, -0.0576240, 0.0363257, -0.0621498, 0.0661049
8: 0.9129899, 0.9904315, 0.8125090, 0.9912007, -0.0782108, 0.1779225
9: -0.0169642, 0.0293804, -0.0291410, 0.1456623, -0.1626264, 0.0585215

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A1_B2_B1_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1413952
time: 1.60 seconds

## Relational analysis of NS_B2_A2_A1_B2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1412840
time: 1.52 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0022585, 0.0241961, -0.0376675, 0.0402609, -0.0425194, 0.0618636
1: -0.0107930, 0.0218282, -0.0300462, 0.0315486, -0.0423416, 0.0518744
2: 0.0029539, 0.0401321, -0.0153962, 0.0983359, -0.0953820, 0.0555283
3: -0.0327061, 0.0208967, -0.0527286, 0.0393295, -0.0720356, 0.0736254
4: -0.0190003, 0.0120081, -0.0551888, 0.0161649, -0.0351652, 0.0671969
5: -0.0030589, 0.0219027, -0.0190961, 0.0282199, -0.0312788, 0.0409988
6: -0.0153214, 0.0369249, -0.0177170, 0.0564468, -0.0717682, 0.0546419
7: -0.0270843, 0.0082864, -0.0484326, 0.0261334, -0.0532177, 0.0567190
8: 0.9076468, 0.9901348, 0.8414712, 0.9913020, -0.0836552, 0.1486636
9: -0.0170160, 0.0319851, -0.0244890, 0.1125363, -0.1295522, 0.0564742

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1406277
time: 3.16 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405149
time: 2.00 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0020111, 0.0228301, -0.0376675, 0.0402609, -0.0422720, 0.0604976
1: -0.0100880, 0.0203515, -0.0300462, 0.0315486, -0.0416366, 0.0503978
2: 0.0038061, 0.0386262, -0.0153962, 0.0983359, -0.0945298, 0.0540224
3: -0.0312670, 0.0187905, -0.0527286, 0.0393295, -0.0705965, 0.0715191
4: -0.0180915, 0.0109365, -0.0551888, 0.0161649, -0.0342564, 0.0661253
5: -0.0026025, 0.0203822, -0.0190961, 0.0282199, -0.0308223, 0.0394783
6: -0.0157730, 0.0349574, -0.0177170, 0.0564468, -0.0722198, 0.0526744
7: -0.0258241, 0.0084808, -0.0484326, 0.0261334, -0.0519575, 0.0569134
8: 0.9129899, 0.9904315, 0.8414712, 0.9913020, -0.0783122, 0.1489603
9: -0.0169642, 0.0293804, -0.0244890, 0.1125363, -0.1295005, 0.0538695

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1419174
time: 1.62 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1417861
time: 2.68 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0022585, 0.0241961, -0.0782218, 0.0507031
1: -0.0408645, 0.0351376, -0.0107930, 0.0218282, -0.0626927, 0.0459306
2: -0.0250656, 0.1267675, 0.0029539, 0.0401321, -0.0651977, 0.1238136
3: -0.0628321, 0.0462464, -0.0327061, 0.0208967, -0.0837288, 0.0789525
4: -0.0751459, 0.0176258, -0.0190003, 0.0120081, -0.0871540, 0.0366262
5: -0.0276357, 0.0303247, -0.0030589, 0.0219027, -0.0495383, 0.0333836
6: -0.0203846, 0.0658701, -0.0153214, 0.0369249, -0.0573096, 0.0811915
7: -0.0576240, 0.0363257, -0.0270843, 0.0082864, -0.0659105, 0.0634100
8: 0.8125090, 0.9912007, 0.9076468, 0.9901348, -0.1776258, 0.0835539
9: -0.0291410, 0.1456623, -0.0170160, 0.0319851, -0.0611262, 0.1626782

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438661, upper bound: 0.1338975
time: 1.27 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1437923, upper bound: 0.1339141
time: 1.33 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0020111, 0.0228301, -0.0768558, 0.0504557
1: -0.0408645, 0.0351376, -0.0100880, 0.0203515, -0.0612160, 0.0452256
2: -0.0250656, 0.1267675, 0.0038061, 0.0386262, -0.0636918, 0.1229613
3: -0.0628321, 0.0462464, -0.0312670, 0.0187905, -0.0816226, 0.0775135
4: -0.0751459, 0.0176258, -0.0180915, 0.0109365, -0.0860824, 0.0357174
5: -0.0276357, 0.0303247, -0.0026025, 0.0203822, -0.0480179, 0.0329272
6: -0.0203846, 0.0658701, -0.0157730, 0.0349574, -0.0553421, 0.0816431
7: -0.0576240, 0.0363257, -0.0258241, 0.0084808, -0.0661049, 0.0621498
8: 0.8125090, 0.9912007, 0.9129899, 0.9904315, -0.1779225, 0.0782108
9: -0.0291410, 0.1456623, -0.0169642, 0.0293804, -0.0585215, 0.1626264

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438661, upper bound: 0.1341557
time: 2.51 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1437923, upper bound: 0.1341786
time: 1.35 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0022585, 0.0241961, -0.0618636, 0.0425194
1: -0.0300462, 0.0315486, -0.0107930, 0.0218282, -0.0518744, 0.0423416
2: -0.0153962, 0.0983359, 0.0029539, 0.0401321, -0.0555283, 0.0953820
3: -0.0527286, 0.0393295, -0.0327061, 0.0208967, -0.0736254, 0.0720356
4: -0.0551888, 0.0161649, -0.0190003, 0.0120081, -0.0671969, 0.0351652
5: -0.0190961, 0.0282199, -0.0030589, 0.0219027, -0.0409988, 0.0312788
6: -0.0177170, 0.0564468, -0.0153214, 0.0369249, -0.0546419, 0.0717682
7: -0.0484326, 0.0261334, -0.0270843, 0.0082864, -0.0567190, 0.0532177
8: 0.8414712, 0.9913020, 0.9076468, 0.9901348, -0.1486636, 0.0836552
9: -0.0244890, 0.1125363, -0.0170160, 0.0319851, -0.0564742, 0.1295522

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439452, upper bound: 0.1343105
time: 1.34 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438730, upper bound: 0.1343318
time: 1.72 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0020111, 0.0228301, -0.0604976, 0.0422720
1: -0.0300462, 0.0315486, -0.0100880, 0.0203515, -0.0503978, 0.0416366
2: -0.0153962, 0.0983359, 0.0038061, 0.0386262, -0.0540224, 0.0945298
3: -0.0527286, 0.0393295, -0.0312670, 0.0187905, -0.0715191, 0.0705965
4: -0.0551888, 0.0161649, -0.0180915, 0.0109365, -0.0661253, 0.0342564
5: -0.0190961, 0.0282199, -0.0026025, 0.0203822, -0.0394783, 0.0308223
6: -0.0177170, 0.0564468, -0.0157730, 0.0349574, -0.0526744, 0.0722198
7: -0.0484326, 0.0261334, -0.0258241, 0.0084808, -0.0569134, 0.0519575
8: 0.8414712, 0.9913020, 0.9129899, 0.9904315, -0.1489603, 0.0783122
9: -0.0244890, 0.1125363, -0.0169642, 0.0293804, -0.0538695, 0.1295005

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439452, upper bound: 0.1348619
time: 1.52 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438730, upper bound: 0.1348850
time: 1.98 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0540257, 0.0484446, -0.1024703, 0.1024703
1: -0.0408645, 0.0351376, -0.0408645, 0.0351376, -0.0760021, 0.0760021
2: -0.0250656, 0.1267675, -0.0250656, 0.1267675, -0.1518331, 0.1518331
3: -0.0628321, 0.0462464, -0.0628321, 0.0462464, -0.1090785, 0.1090785
4: -0.0751459, 0.0176258, -0.0751459, 0.0176258, -0.0927717, 0.0927717
5: -0.0276357, 0.0303247, -0.0276357, 0.0303247, -0.0579604, 0.0579604
6: -0.0203846, 0.0658701, -0.0203846, 0.0658701, -0.0862547, 0.0862547
7: -0.0576240, 0.0363257, -0.0576240, 0.0363257, -0.0939498, 0.0939498
8: 0.8125090, 0.9912007, 0.8125090, 0.9912007, -0.1786917, 0.1786917
9: -0.0291410, 0.1456623, -0.0291410, 0.1456623, -0.1748033, 0.1748033

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473156, upper bound: 0.1449157
time: 1.49 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1449427
time: 1.89 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0540257, 0.0484446, -0.0861121, 0.0942866
1: -0.0300462, 0.0315486, -0.0408645, 0.0351376, -0.0651838, 0.0724131
2: -0.0153962, 0.0983359, -0.0250656, 0.1267675, -0.1421637, 0.1234015
3: -0.0527286, 0.0393295, -0.0628321, 0.0462464, -0.0989751, 0.1021616
4: -0.0551888, 0.0161649, -0.0751459, 0.0176258, -0.0728147, 0.0913108
5: -0.0190961, 0.0282199, -0.0276357, 0.0303247, -0.0494208, 0.0558555
6: -0.0177170, 0.0564468, -0.0203846, 0.0658701, -0.0835871, 0.0768314
7: -0.0484326, 0.0261334, -0.0576240, 0.0363257, -0.0847583, 0.0837575
8: 0.8414712, 0.9913020, 0.8125090, 0.9912007, -0.1497295, 0.1787930
9: -0.0244890, 0.1125363, -0.0291410, 0.1456623, -0.1701513, 0.1416773

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473156, upper bound: 0.1453676
time: 1.66 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1453894
time: 1.37 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0376675, 0.0402609, -0.0942866, 0.0861121
1: -0.0408645, 0.0351376, -0.0300462, 0.0315486, -0.0724131, 0.0651838
2: -0.0250656, 0.1267675, -0.0153962, 0.0983359, -0.1234015, 0.1421637
3: -0.0628321, 0.0462464, -0.0527286, 0.0393295, -0.1021616, 0.0989751
4: -0.0751459, 0.0176258, -0.0551888, 0.0161649, -0.0913108, 0.0728147
5: -0.0276357, 0.0303247, -0.0190961, 0.0282199, -0.0558555, 0.0494208
6: -0.0203846, 0.0658701, -0.0177170, 0.0564468, -0.0768314, 0.0835871
7: -0.0576240, 0.0363257, -0.0484326, 0.0261334, -0.0837575, 0.0847583
8: 0.8125090, 0.9912007, 0.8414712, 0.9913020, -0.1787930, 0.1497295
9: -0.0291410, 0.1456623, -0.0244890, 0.1125363, -0.1416773, 0.1701513

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471016, upper bound: 0.1452927
time: 2.11 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1450822
time: 1.93 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0376675, 0.0402609, -0.0779284, 0.0779284
1: -0.0300462, 0.0315486, -0.0300462, 0.0315486, -0.0615948, 0.0615948
2: -0.0153962, 0.0983359, -0.0153962, 0.0983359, -0.1137321, 0.1137321
3: -0.0527286, 0.0393295, -0.0527286, 0.0393295, -0.0920581, 0.0920581
4: -0.0551888, 0.0161649, -0.0551888, 0.0161649, -0.0713537, 0.0713537
5: -0.0190961, 0.0282199, -0.0190961, 0.0282199, -0.0473160, 0.0473160
6: -0.0177170, 0.0564468, -0.0177170, 0.0564468, -0.0741638, 0.0741638
7: -0.0484326, 0.0261334, -0.0484326, 0.0261334, -0.0745660, 0.0745660
8: 0.8414712, 0.9913020, 0.8414712, 0.9913020, -0.1498308, 0.1498308
9: -0.0244890, 0.1125363, -0.0244890, 0.1125363, -0.1370253, 0.1370253

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B2_A2_A1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473156, upper bound: 0.1459782
time: 1.82 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1459901
time: 1.94 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.53 seconds
NS_B2_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1222177, upper bound: 0.1380821
NS_B2_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379473
NS_B2_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1222177, upper bound: 0.1384036
NS_B2_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1382508
NS_B2_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1412404
NS_B2_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411386
NS_B2_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1414092
NS_B2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1412839
NS_B2_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1379746
NS_B2_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379788
NS_B2_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1387187
NS_B2_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1387198
NS_B2_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1412404
NS_B2_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411806
NS_B2_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305067, upper bound: 0.1417490
NS_B2_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1416182
NS_B2_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1406143
NS_B2_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405008
NS_B2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1413952
NS_B2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1412840
NS_B2_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1406277
NS_B2_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405149
NS_B2_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1419174
NS_B2_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1417861
NS_B2_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1438661, upper bound: 0.1338975
NS_B2_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1437923, upper bound: 0.1339141
NS_B2_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1438661, upper bound: 0.1341557
NS_B2_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1437923, upper bound: 0.1341786
NS_B2_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1439452, upper bound: 0.1343105
NS_B2_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1438730, upper bound: 0.1343318
NS_B2_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1439452, upper bound: 0.1348619
NS_B2_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1438730, upper bound: 0.1348850
NS_B2_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1473156, upper bound: 0.1449157
NS_B2_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1449427
NS_B2_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1473156, upper bound: 0.1453676
NS_B2_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1453894
NS_B2_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1471016, upper bound: 0.1452927
NS_B2_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1450822
NS_B2_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1473156, upper bound: 0.1459782
NS_B2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.53
Output dim: 8, lower bound: -0.1471106, upper bound: 0.1459901

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0008585, 0.0197864, -0.0400207, 0.0401965, -0.0410550, 0.0598071
1: -0.0079790, 0.0172501, -0.0313751, 0.0308171, -0.0387960, 0.0486253
2: 0.0056997, 0.0346689, -0.0159430, 0.1019839, -0.0962842, 0.0506119
3: -0.0289772, 0.0131195, -0.0529631, 0.0394997, -0.0684769, 0.0660826
4: -0.0161520, 0.0077095, -0.0574184, 0.0159635, -0.0321155, 0.0651279
5: -0.0015731, 0.0160486, -0.0197886, 0.0277936, -0.0293667, 0.0358372
6: -0.0146054, 0.0296341, -0.0170941, 0.0570708, -0.0716762, 0.0467281
7: -0.0207292, 0.0069914, -0.0494355, 0.0270410, -0.0477702, 0.0564268
8: 0.9301687, 0.9897363, 0.8383634, 0.9908019, -0.0606331, 0.1513728
9: -0.0169335, 0.0220409, -0.0243421, 0.1167575, -0.1336910, 0.0463830

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219993, upper bound: 0.1379094
time: 1.60 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222056, upper bound: 0.1380821
time: 1.62 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0008622, 0.0198235, -0.0470752, 0.0447411, -0.0456033, 0.0668987
1: -0.0079952, 0.0172803, -0.0362609, 0.0334237, -0.0414189, 0.0535411
2: 0.0056857, 0.0347070, -0.0207431, 0.1147738, -0.1090881, 0.0554501
3: -0.0290225, 0.0131670, -0.0584405, 0.0431066, -0.0721291, 0.0716075
4: -0.0161737, 0.0077346, -0.0665866, 0.0169369, -0.0331106, 0.0743213
5: -0.0015890, 0.0160652, -0.0238807, 0.0293077, -0.0308967, 0.0399459
6: -0.0146109, 0.0296820, -0.0188429, 0.0618150, -0.0764259, 0.0485248
7: -0.0207519, 0.0069973, -0.0536168, 0.0316714, -0.0524233, 0.0606142
8: 0.9300530, 0.9897366, 0.8249218, 0.9910174, -0.0609644, 0.1648148
9: -0.0169470, 0.0220895, -0.0270275, 0.1313044, -0.1482514, 0.0491170

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1220034, upper bound: 0.1377542
time: 1.36 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222156, upper bound: 0.1379473
time: 1.84 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0006668, 0.0186794, -0.0400207, 0.0401965, -0.0408633, 0.0587001
1: -0.0074052, 0.0162957, -0.0313751, 0.0308171, -0.0382223, 0.0476708
2: 0.0060817, 0.0334081, -0.0159430, 0.1019839, -0.0959022, 0.0493511
3: -0.0276826, 0.0114427, -0.0529631, 0.0394997, -0.0671823, 0.0644058
4: -0.0154985, 0.0068147, -0.0574184, 0.0159635, -0.0314620, 0.0642331
5: -0.0012282, 0.0154201, -0.0197886, 0.0277936, -0.0290218, 0.0352087
6: -0.0151734, 0.0279972, -0.0170941, 0.0570708, -0.0722442, 0.0450912
7: -0.0196879, 0.0075711, -0.0494355, 0.0270410, -0.0467289, 0.0570066
8: 0.9345318, 0.9902000, 0.8383634, 0.9908019, -0.0562701, 0.1518366
9: -0.0168997, 0.0202934, -0.0243421, 0.1167575, -0.1336572, 0.0446355

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222130, upper bound: 0.1382363
time: 1.80 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1223969, upper bound: 0.1384036
time: 1.66 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0006705, 0.0187162, -0.0470752, 0.0447411, -0.0454116, 0.0657914
1: -0.0074214, 0.0163257, -0.0362609, 0.0334237, -0.0408451, 0.0525866
2: 0.0060677, 0.0334460, -0.0207431, 0.1147738, -0.1087061, 0.0541891
3: -0.0277272, 0.0114904, -0.0584405, 0.0431066, -0.0708338, 0.0699310
4: -0.0155200, 0.0068400, -0.0665866, 0.0169369, -0.0324569, 0.0734266
5: -0.0012430, 0.0154367, -0.0238807, 0.0293077, -0.0305507, 0.0393174
6: -0.0151799, 0.0280449, -0.0188429, 0.0618150, -0.0769949, 0.0468878
7: -0.0197107, 0.0075772, -0.0536168, 0.0316714, -0.0513821, 0.0611940
8: 0.9344159, 0.9902003, 0.8249218, 0.9910174, -0.0566015, 0.1652785
9: -0.0169131, 0.0203408, -0.0270275, 0.1313044, -0.1482175, 0.0473683

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222204, upper bound: 0.1380858
time: 1.42 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224036, upper bound: 0.1382508
time: 1.41 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0018446, 0.0253376, -0.0400207, 0.0401965, -0.0420411, 0.0653582
1: -0.0107035, 0.0231011, -0.0313751, 0.0308171, -0.0415205, 0.0544762
2: 0.0020702, 0.0405154, -0.0159430, 0.1019839, -0.0999137, 0.0564584
3: -0.0346573, 0.0212074, -0.0529631, 0.0394997, -0.0741570, 0.0741705
4: -0.0197595, 0.0118896, -0.0574184, 0.0159635, -0.0357231, 0.0693080
5: -0.0037002, 0.0221692, -0.0197886, 0.0277936, -0.0314937, 0.0419578
6: -0.0147167, 0.0370870, -0.0170941, 0.0570708, -0.0717875, 0.0541811
7: -0.0257043, 0.0083316, -0.0494355, 0.0270410, -0.0527453, 0.0577671
8: 0.9096730, 0.9901440, 0.8383634, 0.9908019, -0.0811288, 0.1517806
9: -0.0181753, 0.0322051, -0.0243421, 0.1167575, -0.1349328, 0.0565472

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1302950, upper bound: 0.1410337
time: 1.57 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305037, upper bound: 0.1412404
time: 1.62 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0018514, 0.0254070, -0.0470752, 0.0447411, -0.0465925, 0.0724821
1: -0.0107318, 0.0231729, -0.0362609, 0.0334237, -0.0441554, 0.0594338
2: 0.0020248, 0.0405808, -0.0207431, 0.1147738, -0.1127490, 0.0613240
3: -0.0347329, 0.0212980, -0.0584405, 0.0431066, -0.0778395, 0.0797386
4: -0.0198039, 0.0119339, -0.0665866, 0.0169369, -0.0367408, 0.0785206
5: -0.0037292, 0.0222341, -0.0238807, 0.0293077, -0.0330369, 0.0461149
6: -0.0147280, 0.0371689, -0.0188429, 0.0618150, -0.0765430, 0.0560118
7: -0.0257461, 0.0083547, -0.0536168, 0.0316714, -0.0574176, 0.0619715
8: 0.9094694, 0.9901504, 0.8249218, 0.9910174, -0.0815480, 0.1652286
9: -0.0181965, 0.0323164, -0.0270275, 0.1313044, -0.1495008, 0.0593439

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1302965, upper bound: 0.1408911
time: 1.85 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305144, upper bound: 0.1411386
time: 1.52 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0016254, 0.0238650, -0.0400207, 0.0401965, -0.0418219, 0.0638857
1: -0.0099695, 0.0215255, -0.0313751, 0.0308171, -0.0407866, 0.0529006
2: 0.0029962, 0.0389208, -0.0159430, 0.1019839, -0.0989877, 0.0548638
3: -0.0330595, 0.0190400, -0.0529631, 0.0394997, -0.0725592, 0.0720031
4: -0.0187776, 0.0107678, -0.0574184, 0.0159635, -0.0347411, 0.0681861
5: -0.0031760, 0.0205795, -0.0197886, 0.0277936, -0.0309695, 0.0403681
6: -0.0151982, 0.0350693, -0.0170941, 0.0570708, -0.0722689, 0.0521634
7: -0.0245275, 0.0085588, -0.0494355, 0.0270410, -0.0515685, 0.0579943
8: 0.9151753, 0.9904779, 0.8383634, 0.9908019, -0.0756266, 0.1521145
9: -0.0180765, 0.0294723, -0.0243421, 0.1167575, -0.1348339, 0.0538144

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1303332, upper bound: 0.1412018
time: 1.74 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305492, upper bound: 0.1414092
time: 1.64 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0016321, 0.0239339, -0.0470752, 0.0447411, -0.0463732, 0.0710091
1: -0.0099975, 0.0215972, -0.0362609, 0.0334237, -0.0434211, 0.0578581
2: 0.0029513, 0.0389856, -0.0207431, 0.1147738, -0.1118226, 0.0597288
3: -0.0331353, 0.0191305, -0.0584405, 0.0431066, -0.0762419, 0.0775710
4: -0.0188218, 0.0108115, -0.0665866, 0.0169369, -0.0357587, 0.0773981
5: -0.0032044, 0.0206443, -0.0238807, 0.0293077, -0.0325121, 0.0445250
6: -0.0152098, 0.0351517, -0.0188429, 0.0618150, -0.0770248, 0.0539946
7: -0.0245690, 0.0085817, -0.0536168, 0.0316714, -0.0562404, 0.0621985
8: 0.9149743, 0.9904841, 0.8249218, 0.9910174, -0.0760431, 0.1655623
9: -0.0180976, 0.0295814, -0.0270275, 0.1313044, -0.1494020, 0.0566088

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1303345, upper bound: 0.1410613
time: 1.59 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305546, upper bound: 0.1412839
time: 1.90 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0007255, 0.0185341, -0.0361535, 0.0393951, -0.0401206, 0.0546876
1: -0.0074363, 0.0162442, -0.0290321, 0.0310964, -0.0385327, 0.0452763
2: 0.0061794, 0.0333988, -0.0144299, 0.0956708, -0.0894915, 0.0478287
3: -0.0274709, 0.0115269, -0.0516926, 0.0386126, -0.0660835, 0.0632195
4: -0.0154255, 0.0068615, -0.0532982, 0.0159908, -0.0314163, 0.0601597
5: -0.0010335, 0.0154944, -0.0182603, 0.0279541, -0.0289876, 0.0337547
6: -0.0141299, 0.0280419, -0.0174481, 0.0555105, -0.0696404, 0.0454901
7: -0.0199523, 0.0066679, -0.0475553, 0.0252021, -0.0451544, 0.0542232
8: 0.9340628, 0.9897153, 0.8442402, 0.9912613, -0.0571985, 0.1454751
9: -0.0164730, 0.0203785, -0.0240001, 0.1094263, -0.1258993, 0.0443785

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224619, upper bound: 0.1377987
time: 1.52 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1227176, upper bound: 0.1379746
time: 1.96 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0008433, 0.0199433, -0.0366432, 0.0396761, -0.0405193, 0.0565865
1: -0.0080161, 0.0174116, -0.0293602, 0.0312437, -0.0392598, 0.0467718
2: 0.0056414, 0.0348078, -0.0147429, 0.0965330, -0.0908916, 0.0495507
3: -0.0292257, 0.0132736, -0.0520285, 0.0388444, -0.0680701, 0.0653022
4: -0.0162456, 0.0077591, -0.0539100, 0.0160473, -0.0322929, 0.0616691
5: -0.0016268, 0.0160642, -0.0185309, 0.0280405, -0.0296673, 0.0345951
6: -0.0143212, 0.0297787, -0.0175293, 0.0558140, -0.0701352, 0.0473080
7: -0.0207167, 0.0068729, -0.0478392, 0.0254996, -0.0462163, 0.0547121
8: 0.9300352, 0.9897074, 0.8433452, 0.9912735, -0.0612382, 0.1463622
9: -0.0169669, 0.0221489, -0.0241571, 0.1104309, -0.1273978, 0.0463060

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222866, upper bound: 0.1377996
time: 1.43 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225370, upper bound: 0.1379788
time: 1.89 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0005356, 0.0174531, -0.0361535, 0.0393951, -0.0399307, 0.0536066
1: -0.0068692, 0.0153094, -0.0290321, 0.0310964, -0.0379656, 0.0443415
2: 0.0065546, 0.0321588, -0.0144299, 0.0956708, -0.0891163, 0.0465887
3: -0.0262104, 0.0098755, -0.0516926, 0.0386126, -0.0648230, 0.0615681
4: -0.0147856, 0.0059808, -0.0532982, 0.0159908, -0.0307764, 0.0592790
5: -0.0007190, 0.0148735, -0.0182603, 0.0279541, -0.0286731, 0.0331338
6: -0.0146943, 0.0264273, -0.0174481, 0.0555105, -0.0702049, 0.0438755
7: -0.0189206, 0.0072527, -0.0475553, 0.0252021, -0.0441227, 0.0548080
8: 0.9383700, 0.9901789, 0.8442402, 0.9912613, -0.0528913, 0.1459387
9: -0.0164445, 0.0186969, -0.0240001, 0.1094263, -0.1258708, 0.0426970

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1130602, upper bound: 0.1303159
time: 1.46 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1123090, upper bound: 0.1271023
time: 1.15 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0006505, 0.0188606, -0.0366432, 0.0396761, -0.0403266, 0.0555038
1: -0.0074290, 0.0164821, -0.0293602, 0.0312437, -0.0386727, 0.0458423
2: 0.0060065, 0.0335209, -0.0147429, 0.0965330, -0.0905265, 0.0482638
3: -0.0279165, 0.0116459, -0.0520285, 0.0388444, -0.0667609, 0.0636744
4: -0.0155779, 0.0068703, -0.0539100, 0.0160473, -0.0316251, 0.0607803
5: -0.0012776, 0.0154287, -0.0185309, 0.0280405, -0.0293181, 0.0339596
6: -0.0148937, 0.0281489, -0.0175293, 0.0558140, -0.0707077, 0.0456782
7: -0.0196854, 0.0074638, -0.0478392, 0.0254996, -0.0451851, 0.0553030
8: 0.9344307, 0.9901603, 0.8433452, 0.9912735, -0.0568428, 0.1468151
9: -0.0169174, 0.0204018, -0.0241571, 0.1104309, -0.1273483, 0.0445589

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1130577, upper bound: 0.1304177
time: 1.78 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A2_A2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1123078, upper bound: 0.1271130
time: 1.32 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0018446, 0.0253376, -0.0235322, 0.0332591, -0.0351037, 0.0488698
1: -0.0107035, 0.0231011, -0.0221560, 0.0271909, -0.0378944, 0.0452571
2: 0.0020702, 0.0405154, -0.0079077, 0.0733482, -0.0712780, 0.0484231
3: -0.0346573, 0.0212074, -0.0438773, 0.0325672, -0.0672245, 0.0650847
4: -0.0197595, 0.0118896, -0.0405357, 0.0144962, -0.0342558, 0.0524253
5: -0.0037002, 0.0221692, -0.0120661, 0.0256611, -0.0293613, 0.0342353
6: -0.0147167, 0.0370870, -0.0156039, 0.0483375, -0.0630542, 0.0526910
7: -0.0257043, 0.0083316, -0.0403037, 0.0180461, -0.0437504, 0.0486353
8: 0.9096730, 0.9901440, 0.8671969, 0.9909092, -0.0812362, 0.1229471
9: -0.0181753, 0.0322051, -0.0208733, 0.0834956, -0.1016710, 0.0530784

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305492, upper bound: 0.1411006
time: 1.49 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1307681, upper bound: 0.1412868
time: 1.39 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0018514, 0.0254070, -0.0305649, 0.0369758, -0.0388272, 0.0559719
1: -0.0107318, 0.0231729, -0.0257718, 0.0298844, -0.0406162, 0.0489447
2: 0.0020248, 0.0405808, -0.0115335, 0.0860591, -0.0840343, 0.0521143
3: -0.0347329, 0.0212980, -0.0486584, 0.0362744, -0.0710073, 0.0699564
4: -0.0198039, 0.0119339, -0.0473431, 0.0154978, -0.0353017, 0.0592771
5: -0.0037292, 0.0222341, -0.0155673, 0.0272415, -0.0309707, 0.0378015
6: -0.0147280, 0.0371689, -0.0165835, 0.0525247, -0.0672527, 0.0537524
7: -0.0257461, 0.0083547, -0.0444587, 0.0221003, -0.0478464, 0.0528134
8: 0.9094694, 0.9901504, 0.8538592, 0.9911215, -0.0816521, 0.1362912
9: -0.0181965, 0.0323164, -0.0226267, 0.0984207, -0.1166172, 0.0549431

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305508, upper bound: 0.1409683
time: 2.64 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1307813, upper bound: 0.1411806
time: 1.58 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0016254, 0.0238650, -0.0235322, 0.0332591, -0.0348845, 0.0473972
1: -0.0099695, 0.0215255, -0.0221560, 0.0271909, -0.0371605, 0.0436815
2: 0.0029962, 0.0389208, -0.0079077, 0.0733482, -0.0703520, 0.0468285
3: -0.0330595, 0.0190400, -0.0438773, 0.0325672, -0.0656267, 0.0629172
4: -0.0187776, 0.0107678, -0.0405357, 0.0144962, -0.0332738, 0.0513035
5: -0.0031760, 0.0205795, -0.0120661, 0.0256611, -0.0288371, 0.0326456
6: -0.0151982, 0.0350693, -0.0156039, 0.0483375, -0.0635357, 0.0506733
7: -0.0245275, 0.0085588, -0.0403037, 0.0180461, -0.0425737, 0.0488625
8: 0.9151753, 0.9904779, 0.8671969, 0.9909092, -0.0757339, 0.1232810
9: -0.0180765, 0.0294723, -0.0208733, 0.0834956, -0.1015721, 0.0503456

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1306289, upper bound: 0.1415561
time: 2.41 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1308391, upper bound: 0.1417490
time: 1.35 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0016321, 0.0239339, -0.0305649, 0.0369758, -0.0386079, 0.0544988
1: -0.0099975, 0.0215972, -0.0257718, 0.0298844, -0.0398819, 0.0473690
2: 0.0029513, 0.0389856, -0.0115335, 0.0860591, -0.0831078, 0.0505191
3: -0.0331353, 0.0191305, -0.0486584, 0.0362744, -0.0694098, 0.0677888
4: -0.0188218, 0.0108115, -0.0473431, 0.0154978, -0.0343196, 0.0581546
5: -0.0032044, 0.0206443, -0.0155673, 0.0272415, -0.0304460, 0.0362116
6: -0.0152098, 0.0351517, -0.0165835, 0.0525247, -0.0677345, 0.0517352
7: -0.0245690, 0.0085817, -0.0444587, 0.0221003, -0.0466693, 0.0530404
8: 0.9149743, 0.9904841, 0.8538592, 0.9911215, -0.0761472, 0.1366249
9: -0.0180976, 0.0295814, -0.0226267, 0.0984207, -0.1165183, 0.0522081

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1306289, upper bound: 0.1414100
time: 1.76 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1308487, upper bound: 0.1416182
time: 2.67 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0022394, 0.0240106, -0.0400207, 0.0401965, -0.0424359, 0.0640313
1: -0.0107177, 0.0216336, -0.0313751, 0.0308171, -0.0415348, 0.0530087
2: 0.0030754, 0.0399569, -0.0159430, 0.1019839, -0.0989085, 0.0558999
3: -0.0325018, 0.0206539, -0.0529631, 0.0394997, -0.0720015, 0.0736170
4: -0.0188807, 0.0118890, -0.0574184, 0.0159635, -0.0348442, 0.0693074
5: -0.0029819, 0.0217282, -0.0197886, 0.0277936, -0.0307755, 0.0415168
6: -0.0152682, 0.0367036, -0.0170941, 0.0570708, -0.0723390, 0.0537976
7: -0.0269711, 0.0082170, -0.0494355, 0.0270410, -0.0540121, 0.0576525
8: 0.9081902, 0.9901170, 0.8383634, 0.9908019, -0.0826117, 0.1517535
9: -0.0169583, 0.0316862, -0.0243421, 0.1167575, -0.1337158, 0.0560283

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1347767, upper bound: 0.1403786
time: 1.57 seconds

## Relational analysis of NS_B2_A2_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1406007
time: 1.39 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0022434, 0.0240532, -0.0470752, 0.0447411, -0.0469845, 0.0711284
1: -0.0107349, 0.0216789, -0.0362609, 0.0334237, -0.0441586, 0.0579398
2: 0.0030475, 0.0399972, -0.0207431, 0.1147738, -0.1117264, 0.0607403
3: -0.0325491, 0.0207102, -0.0584405, 0.0431066, -0.0756557, 0.0791507
4: -0.0189083, 0.0119164, -0.0665866, 0.0169369, -0.0358452, 0.0785030
5: -0.0029993, 0.0217686, -0.0238807, 0.0293077, -0.0323070, 0.0456493
6: -0.0152734, 0.0367549, -0.0188429, 0.0618150, -0.0770884, 0.0555977
7: -0.0269961, 0.0082304, -0.0536168, 0.0316714, -0.0586675, 0.0618473
8: 0.9080660, 0.9901208, 0.8249218, 0.9910174, -0.0829514, 0.1651990
9: -0.0169712, 0.0317543, -0.0270275, 0.1313044, -0.1482756, 0.0587818

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1347767, upper bound: 0.1402267
time: 1.73 seconds

## Relational analysis of NS_B2_A2_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1404863
time: 1.78 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0019909, 0.0226329, -0.0400207, 0.0401965, -0.0421874, 0.0626536
1: -0.0100082, 0.0201454, -0.0313751, 0.0308171, -0.0408253, 0.0515205
2: 0.0039349, 0.0384408, -0.0159430, 0.1019839, -0.0980490, 0.0543838
3: -0.0310514, 0.0185317, -0.0529631, 0.0394997, -0.0705511, 0.0714948
4: -0.0179658, 0.0108103, -0.0574184, 0.0159635, -0.0339293, 0.0682287
5: -0.0025229, 0.0201964, -0.0197886, 0.0277936, -0.0303165, 0.0399850
6: -0.0157195, 0.0347209, -0.0170941, 0.0570708, -0.0727902, 0.0518150
7: -0.0257033, 0.0084083, -0.0494355, 0.0270410, -0.0527443, 0.0578438
8: 0.9135684, 0.9904129, 0.8383634, 0.9908019, -0.0772335, 0.1520495
9: -0.0169037, 0.0290650, -0.0243421, 0.1167575, -0.1336612, 0.0534071

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1352277, upper bound: 0.1411910
time: 1.55 seconds

## Relational analysis of NS_B2_A2_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1353826, upper bound: 0.1413783
time: 1.47 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0019964, 0.0226894, -0.0470752, 0.0447411, -0.0467375, 0.0697646
1: -0.0100311, 0.0202051, -0.0362609, 0.0334237, -0.0434548, 0.0564660
2: 0.0038981, 0.0384942, -0.0207431, 0.1147738, -0.1108757, 0.0592373
3: -0.0311135, 0.0186067, -0.0584405, 0.0431066, -0.0742201, 0.0770472
4: -0.0180019, 0.0108466, -0.0665866, 0.0169369, -0.0349388, 0.0774333
5: -0.0025452, 0.0202503, -0.0238807, 0.0293077, -0.0318529, 0.0441310
6: -0.0157265, 0.0347896, -0.0188429, 0.0618150, -0.0775415, 0.0536324
7: -0.0257375, 0.0084261, -0.0536168, 0.0316714, -0.0574090, 0.0620429
8: 0.9134017, 0.9904178, 0.8249218, 0.9910174, -0.0776157, 0.1654960
9: -0.0169202, 0.0291553, -0.0270275, 0.1313044, -0.1482246, 0.0561827

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1352277, upper bound: 0.1410659
time: 1.86 seconds

## Relational analysis of NS_B2_A2_A1_B2_B1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1353880, upper bound: 0.1412669
time: 2.98 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0022394, 0.0240106, -0.0235322, 0.0332591, -0.0354985, 0.0475428
1: -0.0107177, 0.0216336, -0.0221560, 0.0271909, -0.0379086, 0.0437895
2: 0.0030754, 0.0399569, -0.0079077, 0.0733482, -0.0702728, 0.0478646
3: -0.0325018, 0.0206539, -0.0438773, 0.0325672, -0.0650690, 0.0645312
4: -0.0188807, 0.0118890, -0.0405357, 0.0144962, -0.0333770, 0.0524247
5: -0.0029819, 0.0217282, -0.0120661, 0.0256611, -0.0286431, 0.0337943
6: -0.0152682, 0.0367036, -0.0156039, 0.0483375, -0.0636057, 0.0523075
7: -0.0269711, 0.0082170, -0.0403037, 0.0180461, -0.0450173, 0.0485207
8: 0.9081902, 0.9901170, 0.8671969, 0.9909092, -0.0827190, 0.1229200
9: -0.0169583, 0.0316862, -0.0208733, 0.0834956, -0.1004540, 0.0525595

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1199217, upper bound: 0.1288912
time: 1.50 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1190971, upper bound: 0.1255403
time: 1.10 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0022434, 0.0240532, -0.0305649, 0.0369758, -0.0392192, 0.0546182
1: -0.0107349, 0.0216789, -0.0257718, 0.0298844, -0.0406194, 0.0474507
2: 0.0030475, 0.0399972, -0.0115335, 0.0860591, -0.0830116, 0.0515306
3: -0.0325491, 0.0207102, -0.0486584, 0.0362744, -0.0688235, 0.0693685
4: -0.0189083, 0.0119164, -0.0473431, 0.0154978, -0.0344060, 0.0592595
5: -0.0029993, 0.0217686, -0.0155673, 0.0272415, -0.0302409, 0.0373360
6: -0.0152734, 0.0367549, -0.0165835, 0.0525247, -0.0677981, 0.0533384
7: -0.0269961, 0.0082304, -0.0444587, 0.0221003, -0.0490964, 0.0526891
8: 0.9080660, 0.9901208, 0.8538592, 0.9911215, -0.0830555, 0.1362616
9: -0.0169712, 0.0317543, -0.0226267, 0.0984207, -0.1153919, 0.0543810

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1350662, upper bound: 0.1402405
time: 1.68 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1352903, upper bound: 0.1404998
time: 2.40 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0019909, 0.0226329, -0.0235322, 0.0332591, -0.0352500, 0.0461651
1: -0.0100082, 0.0201454, -0.0221560, 0.0271909, -0.0371991, 0.0423014
2: 0.0039349, 0.0384408, -0.0079077, 0.0733482, -0.0694133, 0.0463485
3: -0.0310514, 0.0185317, -0.0438773, 0.0325672, -0.0636186, 0.0624089
4: -0.0179658, 0.0108103, -0.0405357, 0.0144962, -0.0324620, 0.0513460
5: -0.0025229, 0.0201964, -0.0120661, 0.0256611, -0.0281841, 0.0322625
6: -0.0157195, 0.0347209, -0.0156039, 0.0483375, -0.0640570, 0.0503249
7: -0.0257033, 0.0084083, -0.0403037, 0.0180461, -0.0437494, 0.0487120
8: 0.9135684, 0.9904129, 0.8671969, 0.9909092, -0.0773408, 0.1232160
9: -0.0169037, 0.0290650, -0.0208733, 0.0834956, -0.1003993, 0.0499383

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1239921, upper bound: 0.1330441
time: 1.46 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1232609, upper bound: 0.1300494
time: 1.34 seconds

## BFS NS instance: NS_B2_A2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0019964, 0.0226894, -0.0305649, 0.0369758, -0.0389722, 0.0532543
1: -0.0100311, 0.0202051, -0.0257718, 0.0298844, -0.0399156, 0.0459769
2: 0.0038981, 0.0384942, -0.0115335, 0.0860591, -0.0821610, 0.0500276
3: -0.0311135, 0.0186067, -0.0486584, 0.0362744, -0.0673879, 0.0672651
4: -0.0180019, 0.0108466, -0.0473431, 0.0154978, -0.0334997, 0.0581898
5: -0.0025452, 0.0202503, -0.0155673, 0.0272415, -0.0297868, 0.0358176
6: -0.0157265, 0.0347896, -0.0165835, 0.0525247, -0.0682512, 0.0513731
7: -0.0257375, 0.0084261, -0.0444587, 0.0221003, -0.0478378, 0.0528848
8: 0.9134017, 0.9904178, 0.8538592, 0.9911215, -0.0777198, 0.1365586
9: -0.0169202, 0.0291553, -0.0226267, 0.0984207, -0.1153410, 0.0517819

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1240446, upper bound: 0.1330424
time: 1.51 seconds

## Relational analysis of NS_B2_A2_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_B2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1233291, upper bound: 0.1302358
time: 1.27 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0400207, 0.0401965, -0.0022394, 0.0240106, -0.0640313, 0.0424359
1: -0.0313751, 0.0308171, -0.0107177, 0.0216336, -0.0530087, 0.0415348
2: -0.0159430, 0.1019839, 0.0030754, 0.0399569, -0.0558999, 0.0989085
3: -0.0529631, 0.0394997, -0.0325018, 0.0206539, -0.0736170, 0.0720015
4: -0.0574184, 0.0159635, -0.0188807, 0.0118890, -0.0693074, 0.0348442
5: -0.0197886, 0.0277936, -0.0029819, 0.0217282, -0.0415168, 0.0307755
6: -0.0170941, 0.0570708, -0.0152682, 0.0367036, -0.0537976, 0.0723390
7: -0.0494355, 0.0270410, -0.0269711, 0.0082170, -0.0576525, 0.0540121
8: 0.8383634, 0.9908019, 0.9081902, 0.9901170, -0.1517535, 0.0826117
9: -0.0243421, 0.1167575, -0.0169583, 0.0316862, -0.0560283, 0.1337158

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1436484, upper bound: 0.1336986
time: 2.17 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438661, upper bound: 0.1338736
time: 1.35 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0470752, 0.0447411, -0.0022434, 0.0240532, -0.0711284, 0.0469845
1: -0.0362609, 0.0334237, -0.0107349, 0.0216789, -0.0579398, 0.0441586
2: -0.0207431, 0.1147738, 0.0030475, 0.0399972, -0.0607403, 0.1117264
3: -0.0584405, 0.0431066, -0.0325491, 0.0207102, -0.0791507, 0.0756557
4: -0.0665866, 0.0169369, -0.0189083, 0.0119164, -0.0785030, 0.0358452
5: -0.0238807, 0.0293077, -0.0029993, 0.0217686, -0.0456493, 0.0323070
6: -0.0188429, 0.0618150, -0.0152734, 0.0367549, -0.0555977, 0.0770884
7: -0.0536168, 0.0316714, -0.0269961, 0.0082304, -0.0618473, 0.0586675
8: 0.8249218, 0.9910174, 0.9080660, 0.9901208, -0.1651990, 0.0829514
9: -0.0270275, 0.1313044, -0.0169712, 0.0317543, -0.0587818, 0.1482756

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1435784, upper bound: 0.1337013
time: 1.48 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1437923, upper bound: 0.1338912
time: 1.35 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0400207, 0.0401965, -0.0019909, 0.0226329, -0.0626536, 0.0421874
1: -0.0313751, 0.0308171, -0.0100082, 0.0201454, -0.0515205, 0.0408253
2: -0.0159430, 0.1019839, 0.0039349, 0.0384408, -0.0543838, 0.0980490
3: -0.0529631, 0.0394997, -0.0310514, 0.0185317, -0.0714948, 0.0705511
4: -0.0574184, 0.0159635, -0.0179658, 0.0108103, -0.0682287, 0.0339293
5: -0.0197886, 0.0277936, -0.0025229, 0.0201964, -0.0399850, 0.0303165
6: -0.0170941, 0.0570708, -0.0157195, 0.0347209, -0.0518150, 0.0727902
7: -0.0494355, 0.0270410, -0.0257033, 0.0084083, -0.0578438, 0.0527443
8: 0.8383634, 0.9908019, 0.9135684, 0.9904129, -0.1520495, 0.0772335
9: -0.0243421, 0.1167575, -0.0169037, 0.0290650, -0.0534071, 0.1336612

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1440826, upper bound: 0.1339714
time: 2.53 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1442614, upper bound: 0.1341314
time: 1.81 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0470752, 0.0447411, -0.0019964, 0.0226894, -0.0697646, 0.0467375
1: -0.0362609, 0.0334237, -0.0100311, 0.0202051, -0.0564660, 0.0434548
2: -0.0207431, 0.1147738, 0.0038981, 0.0384942, -0.0592373, 0.1108757
3: -0.0584405, 0.0431066, -0.0311135, 0.0186067, -0.0770472, 0.0742201
4: -0.0665866, 0.0169369, -0.0180019, 0.0108466, -0.0774333, 0.0349388
5: -0.0238807, 0.0293077, -0.0025452, 0.0202503, -0.0441310, 0.0318529
6: -0.0188429, 0.0618150, -0.0157265, 0.0347896, -0.0536324, 0.0775415
7: -0.0536168, 0.0316714, -0.0257375, 0.0084261, -0.0620429, 0.0574090
8: 0.8249218, 0.9910174, 0.9134017, 0.9904178, -0.1654960, 0.0776157
9: -0.0270275, 0.1313044, -0.0169202, 0.0291553, -0.0561827, 0.1482246

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439956, upper bound: 0.1339822
time: 2.44 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1441779, upper bound: 0.1341551
time: 1.41 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0235322, 0.0332591, -0.0022394, 0.0240106, -0.0475428, 0.0354985
1: -0.0221560, 0.0271909, -0.0107177, 0.0216336, -0.0437895, 0.0379086
2: -0.0079077, 0.0733482, 0.0030754, 0.0399569, -0.0478646, 0.0702728
3: -0.0438773, 0.0325672, -0.0325018, 0.0206539, -0.0645312, 0.0650690
4: -0.0405357, 0.0144962, -0.0188807, 0.0118890, -0.0524247, 0.0333770
5: -0.0120661, 0.0256611, -0.0029819, 0.0217282, -0.0337943, 0.0286431
6: -0.0156039, 0.0483375, -0.0152682, 0.0367036, -0.0523075, 0.0636057
7: -0.0403037, 0.0180461, -0.0269711, 0.0082170, -0.0485207, 0.0450173
8: 0.8671969, 0.9909092, 0.9081902, 0.9901170, -0.1229200, 0.0827190
9: -0.0208733, 0.0834956, -0.0169583, 0.0316862, -0.0525595, 0.1004540

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1329908, upper bound: 0.1194889
time: 1.50 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1284707, upper bound: 0.1187933
time: 1.83 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0305649, 0.0369758, -0.0022434, 0.0240532, -0.0546182, 0.0392192
1: -0.0257718, 0.0298844, -0.0107349, 0.0216789, -0.0474507, 0.0406194
2: -0.0115335, 0.0860591, 0.0030475, 0.0399972, -0.0515306, 0.0830116
3: -0.0486584, 0.0362744, -0.0325491, 0.0207102, -0.0693685, 0.0688235
4: -0.0473431, 0.0154978, -0.0189083, 0.0119164, -0.0592595, 0.0344060
5: -0.0155673, 0.0272415, -0.0029993, 0.0217686, -0.0373360, 0.0302409
6: -0.0165835, 0.0525247, -0.0152734, 0.0367549, -0.0533384, 0.0677981
7: -0.0444587, 0.0221003, -0.0269961, 0.0082304, -0.0526891, 0.0490964
8: 0.8538592, 0.9911215, 0.9080660, 0.9901208, -0.1362616, 0.0830555
9: -0.0226267, 0.0984207, -0.0169712, 0.0317543, -0.0543810, 0.1153919

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1436693, upper bound: 0.1341409
time: 1.82 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438730, upper bound: 0.1343098
time: 1.41 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0235322, 0.0332591, -0.0019909, 0.0226329, -0.0461651, 0.0352500
1: -0.0221560, 0.0271909, -0.0100082, 0.0201454, -0.0423014, 0.0371991
2: -0.0079077, 0.0733482, 0.0039349, 0.0384408, -0.0463485, 0.0694133
3: -0.0438773, 0.0325672, -0.0310514, 0.0185317, -0.0624089, 0.0636186
4: -0.0405357, 0.0144962, -0.0179658, 0.0108103, -0.0513460, 0.0324620
5: -0.0120661, 0.0256611, -0.0025229, 0.0201964, -0.0322625, 0.0281841
6: -0.0156039, 0.0483375, -0.0157195, 0.0347209, -0.0503249, 0.0640570
7: -0.0403037, 0.0180461, -0.0257033, 0.0084083, -0.0487120, 0.0437494
8: 0.8671969, 0.9909092, 0.9135684, 0.9904129, -0.1232160, 0.0773408
9: -0.0208733, 0.0834956, -0.0169037, 0.0290650, -0.0499383, 0.1003993

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366126, upper bound: 0.1234512
time: 1.97 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1327352, upper bound: 0.1227844
time: 1.62 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0305649, 0.0369758, -0.0019964, 0.0226894, -0.0532543, 0.0389722
1: -0.0257718, 0.0298844, -0.0100311, 0.0202051, -0.0459769, 0.0399156
2: -0.0115335, 0.0860591, 0.0038981, 0.0384942, -0.0500276, 0.0821610
3: -0.0486584, 0.0362744, -0.0311135, 0.0186067, -0.0672651, 0.0673879
4: -0.0473431, 0.0154978, -0.0180019, 0.0108466, -0.0581898, 0.0334997
5: -0.0155673, 0.0272415, -0.0025452, 0.0202503, -0.0358176, 0.0297868
6: -0.0165835, 0.0525247, -0.0157265, 0.0347896, -0.0513731, 0.0682512
7: -0.0444587, 0.0221003, -0.0257375, 0.0084261, -0.0528848, 0.0478378
8: 0.8538592, 0.9911215, 0.9134017, 0.9904178, -0.1365586, 0.0777198
9: -0.0226267, 0.0984207, -0.0169202, 0.0291553, -0.0517819, 0.1153410

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366345, upper bound: 0.1235315
time: 1.40 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1329995, upper bound: 0.1228835
time: 1.34 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0400207, 0.0401965, -0.0524763, 0.0475461, -0.0875668, 0.0926728
1: -0.0313751, 0.0308171, -0.0398204, 0.0346778, -0.0660529, 0.0706375
2: -0.0159430, 0.1019839, -0.0240675, 0.1240345, -0.1399775, 0.1260514
3: -0.0529631, 0.0394997, -0.0617548, 0.0455119, -0.0984751, 0.1012545
4: -0.0574184, 0.0159635, -0.0731988, 0.0174483, -0.0748666, 0.0891623
5: -0.0197886, 0.0277936, -0.0267742, 0.0300559, -0.0498444, 0.0545678
6: -0.0170941, 0.0570708, -0.0200035, 0.0649085, -0.0820026, 0.0770743
7: -0.0494355, 0.0270410, -0.0567239, 0.0352673, -0.0847028, 0.0837649
8: 0.8383634, 0.9908019, 0.8153460, 0.9911584, -0.1527950, 0.1754559
9: -0.0243421, 0.1167575, -0.0286096, 0.1424822, -0.1668243, 0.1453670

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1408045, upper bound: 0.1341460
time: 1.61 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1345393, upper bound: 0.1333492
time: 1.72 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0470752, 0.0447411, -0.0529757, 0.0478367, -0.0949118, 0.0977169
1: -0.0362609, 0.0334237, -0.0401569, 0.0348267, -0.0710876, 0.0735806
2: -0.0207431, 0.1147738, -0.0243901, 0.1249149, -0.1456580, 0.1391639
3: -0.0584405, 0.0431066, -0.0621019, 0.0457487, -0.1041892, 0.1052085
4: -0.0665866, 0.0169369, -0.0738265, 0.0175057, -0.0840923, 0.0907634
5: -0.0238807, 0.0293077, -0.0270521, 0.0301428, -0.0540236, 0.0563598
6: -0.0188429, 0.0618150, -0.0201235, 0.0652186, -0.0840615, 0.0819385
7: -0.0536168, 0.0316714, -0.0570138, 0.0356069, -0.0892238, 0.0886853
8: 0.8249218, 0.9910174, 0.8144326, 0.9911709, -0.1662492, 0.1765848
9: -0.0270275, 0.1313044, -0.0287792, 0.1435056, -0.1705331, 0.1600836

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471016, upper bound: 0.1449427
time: 1.57 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471016, upper bound: 0.1449427
time: 1.43 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0235322, 0.0332591, -0.0524763, 0.0475461, -0.0710783, 0.0857355
1: -0.0221560, 0.0271909, -0.0398204, 0.0346778, -0.0568337, 0.0670113
2: -0.0079077, 0.0733482, -0.0240675, 0.1240345, -0.1319422, 0.0974156
3: -0.0438773, 0.0325672, -0.0617548, 0.0455119, -0.0893892, 0.0943220
4: -0.0405357, 0.0144962, -0.0731988, 0.0174483, -0.0579840, 0.0876951
5: -0.0120661, 0.0256611, -0.0267742, 0.0300559, -0.0421220, 0.0524353
6: -0.0156039, 0.0483375, -0.0200035, 0.0649085, -0.0805125, 0.0683410
7: -0.0403037, 0.0180461, -0.0567239, 0.0352673, -0.0755710, 0.0747700
8: 0.8671969, 0.9909092, 0.8153460, 0.9911584, -0.1239614, 0.1755632
9: -0.0208733, 0.0834956, -0.0286096, 0.1424822, -0.1633555, 0.1121052

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372310, upper bound: 0.1393612
time: 1.31 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363592, upper bound: 0.1338939
time: 1.42 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0305649, 0.0369758, -0.0529757, 0.0478367, -0.0784016, 0.0899516
1: -0.0257718, 0.0298844, -0.0401569, 0.0348267, -0.0605985, 0.0700414
2: -0.0115335, 0.0860591, -0.0243901, 0.1249149, -0.1364483, 0.1104492
3: -0.0486584, 0.0362744, -0.0621019, 0.0457487, -0.0944071, 0.0983763
4: -0.0473431, 0.0154978, -0.0738265, 0.0175057, -0.0648488, 0.0893243
5: -0.0155673, 0.0272415, -0.0270521, 0.0301428, -0.0457102, 0.0542936
6: -0.0165835, 0.0525247, -0.0201235, 0.0652186, -0.0818021, 0.0726481
7: -0.0444587, 0.0221003, -0.0570138, 0.0356069, -0.0800656, 0.0791142
8: 0.8538592, 0.9911215, 0.8144326, 0.9911709, -0.1373118, 0.1766888
9: -0.0226267, 0.0984207, -0.0287792, 0.1435056, -0.1661323, 0.1272000

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1474488, upper bound: 0.1453894
time: 1.38 seconds

## Relational analysis of NS_B2_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1474488, upper bound: 0.1453894
time: 2.05 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0524763, 0.0475461, -0.0235322, 0.0332591, -0.0857355, 0.0710783
1: -0.0398204, 0.0346778, -0.0221560, 0.0271909, -0.0670113, 0.0568337
2: -0.0240675, 0.1240345, -0.0079077, 0.0733482, -0.0974156, 0.1319422
3: -0.0617548, 0.0455119, -0.0438773, 0.0325672, -0.0943220, 0.0893892
4: -0.0731988, 0.0174483, -0.0405357, 0.0144962, -0.0876951, 0.0579840
5: -0.0267742, 0.0300559, -0.0120661, 0.0256611, -0.0524353, 0.0421220
6: -0.0200035, 0.0649085, -0.0156039, 0.0483375, -0.0683410, 0.0805125
7: -0.0567239, 0.0352673, -0.0403037, 0.0180461, -0.0747700, 0.0755710
8: 0.8153460, 0.9911584, 0.8671969, 0.9909092, -0.1755632, 0.1239614
9: -0.0286096, 0.1424822, -0.0208733, 0.0834956, -0.1121052, 0.1633555

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1415146, upper bound: 0.1356230
time: 1.54 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1350450, upper bound: 0.1345889
time: 1.67 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0529757, 0.0478367, -0.0305649, 0.0369758, -0.0899516, 0.0784016
1: -0.0401569, 0.0348267, -0.0257718, 0.0298844, -0.0700414, 0.0605985
2: -0.0243901, 0.1249149, -0.0115335, 0.0860591, -0.1104492, 0.1364483
3: -0.0621019, 0.0457487, -0.0486584, 0.0362744, -0.0983763, 0.0944071
4: -0.0738265, 0.0175057, -0.0473431, 0.0154978, -0.0893243, 0.0648488
5: -0.0270521, 0.0301428, -0.0155673, 0.0272415, -0.0542936, 0.0457102
6: -0.0201235, 0.0652186, -0.0165835, 0.0525247, -0.0726481, 0.0818021
7: -0.0570138, 0.0356069, -0.0444587, 0.0221003, -0.0791142, 0.0800656
8: 0.8144326, 0.9911709, 0.8538592, 0.9911215, -0.1766888, 0.1373118
9: -0.0287792, 0.1435056, -0.0226267, 0.0984207, -0.1272000, 0.1661323

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473438, upper bound: 0.1450693
time: 1.87 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473438, upper bound: 0.1450822
time: 1.89 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0235322, 0.0332591, -0.0361535, 0.0393951, -0.0629273, 0.0694127
1: -0.0221560, 0.0271909, -0.0290321, 0.0310964, -0.0532524, 0.0562230
2: -0.0079077, 0.0733482, -0.0144299, 0.0956708, -0.1035786, 0.0877781
3: -0.0438773, 0.0325672, -0.0516926, 0.0386126, -0.0824899, 0.0842599
4: -0.0405357, 0.0144962, -0.0532982, 0.0159908, -0.0565265, 0.0677944
5: -0.0120661, 0.0256611, -0.0182603, 0.0279541, -0.0400202, 0.0439214
6: -0.0156039, 0.0483375, -0.0174481, 0.0555105, -0.0711145, 0.0657856
7: -0.0403037, 0.0180461, -0.0475553, 0.0252021, -0.0655058, 0.0656015
8: 0.8671969, 0.9909092, 0.8442402, 0.9912613, -0.1240644, 0.1466690
9: -0.0208733, 0.0834956, -0.0240001, 0.1094263, -0.1302996, 0.1074957

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 175

## Relational analysis of NS_B2_A2_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1428517, upper bound: 0.1368959
time: 1.29 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375849, upper bound: 0.1361499
time: 1.75 seconds

## BFS NS instance: NS_B2_A2_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0305649, 0.0369758, -0.0366432, 0.0396761, -0.0702410, 0.0736190
1: -0.0257718, 0.0298844, -0.0293602, 0.0312437, -0.0570155, 0.0592447
2: -0.0115335, 0.0860591, -0.0147429, 0.0965330, -0.1080664, 0.1008019
3: -0.0486584, 0.0362744, -0.0520285, 0.0388444, -0.0875027, 0.0883030
4: -0.0473431, 0.0154978, -0.0539100, 0.0160473, -0.0633904, 0.0694078
5: -0.0155673, 0.0272415, -0.0185309, 0.0280405, -0.0436079, 0.0457725
6: -0.0165835, 0.0525247, -0.0175293, 0.0558140, -0.0723975, 0.0700540
7: -0.0444587, 0.0221003, -0.0478392, 0.0254996, -0.0699584, 0.0699395
8: 0.8538592, 0.9911215, 0.8433452, 0.9912735, -0.1374143, 0.1477763
9: -0.0226267, 0.0984207, -0.0241571, 0.1104309, -0.1330576, 0.1225778

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A2_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1480847, upper bound: 0.1459901
time: 1.68 seconds

## Relational analysis of NS_B2_A2_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1480847, upper bound: 0.1459901
time: 1.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 5.28 seconds
NS_B2_A1_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1219993, upper bound: 0.1379094
NS_B2_A1_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1222056, upper bound: 0.1380821
NS_B2_A1_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1220034, upper bound: 0.1377542
NS_B2_A1_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1222156, upper bound: 0.1379473
NS_B2_A1_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1222130, upper bound: 0.1382363
NS_B2_A1_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1223969, upper bound: 0.1384036
NS_B2_A1_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1222204, upper bound: 0.1380858
NS_B2_A1_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1224036, upper bound: 0.1382508
NS_B2_A1_B2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1302950, upper bound: 0.1410337
NS_B2_A1_B2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1305037, upper bound: 0.1412404
NS_B2_A1_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1302965, upper bound: 0.1408911
NS_B2_A1_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1305144, upper bound: 0.1411386
NS_B2_A1_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1303332, upper bound: 0.1412018
NS_B2_A1_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1305492, upper bound: 0.1414092
NS_B2_A1_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1303345, upper bound: 0.1410613
NS_B2_A1_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1305546, upper bound: 0.1412839
NS_B2_A1_B2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1224619, upper bound: 0.1377987
NS_B2_A1_B2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1227176, upper bound: 0.1379746
NS_B2_A1_B2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1222866, upper bound: 0.1377996
NS_B2_A1_B2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1225370, upper bound: 0.1379788
NS_B2_A1_B2_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1130602, upper bound: 0.1303159
NS_B2_A1_B2_B2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1123090, upper bound: 0.1271023
NS_B2_A1_B2_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1130577, upper bound: 0.1304177
NS_B2_A1_B2_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1123078, upper bound: 0.1271130
NS_B2_A1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1305492, upper bound: 0.1411006
NS_B2_A1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1307681, upper bound: 0.1412868
NS_B2_A1_B2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1305508, upper bound: 0.1409683
NS_B2_A1_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1307813, upper bound: 0.1411806
NS_B2_A1_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1306289, upper bound: 0.1415561
NS_B2_A1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1308391, upper bound: 0.1417490
NS_B2_A1_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1306289, upper bound: 0.1414100
NS_B2_A1_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1308487, upper bound: 0.1416182
NS_B2_A2_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1347767, upper bound: 0.1403786
NS_B2_A2_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1349608, upper bound: 0.1406007
NS_B2_A2_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1347767, upper bound: 0.1402267
NS_B2_A2_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1404863
NS_B2_A2_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1352277, upper bound: 0.1411910
NS_B2_A2_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1353826, upper bound: 0.1413783
NS_B2_A2_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1352277, upper bound: 0.1410659
NS_B2_A2_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1353880, upper bound: 0.1412669
NS_B2_A2_A1_B2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1199217, upper bound: 0.1288912
NS_B2_A2_A1_B2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1190971, upper bound: 0.1255403
NS_B2_A2_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1350662, upper bound: 0.1402405
NS_B2_A2_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1352903, upper bound: 0.1404998
NS_B2_A2_A1_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1239921, upper bound: 0.1330441
NS_B2_A2_A1_B2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1232609, upper bound: 0.1300494
NS_B2_A2_A1_B2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1240446, upper bound: 0.1330424
NS_B2_A2_A1_B2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1233291, upper bound: 0.1302358
NS_B2_A2_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1436484, upper bound: 0.1336986
NS_B2_A2_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1438661, upper bound: 0.1338736
NS_B2_A2_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1435784, upper bound: 0.1337013
NS_B2_A2_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1437923, upper bound: 0.1338912
NS_B2_A2_A2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1440826, upper bound: 0.1339714
NS_B2_A2_A2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1442614, upper bound: 0.1341314
NS_B2_A2_A2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1439956, upper bound: 0.1339822
NS_B2_A2_A2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1441779, upper bound: 0.1341551
NS_B2_A2_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1329908, upper bound: 0.1194889
NS_B2_A2_A2_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1284707, upper bound: 0.1187933
NS_B2_A2_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1436693, upper bound: 0.1341409
NS_B2_A2_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1438730, upper bound: 0.1343098
NS_B2_A2_A2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1366126, upper bound: 0.1234512
NS_B2_A2_A2_B1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1327352, upper bound: 0.1227844
NS_B2_A2_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1366345, upper bound: 0.1235315
NS_B2_A2_A2_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1329995, upper bound: 0.1228835
NS_B2_A2_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1408045, upper bound: 0.1341460
NS_B2_A2_A2_B2_B1_A1_A1_A2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1345393, upper bound: 0.1333492
NS_B2_A2_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1471016, upper bound: 0.1449427
NS_B2_A2_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1471016, upper bound: 0.1449427
NS_B2_A2_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1372310, upper bound: 0.1393612
NS_B2_A2_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1363592, upper bound: 0.1338939
NS_B2_A2_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1474488, upper bound: 0.1453894
NS_B2_A2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1474488, upper bound: 0.1453894
NS_B2_A2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1415146, upper bound: 0.1356230
NS_B2_A2_A2_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1350450, upper bound: 0.1345889
NS_B2_A2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1473438, upper bound: 0.1450693
NS_B2_A2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1473438, upper bound: 0.1450822
NS_B2_A2_A2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1428517, upper bound: 0.1368959
NS_B2_A2_A2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1375849, upper bound: 0.1361499
NS_B2_A2_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1480847, upper bound: 0.1459901
NS_B2_A2_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.28
Output dim: 8, lower bound: -0.1480847, upper bound: 0.1459901

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0008445, 0.0196644, -0.0344647, 0.0369358, -0.0377802, 0.0541291
1: -0.0079262, 0.0171530, -0.0276318, 0.0290732, -0.0369994, 0.0447848
2: 0.0057466, 0.0345457, -0.0123552, 0.0920904, -0.0863438, 0.0469010
3: -0.0288320, 0.0129648, -0.0490528, 0.0368465, -0.0656785, 0.0620176
4: -0.0160813, 0.0076267, -0.0504141, 0.0152957, -0.0313770, 0.0580408
5: -0.0015185, 0.0159950, -0.0166910, 0.0267636, -0.0282821, 0.0326861
6: -0.0145175, 0.0294798, -0.0159031, 0.0535622, -0.0680797, 0.0453829
7: -0.0206520, 0.0069385, -0.0461851, 0.0236088, -0.0442608, 0.0531236
8: 0.9305482, 0.9897312, 0.8486274, 0.9906526, -0.0601044, 0.1411038
9: -0.0168900, 0.0218748, -0.0225825, 0.1053489, -0.1222389, 0.0444573

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219993, upper bound: 0.1379094
time: 1.74 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219993, upper bound: 0.1379094
time: 1.77 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0008585, 0.0197864, -0.0374161, 0.0386484, -0.0395070, 0.0572025
1: -0.0079790, 0.0172501, -0.0296055, 0.0299873, -0.0379663, 0.0468556
2: 0.0056997, 0.0346689, -0.0142365, 0.0973543, -0.0916546, 0.0489054
3: -0.0289772, 0.0131195, -0.0511066, 0.0382320, -0.0672092, 0.0642261
4: -0.0161520, 0.0077095, -0.0541071, 0.0156445, -0.0317965, 0.0618167
5: -0.0015731, 0.0160486, -0.0183219, 0.0273059, -0.0288790, 0.0343705
6: -0.0146054, 0.0296341, -0.0164864, 0.0554174, -0.0700228, 0.0461205
7: -0.0207292, 0.0069914, -0.0479048, 0.0253524, -0.0460816, 0.0548962
8: 0.9301687, 0.9897363, 0.8432024, 0.9907125, -0.0605438, 0.1465339
9: -0.0169335, 0.0220409, -0.0234372, 0.1113705, -0.1283040, 0.0454781

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222056, upper bound: 0.1380821
time: 1.78 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222056, upper bound: 0.1380821
time: 1.76 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0008481, 0.0197013, -0.0406593, 0.0410263, -0.0418744, 0.0603606
1: -0.0079423, 0.0171829, -0.0319358, 0.0314684, -0.0394107, 0.0491187
2: 0.0057327, 0.0345835, -0.0166527, 0.1032727, -0.0975400, 0.0512362
3: -0.0288769, 0.0130119, -0.0539511, 0.0401111, -0.0689880, 0.0669630
4: -0.0161028, 0.0076516, -0.0585825, 0.0161742, -0.0322770, 0.0662341
5: -0.0015343, 0.0160115, -0.0203479, 0.0281691, -0.0297034, 0.0363594
6: -0.0145230, 0.0295273, -0.0173086, 0.0577686, -0.0722916, 0.0468359
7: -0.0206745, 0.0069445, -0.0499274, 0.0276995, -0.0483740, 0.0568720
8: 0.9304334, 0.9897315, 0.8366903, 0.9908466, -0.0604132, 0.1530412
9: -0.0169034, 0.0219230, -0.0248578, 0.1185465, -0.1354499, 0.0467808

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219993, upper bound: 0.1377536
time: 1.93 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1219993, upper bound: 0.1377542
time: 1.70 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0008622, 0.0198235, -0.0442646, 0.0430593, -0.0439215, 0.0640882
1: -0.0079952, 0.0172803, -0.0343498, 0.0325288, -0.0405241, 0.0516300
2: 0.0056857, 0.0347070, -0.0188998, 0.1097855, -0.1040998, 0.0536068
3: -0.0290225, 0.0131670, -0.0564307, 0.0417345, -0.0707569, 0.0695977
4: -0.0161737, 0.0077346, -0.0630077, 0.0165930, -0.0327668, 0.0707423
5: -0.0015890, 0.0160652, -0.0222891, 0.0287825, -0.0303715, 0.0383543
6: -0.0146109, 0.0296820, -0.0181096, 0.0600264, -0.0746373, 0.0477916
7: -0.0207519, 0.0069973, -0.0519612, 0.0297876, -0.0505395, 0.0589585
8: 0.9300530, 0.9897366, 0.8301442, 0.9909204, -0.0608674, 0.1595923
9: -0.0169470, 0.0220895, -0.0260178, 0.1254783, -0.1424253, 0.0481073

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222056, upper bound: 0.1379433
time: 1.72 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2_B2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222056, upper bound: 0.1379473
time: 1.56 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0006536, 0.0185650, -0.0344647, 0.0369358, -0.0375893, 0.0530297
1: -0.0073557, 0.0162048, -0.0276318, 0.0290732, -0.0364289, 0.0438366
2: 0.0061261, 0.0332928, -0.0123552, 0.0920904, -0.0859643, 0.0456480
3: -0.0275467, 0.0112977, -0.0490528, 0.0368465, -0.0643932, 0.0603505
4: -0.0154323, 0.0067368, -0.0504141, 0.0152957, -0.0307280, 0.0571508
5: -0.0011785, 0.0153697, -0.0166910, 0.0267636, -0.0279421, 0.0320608
6: -0.0150842, 0.0278528, -0.0159031, 0.0535622, -0.0686464, 0.0437559
7: -0.0196152, 0.0075182, -0.0461851, 0.0236088, -0.0432240, 0.0537033
8: 0.9348881, 0.9901945, 0.8486274, 0.9906526, -0.0557646, 0.1415671
9: -0.0168587, 0.0201396, -0.0225825, 0.1053489, -0.1222076, 0.0427220

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222130, upper bound: 0.1382363
time: 2.17 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222130, upper bound: 0.1382363
time: 1.92 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0006668, 0.0186794, -0.0374161, 0.0386484, -0.0393152, 0.0560955
1: -0.0074052, 0.0162957, -0.0296055, 0.0299873, -0.0373925, 0.0459012
2: 0.0060817, 0.0334081, -0.0142365, 0.0973543, -0.0912726, 0.0476447
3: -0.0276826, 0.0114427, -0.0511066, 0.0382320, -0.0659146, 0.0625493
4: -0.0154985, 0.0068147, -0.0541071, 0.0156445, -0.0311430, 0.0609219
5: -0.0012282, 0.0154201, -0.0183219, 0.0273059, -0.0285341, 0.0337420
6: -0.0151734, 0.0279972, -0.0164864, 0.0554174, -0.0705908, 0.0444836
7: -0.0196879, 0.0075711, -0.0479048, 0.0253524, -0.0450403, 0.0554759
8: 0.9345318, 0.9902000, 0.8432024, 0.9907125, -0.0561807, 0.1469977
9: -0.0168997, 0.0202934, -0.0234372, 0.1113705, -0.1282702, 0.0437306

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1223969, upper bound: 0.1384036
time: 1.64 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1223969, upper bound: 0.1384036
time: 1.39 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0006572, 0.0186016, -0.0406593, 0.0410263, -0.0416836, 0.0592610
1: -0.0073718, 0.0162347, -0.0319358, 0.0314684, -0.0388402, 0.0481704
2: 0.0061122, 0.0333304, -0.0166527, 0.1032727, -0.0971605, 0.0499831
3: -0.0275910, 0.0113452, -0.0539511, 0.0401111, -0.0677021, 0.0652964
4: -0.0154536, 0.0067618, -0.0585825, 0.0161742, -0.0316278, 0.0653443
5: -0.0011932, 0.0153863, -0.0203479, 0.0281691, -0.0293623, 0.0357342
6: -0.0150907, 0.0279003, -0.0173086, 0.0577686, -0.0728593, 0.0452089
7: -0.0196378, 0.0075244, -0.0499274, 0.0276995, -0.0473373, 0.0574518
8: 0.9347728, 0.9901949, 0.8366903, 0.9908466, -0.0560737, 0.1535046
9: -0.0168720, 0.0201866, -0.0248578, 0.1185465, -0.1354185, 0.0450444

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222130, upper bound: 0.1380855
time: 2.13 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222130, upper bound: 0.1380858
time: 1.54 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0006705, 0.0187162, -0.0442646, 0.0430593, -0.0437298, 0.0629809
1: -0.0074214, 0.0163257, -0.0343498, 0.0325288, -0.0399502, 0.0506755
2: 0.0060677, 0.0334460, -0.0188998, 0.1097855, -0.1037178, 0.0523458
3: -0.0277272, 0.0114904, -0.0564307, 0.0417345, -0.0694616, 0.0679211
4: -0.0155200, 0.0068400, -0.0630077, 0.0165930, -0.0321130, 0.0698476
5: -0.0012430, 0.0154367, -0.0222891, 0.0287825, -0.0300255, 0.0377258
6: -0.0151799, 0.0280449, -0.0181096, 0.0600264, -0.0752063, 0.0461545
7: -0.0197107, 0.0075772, -0.0519612, 0.0297876, -0.0494983, 0.0595384
8: 0.9344159, 0.9902003, 0.8301442, 0.9909204, -0.0565045, 0.1600561
9: -0.0169131, 0.0203408, -0.0260178, 0.1254783, -0.1423914, 0.0463586

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1223969, upper bound: 0.1382508
time: 1.64 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1223969, upper bound: 0.1382508
time: 1.46 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0018116, 0.0250305, -0.0344647, 0.0369358, -0.0387473, 0.0594952
1: -0.0105778, 0.0227850, -0.0276318, 0.0290732, -0.0396511, 0.0504168
2: 0.0022700, 0.0402259, -0.0123552, 0.0920904, -0.0898204, 0.0525812
3: -0.0343236, 0.0208084, -0.0490528, 0.0368465, -0.0711701, 0.0698612
4: -0.0195640, 0.0116928, -0.0504141, 0.0152957, -0.0348596, 0.0621068
5: -0.0035712, 0.0218832, -0.0166910, 0.0267636, -0.0303348, 0.0385742
6: -0.0146094, 0.0367258, -0.0159031, 0.0535622, -0.0681715, 0.0526289
7: -0.0255142, 0.0082131, -0.0461851, 0.0236088, -0.0491229, 0.0543982
8: 0.9105793, 0.9901145, 0.8486274, 0.9906526, -0.0800733, 0.1414871
9: -0.0180809, 0.0317093, -0.0225825, 0.1053489, -0.1234298, 0.0542918

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1191620, upper bound: 0.1310569
time: 1.50 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1171437, upper bound: 0.1254592
time: 1.14 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0018446, 0.0253376, -0.0374161, 0.0386484, -0.0404930, 0.0627536
1: -0.0107035, 0.0231011, -0.0296055, 0.0299873, -0.0406908, 0.0527066
2: 0.0020702, 0.0405154, -0.0142365, 0.0973543, -0.0952842, 0.0547519
3: -0.0346573, 0.0212074, -0.0511066, 0.0382320, -0.0728894, 0.0723140
4: -0.0197595, 0.0118896, -0.0541071, 0.0156445, -0.0354041, 0.0659967
5: -0.0037002, 0.0221692, -0.0183219, 0.0273059, -0.0310061, 0.0404911
6: -0.0147167, 0.0370870, -0.0164864, 0.0554174, -0.0701341, 0.0535735
7: -0.0257043, 0.0083316, -0.0479048, 0.0253524, -0.0510567, 0.0562364
8: 0.9096730, 0.9901440, 0.8432024, 0.9907125, -0.0810395, 0.1469417
9: -0.0181753, 0.0322051, -0.0234372, 0.1113705, -0.1295458, 0.0556423

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1196908, upper bound: 0.1318666
time: 1.33 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1189056, upper bound: 0.1281031
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0018184, 0.0250992, -0.0406593, 0.0410263, -0.0428447, 0.0657586
1: -0.0106059, 0.0228562, -0.0319358, 0.0314684, -0.0420744, 0.0547920
2: 0.0022251, 0.0402907, -0.0166527, 0.1032727, -0.1010476, 0.0569434
3: -0.0343985, 0.0208982, -0.0539511, 0.0401111, -0.0745096, 0.0748493
4: -0.0196078, 0.0117366, -0.0585825, 0.0161742, -0.0357820, 0.0703191
5: -0.0035999, 0.0219476, -0.0203479, 0.0281691, -0.0317690, 0.0422955
6: -0.0146205, 0.0368069, -0.0173086, 0.0577686, -0.0723891, 0.0541156
7: -0.0255555, 0.0082361, -0.0499274, 0.0276995, -0.0532550, 0.0581635
8: 0.9103777, 0.9901208, 0.8366903, 0.9908466, -0.0804689, 0.1534305
9: -0.0181019, 0.0318193, -0.0248578, 0.1185465, -0.1366484, 0.0566771

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1191583, upper bound: 0.1310150
time: 2.10 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1171208, upper bound: 0.1254598
time: 1.34 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0018514, 0.0254070, -0.0442646, 0.0430593, -0.0449108, 0.0696716
1: -0.0107318, 0.0231729, -0.0343498, 0.0325288, -0.0432606, 0.0575227
2: 0.0020248, 0.0405808, -0.0188998, 0.1097855, -0.1077607, 0.0594807
3: -0.0347329, 0.0212980, -0.0564307, 0.0417345, -0.0764674, 0.0777287
4: -0.0198039, 0.0119339, -0.0630077, 0.0165930, -0.0363969, 0.0749416
5: -0.0037292, 0.0222341, -0.0222891, 0.0287825, -0.0325117, 0.0445232
6: -0.0147280, 0.0371689, -0.0181096, 0.0600264, -0.0747544, 0.0552785
7: -0.0257461, 0.0083547, -0.0519612, 0.0297876, -0.0555337, 0.0603159
8: 0.9094694, 0.9901504, 0.8301442, 0.9909204, -0.0814511, 0.1600062
9: -0.0181965, 0.0323164, -0.0260178, 0.1254783, -0.1436747, 0.0583342

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1197030, upper bound: 0.1318650
time: 1.33 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1189056, upper bound: 0.1281048
time: 1.76 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0015939, 0.0235713, -0.0344647, 0.0369358, -0.0385297, 0.0580361
1: -0.0098501, 0.0212225, -0.0276318, 0.0290732, -0.0389233, 0.0488543
2: 0.0031875, 0.0386447, -0.0123552, 0.0920904, -0.0889029, 0.0509999
3: -0.0327384, 0.0186572, -0.0490528, 0.0368465, -0.0695849, 0.0677100
4: -0.0185900, 0.0105802, -0.0504141, 0.0152957, -0.0338857, 0.0609942
5: -0.0030540, 0.0203049, -0.0166910, 0.0267636, -0.0298176, 0.0369960
6: -0.0150916, 0.0347208, -0.0159031, 0.0535622, -0.0686538, 0.0506239
7: -0.0243457, 0.0084422, -0.0461851, 0.0236088, -0.0479544, 0.0546273
8: 0.9160390, 0.9904495, 0.8486274, 0.9906526, -0.0746136, 0.1418221
9: -0.0179855, 0.0290031, -0.0225825, 0.1053489, -0.1233344, 0.0515856

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1207342, upper bound: 0.1326297
time: 1.35 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1182663, upper bound: 0.1269597
time: 1.17 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0016254, 0.0238650, -0.0374161, 0.0386484, -0.0402738, 0.0612811
1: -0.0099695, 0.0215255, -0.0296055, 0.0299873, -0.0399569, 0.0511310
2: 0.0029962, 0.0389208, -0.0142365, 0.0973543, -0.0943581, 0.0531573
3: -0.0330595, 0.0190400, -0.0511066, 0.0382320, -0.0712916, 0.0701466
4: -0.0187776, 0.0107678, -0.0541071, 0.0156445, -0.0344221, 0.0648749
5: -0.0031760, 0.0205795, -0.0183219, 0.0273059, -0.0304818, 0.0389014
6: -0.0151982, 0.0350693, -0.0164864, 0.0554174, -0.0706156, 0.0515558
7: -0.0245275, 0.0085588, -0.0479048, 0.0253524, -0.0498799, 0.0564636
8: 0.9151753, 0.9904779, 0.8432024, 0.9907125, -0.0755372, 0.1472756
9: -0.0180765, 0.0294723, -0.0234372, 0.1113705, -0.1294469, 0.0529095

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1211660, upper bound: 0.1333120
time: 1.60 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1201829, upper bound: 0.1292454
time: 1.32 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0016006, 0.0236395, -0.0406593, 0.0410263, -0.0426269, 0.0642989
1: -0.0098777, 0.0212934, -0.0319358, 0.0314684, -0.0413461, 0.0532292
2: 0.0031430, 0.0387089, -0.0166527, 0.1032727, -0.1001297, 0.0553616
3: -0.0328135, 0.0187464, -0.0539511, 0.0401111, -0.0729246, 0.0726976
4: -0.0186338, 0.0106234, -0.0585825, 0.0161742, -0.0348080, 0.0692059
5: -0.0030821, 0.0203689, -0.0203479, 0.0281691, -0.0312512, 0.0407169
6: -0.0151033, 0.0348022, -0.0173086, 0.0577686, -0.0728718, 0.0521108
7: -0.0243865, 0.0084649, -0.0499274, 0.0276995, -0.0520860, 0.0583923
8: 0.9158401, 0.9904556, 0.8366903, 0.9908466, -0.0750064, 0.1537653
9: -0.0180065, 0.0291108, -0.0248578, 0.1185465, -0.1365529, 0.0539686

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1207343, upper bound: 0.1325885
time: 1.31 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1182520, upper bound: 0.1269597
time: 1.26 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0016321, 0.0239339, -0.0442646, 0.0430593, -0.0446915, 0.0681985
1: -0.0099975, 0.0215972, -0.0343498, 0.0325288, -0.0425263, 0.0559469
2: 0.0029513, 0.0389856, -0.0188998, 0.1097855, -0.1068342, 0.0578855
3: -0.0331353, 0.0191305, -0.0564307, 0.0417345, -0.0748698, 0.0755612
4: -0.0188218, 0.0108115, -0.0630077, 0.0165930, -0.0354148, 0.0738192
5: -0.0032044, 0.0206443, -0.0222891, 0.0287825, -0.0319869, 0.0429334
6: -0.0152098, 0.0351517, -0.0181096, 0.0600264, -0.0752362, 0.0532613
7: -0.0245690, 0.0085817, -0.0519612, 0.0297876, -0.0543566, 0.0605429
8: 0.9149743, 0.9904841, 0.8301442, 0.9909204, -0.0759462, 0.1603398
9: -0.0180976, 0.0295814, -0.0260178, 0.1254783, -0.1435759, 0.0555992

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1211772, upper bound: 0.1333096
time: 1.28 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1201849, upper bound: 0.1292666
time: 1.36 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0007108, 0.0184066, -0.0284350, 0.0354714, -0.0361822, 0.0468416
1: -0.0073812, 0.0161425, -0.0246200, 0.0287076, -0.0360888, 0.0407625
2: 0.0062285, 0.0332702, -0.0102784, 0.0818910, -0.0756625, 0.0435486
3: -0.0273192, 0.0113648, -0.0467897, 0.0349469, -0.0622661, 0.0581545
4: -0.0153522, 0.0067747, -0.0450997, 0.0150724, -0.0304246, 0.0518744
5: -0.0009780, 0.0154386, -0.0143681, 0.0265485, -0.0275265, 0.0298066
6: -0.0140415, 0.0278816, -0.0161225, 0.0510076, -0.0650491, 0.0440041
7: -0.0198714, 0.0066141, -0.0430755, 0.0208416, -0.0407130, 0.0496897
8: 0.9344603, 0.9897102, 0.8583730, 0.9910616, -0.0566013, 0.1313372
9: -0.0164280, 0.0202060, -0.0220514, 0.0935769, -0.1100049, 0.0422574

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224619, upper bound: 0.1377987
time: 1.48 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224619, upper bound: 0.1377987
time: 1.50 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0007255, 0.0185341, -0.0309649, 0.0366758, -0.0374014, 0.0494990
1: -0.0074363, 0.0162442, -0.0259040, 0.0295298, -0.0369660, 0.0421482
2: 0.0061794, 0.0333988, -0.0114623, 0.0865121, -0.0803328, 0.0448611
3: -0.0274709, 0.0115269, -0.0483241, 0.0361448, -0.0636157, 0.0598509
4: -0.0154255, 0.0068615, -0.0474761, 0.0153866, -0.0308122, 0.0543376
5: -0.0010335, 0.0154944, -0.0155637, 0.0270338, -0.0280673, 0.0310582
6: -0.0141299, 0.0280419, -0.0165459, 0.0524446, -0.0665744, 0.0445878
7: -0.0199523, 0.0066679, -0.0445603, 0.0221032, -0.0420554, 0.0512282
8: 0.9340628, 0.9897153, 0.8537089, 0.9911139, -0.0570511, 0.1360064
9: -0.0164730, 0.0203785, -0.0225224, 0.0987514, -0.1152244, 0.0429009

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1227176, upper bound: 0.1379746
time: 4.72 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1227176, upper bound: 0.1379746
time: 1.90 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0008285, 0.0198145, -0.0286755, 0.0355844, -0.0364130, 0.0484901
1: -0.0079606, 0.0173092, -0.0247421, 0.0287855, -0.0367461, 0.0420513
2: 0.0056907, 0.0346785, -0.0103954, 0.0823183, -0.0766276, 0.0450739
3: -0.0290730, 0.0131103, -0.0469357, 0.0350644, -0.0641374, 0.0600459
4: -0.0161713, 0.0076718, -0.0453264, 0.0151019, -0.0312732, 0.0529982
5: -0.0015693, 0.0160078, -0.0144832, 0.0265941, -0.0281634, 0.0304910
6: -0.0142331, 0.0296163, -0.0161557, 0.0511419, -0.0653750, 0.0457720
7: -0.0206350, 0.0068198, -0.0432145, 0.0209755, -0.0416105, 0.0500344
8: 0.9304348, 0.9897023, 0.8579325, 0.9910684, -0.0606336, 0.1317698
9: -0.0169212, 0.0219735, -0.0221056, 0.0940748, -0.1109960, 0.0440792

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222814, upper bound: 0.1377996
time: 2.77 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222814, upper bound: 0.1377996
time: 1.70 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0008433, 0.0199433, -0.0314061, 0.0368772, -0.0377204, 0.0513493
1: -0.0080161, 0.0174116, -0.0261264, 0.0296663, -0.0376823, 0.0435380
2: 0.0056414, 0.0348078, -0.0116746, 0.0872909, -0.0816496, 0.0464824
3: -0.0292257, 0.0132736, -0.0485868, 0.0363556, -0.0655813, 0.0618604
4: -0.0162456, 0.0077591, -0.0478882, 0.0154387, -0.0316843, 0.0556473
5: -0.0016268, 0.0160642, -0.0157726, 0.0271139, -0.0287407, 0.0318367
6: -0.0143212, 0.0297787, -0.0166032, 0.0526870, -0.0670082, 0.0463820
7: -0.0207167, 0.0068729, -0.0448126, 0.0223535, -0.0430702, 0.0516855
8: 0.9300352, 0.9897074, 0.8529109, 0.9911251, -0.0610899, 0.1367965
9: -0.0169669, 0.0221489, -0.0226195, 0.0996570, -0.1166239, 0.0447684

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225202, upper bound: 0.1379788
time: 1.43 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225202, upper bound: 0.1379788
time: 1.53 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0018116, 0.0250305, -0.0196186, 0.0313761, -0.0331877, 0.0446491
1: -0.0105778, 0.0227850, -0.0201462, 0.0259039, -0.0364818, 0.0429312
2: 0.0022700, 0.0402259, -0.0060397, 0.0662618, -0.0639918, 0.0462657
3: -0.0343236, 0.0208084, -0.0415117, 0.0306548, -0.0649783, 0.0623202
4: -0.0195640, 0.0116928, -0.0367968, 0.0140008, -0.0335647, 0.0484896
5: -0.0035712, 0.0218832, -0.0102173, 0.0248999, -0.0284712, 0.0321005
6: -0.0146094, 0.0367258, -0.0149622, 0.0461056, -0.0607149, 0.0516881
7: -0.0255142, 0.0082131, -0.0379991, 0.0160711, -0.0415853, 0.0462122
8: 0.9105793, 0.9901145, 0.8745046, 0.9907982, -0.0802189, 0.1156098
9: -0.0180809, 0.0317093, -0.0200594, 0.0753439, -0.0934248, 0.0517687

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1194914, upper bound: 0.1312529
time: 1.31 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1179944, upper bound: 0.1266980
time: 1.40 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0018446, 0.0253376, -0.0210255, 0.0320873, -0.0339319, 0.0463631
1: -0.0107035, 0.0231011, -0.0208819, 0.0263850, -0.0370885, 0.0439830
2: 0.0020702, 0.0405154, -0.0067022, 0.0688817, -0.0668115, 0.0472175
3: -0.0346573, 0.0212074, -0.0423613, 0.0313443, -0.0660017, 0.0635687
4: -0.0197595, 0.0118896, -0.0381756, 0.0141864, -0.0339460, 0.0500652
5: -0.0037002, 0.0221692, -0.0108888, 0.0251874, -0.0288876, 0.0330580
6: -0.0147167, 0.0370870, -0.0152089, 0.0469349, -0.0616516, 0.0522960
7: -0.0257043, 0.0083316, -0.0388606, 0.0166887, -0.0423930, 0.0471922
8: 0.9096730, 0.9901440, 0.8717944, 0.9908266, -0.0811536, 0.1183496
9: -0.0181753, 0.0322051, -0.0202842, 0.0783216, -0.0964969, 0.0524893

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1200191, upper bound: 0.1321152
time: 1.29 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1192725, upper bound: 0.1286990
time: 1.24 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0018184, 0.0250992, -0.0255325, 0.0345688, -0.0363872, 0.0506318
1: -0.0106059, 0.0228562, -0.0232305, 0.0282302, -0.0388362, 0.0460867
2: 0.0022251, 0.0402907, -0.0091065, 0.0770265, -0.0748013, 0.0493972
3: -0.0343985, 0.0208982, -0.0455996, 0.0338272, -0.0682257, 0.0664978
4: -0.0196078, 0.0117366, -0.0426146, 0.0148631, -0.0344710, 0.0543511
5: -0.0035999, 0.0219476, -0.0131736, 0.0262754, -0.0298753, 0.0351212
6: -0.0146205, 0.0368069, -0.0156689, 0.0496528, -0.0642732, 0.0524759
7: -0.0255555, 0.0082361, -0.0415158, 0.0193868, -0.0449423, 0.0497519
8: 0.9103777, 0.9901208, 0.8631234, 0.9909771, -0.0805994, 0.1269975
9: -0.0181019, 0.0318193, -0.0215279, 0.0880855, -0.1061875, 0.0533472

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1194907, upper bound: 0.1312027
time: 1.42 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1179590, upper bound: 0.1266907
time: 1.14 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0018514, 0.0254070, -0.0277824, 0.0356670, -0.0375184, 0.0531893
1: -0.0107318, 0.0231729, -0.0243561, 0.0289956, -0.0397274, 0.0475290
2: 0.0020248, 0.0405808, -0.0101839, 0.0811201, -0.0790953, 0.0507647
3: -0.0347329, 0.0212980, -0.0469746, 0.0349155, -0.0696484, 0.0682727
4: -0.0198039, 0.0119339, -0.0447152, 0.0151561, -0.0349600, 0.0566491
5: -0.0037292, 0.0222341, -0.0142424, 0.0267182, -0.0304474, 0.0364766
6: -0.0147280, 0.0371689, -0.0160680, 0.0509725, -0.0657006, 0.0532369
7: -0.0257461, 0.0083547, -0.0428509, 0.0204652, -0.0462113, 0.0512056
8: 0.9094694, 0.9901504, 0.8589616, 0.9910289, -0.0815595, 0.1311888
9: -0.0181965, 0.0323164, -0.0219649, 0.0926599, -0.1108563, 0.0542813

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1200349, upper bound: 0.1321116
time: 1.66 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1192730, upper bound: 0.1286993
time: 1.32 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0015939, 0.0235713, -0.0196186, 0.0313761, -0.0329701, 0.0431900
1: -0.0098501, 0.0212225, -0.0201462, 0.0259039, -0.0357540, 0.0413687
2: 0.0031875, 0.0386447, -0.0060397, 0.0662618, -0.0630743, 0.0446844
3: -0.0327384, 0.0186572, -0.0415117, 0.0306548, -0.0633931, 0.0601689
4: -0.0185900, 0.0105802, -0.0367968, 0.0140008, -0.0325908, 0.0473770
5: -0.0030540, 0.0203049, -0.0102173, 0.0248999, -0.0279539, 0.0305223
6: -0.0150916, 0.0347208, -0.0149622, 0.0461056, -0.0611972, 0.0496830
7: -0.0243457, 0.0084422, -0.0379991, 0.0160711, -0.0404168, 0.0464413
8: 0.9160390, 0.9904495, 0.8745046, 0.9907982, -0.0747592, 0.1159449
9: -0.0179855, 0.0290031, -0.0200594, 0.0753439, -0.0933294, 0.0490626

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1214429, upper bound: 0.1337152
time: 1.35 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1201162, upper bound: 0.1296799
time: 1.19 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0016254, 0.0238650, -0.0210255, 0.0320873, -0.0337127, 0.0448905
1: -0.0099695, 0.0215255, -0.0208819, 0.0263850, -0.0363546, 0.0424074
2: 0.0029962, 0.0389208, -0.0067022, 0.0688817, -0.0658855, 0.0456230
3: -0.0330595, 0.0190400, -0.0423613, 0.0313443, -0.0644038, 0.0614013
4: -0.0187776, 0.0107678, -0.0381756, 0.0141864, -0.0329640, 0.0489434
5: -0.0031760, 0.0205795, -0.0108888, 0.0251874, -0.0283634, 0.0314684
6: -0.0151982, 0.0350693, -0.0152089, 0.0469349, -0.0621331, 0.0502783
7: -0.0245275, 0.0085588, -0.0388606, 0.0166887, -0.0412162, 0.0474194
8: 0.9151753, 0.9904779, 0.8717944, 0.9908266, -0.0756513, 0.1186835
9: -0.0180765, 0.0294723, -0.0202842, 0.0783216, -0.0963980, 0.0497565

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1218259, upper bound: 0.1344058
time: 1.54 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1210687, upper bound: 0.1311868
time: 1.23 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0016006, 0.0236395, -0.0255325, 0.0345688, -0.0361694, 0.0491720
1: -0.0098777, 0.0212934, -0.0232305, 0.0282302, -0.0381079, 0.0445239
2: 0.0031430, 0.0387089, -0.0091065, 0.0770265, -0.0738834, 0.0478154
3: -0.0328135, 0.0187464, -0.0455996, 0.0338272, -0.0666407, 0.0643460
4: -0.0186338, 0.0106234, -0.0426146, 0.0148631, -0.0334969, 0.0532379
5: -0.0030821, 0.0203689, -0.0131736, 0.0262754, -0.0293575, 0.0335426
6: -0.0151033, 0.0348022, -0.0156689, 0.0496528, -0.0647560, 0.0504711
7: -0.0243865, 0.0084649, -0.0415158, 0.0193868, -0.0437733, 0.0499807
8: 0.9158401, 0.9904556, 0.8631234, 0.9909771, -0.0751370, 0.1273323
9: -0.0180065, 0.0291108, -0.0215279, 0.0880855, -0.1060920, 0.0506387

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2_B1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1214427, upper bound: 0.1336809
time: 1.49 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2_B1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1200971, upper bound: 0.1296792
time: 1.22 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0016321, 0.0239339, -0.0277824, 0.0356670, -0.0372991, 0.0517163
1: -0.0099975, 0.0215972, -0.0243561, 0.0289956, -0.0389931, 0.0459533
2: 0.0029513, 0.0389856, -0.0101839, 0.0811201, -0.0781688, 0.0491695
3: -0.0331353, 0.0191305, -0.0469746, 0.0349155, -0.0680508, 0.0661051
4: -0.0188218, 0.0108115, -0.0447152, 0.0151561, -0.0339779, 0.0555267
5: -0.0032044, 0.0206443, -0.0142424, 0.0267182, -0.0299227, 0.0348867
6: -0.0152098, 0.0351517, -0.0160680, 0.0509725, -0.0661824, 0.0512197
7: -0.0245690, 0.0085817, -0.0428509, 0.0204652, -0.0450342, 0.0514326
8: 0.9149743, 0.9904841, 0.8589616, 0.9910289, -0.0760546, 0.1315225
9: -0.0180976, 0.0295814, -0.0219649, 0.0926599, -0.1107575, 0.0515463

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 175
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 175

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2_B2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1218344, upper bound: 0.1343988
time: 1.24 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_A2_B2_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1210705, upper bound: 0.1311924
time: 1.43 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.49 + 597.05 = 601.54 seconds
