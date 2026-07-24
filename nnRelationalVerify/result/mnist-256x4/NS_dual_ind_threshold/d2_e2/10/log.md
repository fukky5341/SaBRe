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
execution time: IAR + RelationalAnalysis = 1.74 + 2.65 = 4.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1506555, upper bound: 0.1506555

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1330585, upper bound: 0.1443145
time: 2.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1495534, upper bound: 0.1495534
time: 1.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.19
Output dim: 8, lower bound: -0.1330585, upper bound: 0.1443145
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.19
Output dim: 8, lower bound: -0.1495534, upper bound: 0.1495534

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0019119, 0.0277417, -0.0478219, 0.0511951, -0.0531070, 0.0755637
1: -0.0111849, 0.0254320, -0.0373440, 0.0367610, -0.0479459, 0.0627760
2: 0.0010098, 0.0422721, -0.0243897, 0.1169914, -0.1159816, 0.0666618
3: -0.0364435, 0.0234478, -0.0615586, 0.0470138, -0.0834573, 0.0850064
4: -0.0209180, 0.0126847, -0.0690749, 0.0184173, -0.0393353, 0.0817596
5: -0.0044232, 0.0238883, -0.0262166, 0.0311483, -0.0355714, 0.0501049
6: -0.0160163, 0.0386838, -0.0218710, 0.0639239, -0.0799402, 0.0605549
7: -0.0265311, 0.0099074, -0.0561894, 0.0327967, -0.0593279, 0.0660967
8: 0.9059174, 0.9909688, 0.8201787, 0.9919229, -0.0860055, 0.1707901
9: -0.0189111, 0.0356451, -0.0293109, 0.1344257, -0.1533367, 0.0649560

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1293841, upper bound: 0.1330874
time: 1.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1433198
time: 1.33 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0545596, 0.0590740, -0.0649462, 0.0711331, -0.1256926, 0.1240202
1: -0.0416306, 0.0377633, -0.0514351, 0.0405276, -0.0821582, 0.0891984
2: -0.0291157, 0.1281760, -0.0369548, 0.1439655, -0.1730812, 0.1651307
3: -0.0648195, 0.0518410, -0.0710570, 0.0654267, -0.1302463, 0.1228979
4: -0.0767542, 0.0194023, -0.0885529, 0.0217181, -0.0984723, 0.1079551
5: -0.0305920, 0.0315645, -0.0391050, 0.0332697, -0.0638617, 0.0706695
6: -0.0251009, 0.0672517, -0.0304398, 0.0748164, -0.0999173, 0.0976915
7: -0.0620919, 0.0371661, -0.0706228, 0.0447955, -0.1068874, 0.1077889
8: 0.8074871, 0.9919812, 0.7881831, 0.9921900, -0.1847029, 0.2037981
9: -0.0311402, 0.1496067, -0.0352478, 0.1706189, -0.2017591, 0.1848545

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1443145, upper bound: 0.1330585
time: 2.78 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1443145, upper bound: 0.1495534
time: 2.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.85 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 6.85
Output dim: 8, lower bound: -0.1293841, upper bound: 0.1330874
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.85
Output dim: 8, lower bound: -0.1317660, upper bound: 0.1433198
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.85
Output dim: 8, lower bound: -0.1443145, upper bound: 0.1330585
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.85
Output dim: 8, lower bound: -0.1443145, upper bound: 0.1495534

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0019119, 0.0277417, -0.0324517, 0.0387565, -0.0406684, 0.0601935
1: -0.0111849, 0.0254320, -0.0268791, 0.0314468, -0.0426317, 0.0523111
2: 0.0010098, 0.0422721, -0.0131081, 0.0897795, -0.0887696, 0.0553801
3: -0.0364435, 0.0234478, -0.0505909, 0.0377685, -0.0742120, 0.0740387
4: -0.0209180, 0.0126847, -0.0494978, 0.0160478, -0.0369658, 0.0621825
5: -0.0044232, 0.0238883, -0.0168407, 0.0281426, -0.0325658, 0.0407290
6: -0.0160163, 0.0386838, -0.0175819, 0.0541344, -0.0701507, 0.0562658
7: -0.0265311, 0.0099074, -0.0457369, 0.0232424, -0.0497736, 0.0556443
8: 0.9059174, 0.9909688, 0.8497119, 0.9915081, -0.0855907, 0.1412569
9: -0.0189111, 0.0356451, -0.0235396, 0.1020685, -0.1209795, 0.0591847

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1391780
time: 1.48 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1433198
time: 2.23 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0545596, 0.0590740, -0.0019119, 0.0277417, -0.0823013, 0.0609859
1: -0.0416306, 0.0377633, -0.0111849, 0.0254320, -0.0670626, 0.0489482
2: -0.0291157, 0.1281760, 0.0010098, 0.0422721, -0.0713878, 0.1271661
3: -0.0648195, 0.0518410, -0.0364435, 0.0234478, -0.0882673, 0.0882845
4: -0.0767542, 0.0194023, -0.0209180, 0.0126847, -0.0894389, 0.0403203
5: -0.0305920, 0.0315645, -0.0044232, 0.0238883, -0.0544803, 0.0359877
6: -0.0251009, 0.0672517, -0.0160163, 0.0386838, -0.0637847, 0.0832681
7: -0.0620919, 0.0371661, -0.0265311, 0.0099074, -0.0719993, 0.0636973
8: 0.8074871, 0.9919812, 0.9059174, 0.9909688, -0.1834817, 0.0860638
9: -0.0311402, 0.1496067, -0.0189111, 0.0356451, -0.0667854, 0.1685178

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1330874, upper bound: 0.1293841
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1433198, upper bound: 0.1317660
time: 1.47 seconds

## BFS NS instance: NS_A2_B2

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1330874, upper bound: 0.1422156
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1433198, upper bound: 0.1470386
time: 2.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.72 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1391780
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1433198
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 5.72
Output dim: 8, lower bound: -0.1330874, upper bound: 0.1293841
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 8, lower bound: -0.1433198, upper bound: 0.1317660
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 8, lower bound: -0.1330874, upper bound: 0.1422156
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.72
Output dim: 8, lower bound: -0.1433198, upper bound: 0.1470386

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0006995, 0.0190184, -0.0324517, 0.0387565, -0.0394560, 0.0514701
1: -0.0075436, 0.0165633, -0.0268791, 0.0314468, -0.0389905, 0.0434424
2: 0.0059481, 0.0337382, -0.0131081, 0.0897795, -0.0838314, 0.0468463
3: -0.0280911, 0.0118530, -0.0505909, 0.0377685, -0.0658596, 0.0624438
4: -0.0156933, 0.0070357, -0.0494978, 0.0160478, -0.0317410, 0.0565335
5: -0.0014174, 0.0155574, -0.0168407, 0.0281426, -0.0295600, 0.0323981
6: -0.0156307, 0.0284073, -0.0175819, 0.0541344, -0.0697650, 0.0459892
7: -0.0198765, 0.0079921, -0.0457369, 0.0232424, -0.0431190, 0.0537290
8: 0.9335532, 0.9904189, 0.8497119, 0.9915081, -0.0579549, 0.1407070
9: -0.0171030, 0.0207279, -0.0235396, 0.1020685, -0.1191714, 0.0442675

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1293841
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1391780
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0016698, 0.0243654, -0.0324517, 0.0387565, -0.0404263, 0.0568171
1: -0.0101601, 0.0220349, -0.0268791, 0.0314468, -0.0416070, 0.0489140
2: 0.0026673, 0.0393722, -0.0131081, 0.0897795, -0.0871122, 0.0524803
3: -0.0336137, 0.0196589, -0.0505909, 0.0377685, -0.0713822, 0.0702498
4: -0.0190942, 0.0110712, -0.0494978, 0.0160478, -0.0351420, 0.0605690
5: -0.0034083, 0.0210243, -0.0168407, 0.0281426, -0.0315509, 0.0378650
6: -0.0156797, 0.0356317, -0.0175819, 0.0541344, -0.0698141, 0.0532136
7: -0.0247940, 0.0090796, -0.0457369, 0.0232424, -0.0480364, 0.0548165
8: 0.9138293, 0.9907394, 0.8497119, 0.9915081, -0.0776788, 0.1410275
9: -0.0182961, 0.0302396, -0.0235396, 0.1020685, -0.1203645, 0.0537792

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1312569
time: 1.72 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1421142
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0391726, 0.0412229, -0.0019119, 0.0277417, -0.0669144, 0.0431348
1: -0.0310785, 0.0320921, -0.0111849, 0.0254320, -0.0565105, 0.0432770
2: -0.0164209, 0.1010427, 0.0010098, 0.0422721, -0.0586930, 0.1000329
3: -0.0538648, 0.0401076, -0.0364435, 0.0234478, -0.0773126, 0.0765511
4: -0.0571271, 0.0163730, -0.0209180, 0.0126847, -0.0698118, 0.0372910
5: -0.0199711, 0.0285410, -0.0044232, 0.0238883, -0.0438594, 0.0329642
6: -0.0182337, 0.0574382, -0.0160163, 0.0386838, -0.0569175, 0.0734546
7: -0.0493294, 0.0271645, -0.0265311, 0.0099074, -0.0592368, 0.0536957
8: 0.8386118, 0.9915689, 0.9059174, 0.9909688, -0.1523570, 0.0856515
9: -0.0250701, 0.1156761, -0.0189111, 0.0356451, -0.0607153, 0.1345871

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391780, upper bound: 0.1234112
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391780, upper bound: 0.1317660
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A1

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

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1352364
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1422156
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_A2

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519
time: 1.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.27 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1293841
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1391780
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1312569
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1234112, upper bound: 0.1421142
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1391780, upper bound: 0.1234112
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1391780, upper bound: 0.1317660
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1352364
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1363968, upper bound: 0.1422156
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.27
Output dim: 8, lower bound: -0.1453975, upper bound: 0.1352519

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0006995, 0.0190184, -0.0391723, 0.0412152, -0.0419146, 0.0581907
1: -0.0075436, 0.0165633, -0.0310770, 0.0320843, -0.0396279, 0.0476403
2: 0.0059481, 0.0337382, -0.0164167, 0.1010380, -0.0950899, 0.0501550
3: -0.0280911, 0.0118530, -0.0538561, 0.0401018, -0.0681930, 0.0657090
4: -0.0156933, 0.0070357, -0.0571230, 0.0163703, -0.0320636, 0.0641587
5: -0.0014174, 0.0155574, -0.0199682, 0.0285364, -0.0299538, 0.0355256
6: -0.0156307, 0.0284073, -0.0182298, 0.0574338, -0.0730645, 0.0466371
7: -0.0198765, 0.0079921, -0.0493276, 0.0271584, -0.0470349, 0.0573197
8: 0.9335532, 0.9904189, 0.8386213, 0.9915666, -0.0580134, 0.1517976
9: -0.0171030, 0.0207279, -0.0250652, 0.1156667, -0.1327697, 0.0457931

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1227302, upper bound: 0.1381548
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1231885, upper bound: 0.1389203
time: 1.54 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0016698, 0.0243654, -0.0391723, 0.0412152, -0.0428850, 0.0635377
1: -0.0101601, 0.0220349, -0.0310770, 0.0320843, -0.0422444, 0.0531119
2: 0.0026673, 0.0393722, -0.0164167, 0.1010380, -0.0983707, 0.0557889
3: -0.0336137, 0.0196589, -0.0538561, 0.0401018, -0.0737156, 0.0735150
4: -0.0190942, 0.0110712, -0.0571230, 0.0163703, -0.0354645, 0.0681942
5: -0.0034083, 0.0210243, -0.0199682, 0.0285364, -0.0319447, 0.0409925
6: -0.0156797, 0.0356317, -0.0182298, 0.0574338, -0.0731135, 0.0538615
7: -0.0247940, 0.0090796, -0.0493276, 0.0271584, -0.0519523, 0.0584072
8: 0.9138293, 0.9907394, 0.8386213, 0.9915666, -0.0777373, 0.1521181
9: -0.0182961, 0.0302396, -0.0250652, 0.1156667, -0.1339628, 0.0553048

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1312433, upper bound: 0.1413901
time: 1.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1315190, upper bound: 0.1418500
time: 1.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0391726, 0.0412229, -0.0006995, 0.0190184, -0.0581910, 0.0419224
1: -0.0310785, 0.0320921, -0.0075436, 0.0165633, -0.0476418, 0.0396357
2: -0.0164209, 0.1010427, 0.0059481, 0.0337382, -0.0501591, 0.0950946
3: -0.0538648, 0.0401076, -0.0280911, 0.0118530, -0.0657178, 0.0681987
4: -0.0571271, 0.0163730, -0.0156933, 0.0070357, -0.0641628, 0.0320663
5: -0.0199711, 0.0285410, -0.0014174, 0.0155574, -0.0355285, 0.0299584
6: -0.0182337, 0.0574382, -0.0156307, 0.0284073, -0.0466410, 0.0730689
7: -0.0493294, 0.0271645, -0.0198765, 0.0079921, -0.0573215, 0.0470411
8: 0.8386118, 0.9915689, 0.9335532, 0.9904189, -0.1518070, 0.0580157
9: -0.0250701, 0.1156761, -0.0171030, 0.0207279, -0.0457980, 0.1327790

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384570, upper bound: 0.1226039
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389203, upper bound: 0.1231885
time: 1.33 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0391726, 0.0412229, -0.0016698, 0.0243654, -0.0635380, 0.0428927
1: -0.0310785, 0.0320921, -0.0101601, 0.0220349, -0.0531133, 0.0422522
2: -0.0164209, 0.1010427, 0.0026673, 0.0393722, -0.0557931, 0.0983754
3: -0.0538648, 0.0401076, -0.0336137, 0.0196589, -0.0735237, 0.0737213
4: -0.0571271, 0.0163730, -0.0190942, 0.0110712, -0.0681983, 0.0354672
5: -0.0199711, 0.0285410, -0.0034083, 0.0210243, -0.0409954, 0.0319493
6: -0.0182337, 0.0574382, -0.0156797, 0.0356317, -0.0538653, 0.0731180
7: -0.0493294, 0.0271645, -0.0247940, 0.0090796, -0.0584090, 0.0519585
8: 0.8386118, 0.9915689, 0.9138293, 0.9907394, -0.1521276, 0.0777396
9: -0.0250701, 0.1156761, -0.0182961, 0.0302396, -0.0553097, 0.1339721

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384570, upper bound: 0.1305137
time: 1.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389203, upper bound: 0.1310014
time: 3.22 seconds

## BFS NS instance: NS_A2_B2_A1_B1

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1341569
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1349809
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1406437
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1419607
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1

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

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1442675, upper bound: 0.1342772
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1349960
time: 1.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2

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

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1442675, upper bound: 0.1452948
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1462186
time: 1.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.21 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1227302, upper bound: 0.1381548
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1231885, upper bound: 0.1389203
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1312433, upper bound: 0.1413901
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1315190, upper bound: 0.1418500
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1384570, upper bound: 0.1226039
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1389203, upper bound: 0.1231885
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1384570, upper bound: 0.1305137
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1389203, upper bound: 0.1310014
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1341569
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1349809
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1354466, upper bound: 0.1406437
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1361462, upper bound: 0.1419607
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1442675, upper bound: 0.1342772
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1349960
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1442675, upper bound: 0.1452948
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.21
Output dim: 8, lower bound: -0.1451483, upper bound: 0.1462186

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0008707, 0.0198998, -0.0381500, 0.0405508, -0.0414214, 0.0580499
1: -0.0080284, 0.0173411, -0.0303754, 0.0317010, -0.0397293, 0.0477164
2: 0.0056564, 0.0347844, -0.0157151, 0.0991924, -0.0935360, 0.0504995
3: -0.0291138, 0.0132635, -0.0530729, 0.0395648, -0.0686786, 0.0663365
4: -0.0162181, 0.0077863, -0.0558047, 0.0162255, -0.0324436, 0.0635909
5: -0.0016230, 0.0160990, -0.0193699, 0.0283094, -0.0299325, 0.0354689
6: -0.0146490, 0.0297791, -0.0179225, 0.0567553, -0.0714044, 0.0477015
7: -0.0208000, 0.0070211, -0.0487178, 0.0264666, -0.0472666, 0.0557389
8: 0.9298151, 0.9897382, 0.8405712, 0.9914240, -0.0616090, 0.1491671
9: -0.0169754, 0.0221925, -0.0246670, 0.1135311, -0.1305065, 0.0468595

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381247
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381548
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0006787, 0.0187909, -0.0391723, 0.0412152, -0.0418939, 0.0579632
1: -0.0074538, 0.0163851, -0.0310770, 0.0320843, -0.0395381, 0.0474621
2: 0.0060388, 0.0335214, -0.0164167, 0.1010380, -0.0949992, 0.0499381
3: -0.0278158, 0.0115854, -0.0538561, 0.0401018, -0.0679176, 0.0654415
4: -0.0155632, 0.0068907, -0.0571230, 0.0163703, -0.0319335, 0.0640138
5: -0.0012751, 0.0154698, -0.0199682, 0.0285364, -0.0298115, 0.0354380
6: -0.0152177, 0.0281398, -0.0182298, 0.0574338, -0.0726515, 0.0463696
7: -0.0197580, 0.0076004, -0.0493276, 0.0271584, -0.0469164, 0.0569279
8: 0.9341831, 0.9902020, 0.8386213, 0.9915666, -0.0573835, 0.1515808
9: -0.0169410, 0.0204402, -0.0250652, 0.1156667, -0.1326077, 0.0455054

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1384570
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1389203
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0018666, 0.0255524, -0.0381500, 0.0405508, -0.0424174, 0.0637024
1: -0.0107909, 0.0233222, -0.0303754, 0.0317010, -0.0424918, 0.0536976
2: 0.0019298, 0.0407174, -0.0157151, 0.0991924, -0.0972626, 0.0564326
3: -0.0348904, 0.0214866, -0.0530729, 0.0395648, -0.0744552, 0.0745595
4: -0.0198965, 0.0120266, -0.0558047, 0.0162255, -0.0361220, 0.0678313
5: -0.0037908, 0.0223690, -0.0193699, 0.0283094, -0.0321002, 0.0417389
6: -0.0147785, 0.0373392, -0.0179225, 0.0567553, -0.0715338, 0.0552617
7: -0.0258353, 0.0084135, -0.0487178, 0.0264666, -0.0523019, 0.0571313
8: 0.9090434, 0.9901649, 0.8405712, 0.9914240, -0.0823806, 0.1495938
9: -0.0182422, 0.0325506, -0.0246670, 0.1135311, -0.1317733, 0.0572177

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1309670, upper bound: 0.1413475
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1309670, upper bound: 0.1413901
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0016469, 0.0240765, -0.0391723, 0.0412152, -0.0428621, 0.0632488
1: -0.0100552, 0.0217439, -0.0310770, 0.0320843, -0.0421395, 0.0528209
2: 0.0028583, 0.0391192, -0.0164167, 0.1010380, -0.0981797, 0.0555360
3: -0.0332908, 0.0193161, -0.0538561, 0.0401018, -0.0733926, 0.0731722
4: -0.0189128, 0.0109020, -0.0571230, 0.0163703, -0.0352831, 0.0680250
5: -0.0032642, 0.0207770, -0.0199682, 0.0285364, -0.0318006, 0.0407453
6: -0.0152594, 0.0353205, -0.0182298, 0.0574338, -0.0726932, 0.0535503
7: -0.0246563, 0.0086383, -0.0493276, 0.0271584, -0.0518147, 0.0579659
8: 0.9145586, 0.9904979, 0.8386213, 0.9915666, -0.0770080, 0.1518766
9: -0.0181429, 0.0298088, -0.0250652, 0.1156667, -0.1338096, 0.0548740

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1310405, upper bound: 0.1415083
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1310405, upper bound: 0.1418500
time: 1.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0006837, 0.0188431, -0.0728688, 0.0491283
1: -0.0408645, 0.0351376, -0.0074740, 0.0164229, -0.0572874, 0.0426116
2: -0.0250656, 0.1267675, 0.0060170, 0.0335689, -0.0586345, 0.1207505
3: -0.0628321, 0.0462464, -0.0278730, 0.0116460, -0.0744781, 0.0741194
4: -0.0751459, 0.0176258, -0.0155916, 0.0069257, -0.0820716, 0.0332175
5: -0.0276357, 0.0303247, -0.0013184, 0.0154896, -0.0431253, 0.0316431
6: -0.0203846, 0.0658701, -0.0154259, 0.0281984, -0.0485831, 0.0812959
7: -0.0576240, 0.0363257, -0.0197852, 0.0077672, -0.0653913, 0.0561110
8: 0.8125090, 0.9912007, 0.9340367, 0.9902968, -0.1777878, 0.0571640
9: -0.0291410, 0.1456623, -0.0169929, 0.0205112, -0.0496522, 0.1626552

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1224212
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1226039
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0006995, 0.0190184, -0.0566859, 0.0409604
1: -0.0300462, 0.0315486, -0.0075436, 0.0165633, -0.0466096, 0.0390922
2: -0.0153962, 0.0983359, 0.0059481, 0.0337382, -0.0491344, 0.0923878
3: -0.0527286, 0.0393295, -0.0280911, 0.0118530, -0.0645816, 0.0674206
4: -0.0551888, 0.0161649, -0.0156933, 0.0070357, -0.0622245, 0.0318581
5: -0.0190961, 0.0282199, -0.0014174, 0.0155574, -0.0346535, 0.0296373
6: -0.0177170, 0.0564468, -0.0156307, 0.0284073, -0.0461243, 0.0720775
7: -0.0484326, 0.0261334, -0.0198765, 0.0079921, -0.0564247, 0.0460099
8: 0.8414712, 0.9913020, 0.9335532, 0.9904189, -0.1489477, 0.0577488
9: -0.0244890, 0.1125363, -0.0171030, 0.0207279, -0.0452169, 0.1296393

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1227302
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1231885
time: 2.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0016533, 0.0241489, -0.0781747, 0.0500979
1: -0.0408645, 0.0351376, -0.0100820, 0.0218138, -0.0626783, 0.0452196
2: -0.0250656, 0.1267675, 0.0028107, 0.0391824, -0.0642480, 0.1239568
3: -0.0628321, 0.0462464, -0.0333673, 0.0194011, -0.0822332, 0.0796138
4: -0.0751459, 0.0176258, -0.0189572, 0.0109465, -0.0860924, 0.0365831
5: -0.0276357, 0.0303247, -0.0033052, 0.0208381, -0.0484738, 0.0336299
6: -0.0203846, 0.0658701, -0.0154685, 0.0353978, -0.0557824, 0.0813386
7: -0.0576240, 0.0363257, -0.0246929, 0.0088278, -0.0664518, 0.0610186
8: 0.8125090, 0.9912007, 0.9143675, 0.9906009, -0.1780919, 0.0768332
9: -0.0291410, 0.1456623, -0.0181882, 0.0299198, -0.0590609, 0.1638505

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1425401, upper bound: 0.1304362
time: 2.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1425401, upper bound: 0.1305137
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0016698, 0.0243654, -0.0620329, 0.0419307
1: -0.0300462, 0.0315486, -0.0101601, 0.0220349, -0.0520811, 0.0417087
2: -0.0153962, 0.0983359, 0.0026673, 0.0393722, -0.0547684, 0.0956686
3: -0.0527286, 0.0393295, -0.0336137, 0.0196589, -0.0723875, 0.0729432
4: -0.0551888, 0.0161649, -0.0190942, 0.0110712, -0.0662600, 0.0352591
5: -0.0190961, 0.0282199, -0.0034083, 0.0210243, -0.0401204, 0.0316282
6: -0.0177170, 0.0564468, -0.0156797, 0.0356317, -0.0533487, 0.0721265
7: -0.0484326, 0.0261334, -0.0247940, 0.0090796, -0.0575122, 0.0509274
8: 0.8414712, 0.9913020, 0.9138293, 0.9907394, -0.1492682, 0.0774727
9: -0.0244890, 0.1125363, -0.0182961, 0.0302396, -0.0547286, 0.1308323

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1426114, upper bound: 0.1307366
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1426114, upper bound: 0.1310014
time: 1.51 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

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

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1344610
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1349809
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0022585, 0.0241961, -0.0381503, 0.0405585, -0.0428170, 0.0623464
1: -0.0107930, 0.0218282, -0.0303768, 0.0317087, -0.0425018, 0.0522050
2: 0.0029539, 0.0401321, -0.0157193, 0.0991971, -0.0962432, 0.0558514
3: -0.0327061, 0.0208967, -0.0530817, 0.0395705, -0.0722766, 0.0739785
4: -0.0190003, 0.0120081, -0.0558087, 0.0162281, -0.0352284, 0.0678168
5: -0.0030589, 0.0219027, -0.0193728, 0.0283141, -0.0313730, 0.0412755
6: -0.0153214, 0.0369249, -0.0179263, 0.0567597, -0.0720812, 0.0548512
7: -0.0270843, 0.0082864, -0.0487196, 0.0264727, -0.0535570, 0.0570060
8: 0.9076468, 0.9901348, 0.8405618, 0.9914264, -0.0837796, 0.1495730
9: -0.0170160, 0.0319851, -0.0246719, 0.1135405, -0.1305564, 0.0566570

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1406302
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1406437
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0020111, 0.0228301, -0.0391726, 0.0412229, -0.0432340, 0.0620028
1: -0.0100880, 0.0203515, -0.0310785, 0.0320921, -0.0421801, 0.0514300
2: 0.0038061, 0.0386262, -0.0164209, 0.1010427, -0.0972366, 0.0550471
3: -0.0312670, 0.0187905, -0.0538648, 0.0401076, -0.0713747, 0.0726553
4: -0.0180915, 0.0109365, -0.0571271, 0.0163730, -0.0344645, 0.0680636
5: -0.0026025, 0.0203822, -0.0199711, 0.0285410, -0.0311435, 0.0403533
6: -0.0157730, 0.0349574, -0.0182337, 0.0574382, -0.0732112, 0.0531911
7: -0.0258241, 0.0084808, -0.0493294, 0.0271645, -0.0529886, 0.0578102
8: 0.9129899, 0.9904315, 0.8386118, 0.9915689, -0.0785791, 0.1518196
9: -0.0169642, 0.0293804, -0.0250701, 0.1156761, -0.1326402, 0.0544505

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1413976
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1419607
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

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

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1340690
time: 1.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1342772
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

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

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1344760
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1349960
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0381503, 0.0405585, -0.0945842, 0.0865949
1: -0.0408645, 0.0351376, -0.0303768, 0.0317087, -0.0725732, 0.0655144
2: -0.0250656, 0.1267675, -0.0157193, 0.0991971, -0.1242627, 0.1424868
3: -0.0628321, 0.0462464, -0.0530817, 0.0395705, -0.1024026, 0.0993282
4: -0.0751459, 0.0176258, -0.0558087, 0.0162281, -0.0913740, 0.0734345
5: -0.0276357, 0.0303247, -0.0193728, 0.0283141, -0.0559498, 0.0496975
6: -0.0203846, 0.0658701, -0.0179263, 0.0567597, -0.0771444, 0.0837964
7: -0.0576240, 0.0363257, -0.0487196, 0.0264727, -0.0840968, 0.0850453
8: 0.8125090, 0.9912007, 0.8405618, 0.9914264, -0.1789174, 0.1506389
9: -0.0291410, 0.1456623, -0.0246719, 0.1135405, -0.1426815, 0.1703342

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1451275
time: 2.00 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1452948
time: 1.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0391726, 0.0412229, -0.0788904, 0.0794335
1: -0.0300462, 0.0315486, -0.0310785, 0.0320921, -0.0621383, 0.0626271
2: -0.0153962, 0.0983359, -0.0164209, 0.1010427, -0.1164389, 0.1147568
3: -0.0527286, 0.0393295, -0.0538648, 0.0401076, -0.0928362, 0.0931943
4: -0.0551888, 0.0161649, -0.0571271, 0.0163730, -0.0715618, 0.0732920
5: -0.0190961, 0.0282199, -0.0199711, 0.0285410, -0.0476371, 0.0481910
6: -0.0177170, 0.0564468, -0.0182337, 0.0574382, -0.0751552, 0.0746804
7: -0.0484326, 0.0261334, -0.0493294, 0.0271645, -0.0755971, 0.0754628
8: 0.8414712, 0.9913020, 0.8386118, 0.9915689, -0.1500977, 0.1526902
9: -0.0244890, 0.1125363, -0.0250701, 0.1156761, -0.1401651, 0.1376064

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1477181, upper bound: 0.1456012
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1477181, upper bound: 0.1462186
time: 1.89 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.80 seconds
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381247
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1224212, upper bound: 0.1381548
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1384570
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1226039, upper bound: 0.1389203
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1309670, upper bound: 0.1413475
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1309670, upper bound: 0.1413901
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1310405, upper bound: 0.1415083
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1310405, upper bound: 0.1418500
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1224212
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1226039
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1227302
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1231885
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1425401, upper bound: 0.1304362
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1425401, upper bound: 0.1305137
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1426114, upper bound: 0.1307366
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1426114, upper bound: 0.1310014
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1344610
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1349809
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1406302
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1351125, upper bound: 0.1406437
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1413976
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1354923, upper bound: 0.1419607
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1340690
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1342772
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1344760
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1349960
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1451275
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1473200, upper bound: 0.1452948
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1477181, upper bound: 0.1456012
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.80
Output dim: 8, lower bound: -0.1477181, upper bound: 0.1462186

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0008707, 0.0198998, -0.0540253, 0.0484324, -0.0493031, 0.0739251
1: -0.0080284, 0.0173411, -0.0408621, 0.0351249, -0.0431533, 0.0582032
2: 0.0056564, 0.0347844, -0.0250597, 0.1267601, -0.1211037, 0.0598441
3: -0.0291138, 0.0132635, -0.0628187, 0.0462371, -0.0753509, 0.0760823
4: -0.0162181, 0.0077863, -0.0751395, 0.0176215, -0.0338396, 0.0829258
5: -0.0016230, 0.0160990, -0.0276314, 0.0303174, -0.0319404, 0.0437304
6: -0.0146490, 0.0297791, -0.0203774, 0.0658632, -0.0805122, 0.0501565
7: -0.0208000, 0.0070211, -0.0576211, 0.0363162, -0.0571162, 0.0646422
8: 0.9298151, 0.9897382, 0.8125240, 0.9911965, -0.0613815, 0.1772143
9: -0.0169754, 0.0221925, -0.0291325, 0.1456473, -0.1626228, 0.0513250

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1379433
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379473
time: 1.62 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0008707, 0.0198998, -0.0376672, 0.0402532, -0.0411239, 0.0575671
1: -0.0080284, 0.0173411, -0.0300448, 0.0315408, -0.0395692, 0.0473858
2: 0.0056564, 0.0347844, -0.0153920, 0.0983312, -0.0926748, 0.0501764
3: -0.0291138, 0.0132635, -0.0527199, 0.0393237, -0.0684375, 0.0659834
4: -0.0162181, 0.0077863, -0.0551848, 0.0161622, -0.0323803, 0.0629710
5: -0.0016230, 0.0160990, -0.0190932, 0.0282152, -0.0298383, 0.0351922
6: -0.0146490, 0.0297791, -0.0177132, 0.0564424, -0.0710914, 0.0474923
7: -0.0208000, 0.0070211, -0.0484307, 0.0261273, -0.0469273, 0.0554518
8: 0.9298151, 0.9897382, 0.8414806, 0.9912996, -0.0614845, 0.1482576
9: -0.0169754, 0.0221925, -0.0244842, 0.1125271, -0.1295025, 0.0466766

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1379746
time: 2.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379788
time: 1.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0006787, 0.0187909, -0.0540253, 0.0484324, -0.0491112, 0.0728161
1: -0.0074538, 0.0163851, -0.0408621, 0.0351249, -0.0425787, 0.0572473
2: 0.0060388, 0.0335214, -0.0250597, 0.1267601, -0.1207213, 0.0585811
3: -0.0278158, 0.0115854, -0.0628187, 0.0462371, -0.0740529, 0.0744042
4: -0.0155632, 0.0068907, -0.0751395, 0.0176215, -0.0331847, 0.0820302
5: -0.0012751, 0.0154698, -0.0276314, 0.0303174, -0.0315925, 0.0431012
6: -0.0152177, 0.0281398, -0.0203774, 0.0658632, -0.0810808, 0.0485172
7: -0.0197580, 0.0076004, -0.0576211, 0.0363162, -0.0560742, 0.0652215
8: 0.9341831, 0.9902020, 0.8125240, 0.9911965, -0.0570135, 0.1776780
9: -0.0169410, 0.0204402, -0.0291325, 0.1456473, -0.1625883, 0.0495726

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225916, upper bound: 0.1382508
time: 2.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224089, upper bound: 0.1382508
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0006787, 0.0187909, -0.0376672, 0.0402532, -0.0409319, 0.0564581
1: -0.0074538, 0.0163851, -0.0300448, 0.0315408, -0.0389946, 0.0464299
2: 0.0060388, 0.0335214, -0.0153920, 0.0983312, -0.0922924, 0.0489135
3: -0.0278158, 0.0115854, -0.0527199, 0.0393237, -0.0671395, 0.0643053
4: -0.0155632, 0.0068907, -0.0551848, 0.0161622, -0.0317254, 0.0620755
5: -0.0012751, 0.0154698, -0.0190932, 0.0282152, -0.0294904, 0.0345630
6: -0.0152177, 0.0281398, -0.0177132, 0.0564424, -0.0716600, 0.0458530
7: -0.0197580, 0.0076004, -0.0484307, 0.0261273, -0.0458852, 0.0560311
8: 0.9341831, 0.9902020, 0.8414806, 0.9912996, -0.0571165, 0.1487214
9: -0.0169410, 0.0204402, -0.0244842, 0.1125271, -0.1294680, 0.0449243

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225916, upper bound: 0.1387187
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224089, upper bound: 0.1387198
time: 2.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0018666, 0.0255524, -0.0540253, 0.0484324, -0.0502990, 0.0795777
1: -0.0107909, 0.0233222, -0.0408621, 0.0351249, -0.0459158, 0.0641844
2: 0.0019298, 0.0407174, -0.0250597, 0.1267601, -0.1248303, 0.0657771
3: -0.0348904, 0.0214866, -0.0628187, 0.0462371, -0.0811275, 0.0843053
4: -0.0198965, 0.0120266, -0.0751395, 0.0176215, -0.0375180, 0.0871661
5: -0.0037908, 0.0223690, -0.0276314, 0.0303174, -0.0341081, 0.0500004
6: -0.0147785, 0.0373392, -0.0203774, 0.0658632, -0.0806417, 0.0577166
7: -0.0258353, 0.0084135, -0.0576211, 0.0363162, -0.0621515, 0.0660346
8: 0.9090434, 0.9901649, 0.8125240, 0.9911965, -0.0821531, 0.1776410
9: -0.0182422, 0.0325506, -0.0291325, 0.1456473, -0.1638896, 0.0616831

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1309417, upper bound: 0.1411440
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411387
time: 1.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0018666, 0.0255524, -0.0376672, 0.0402532, -0.0421198, 0.0632196
1: -0.0107909, 0.0233222, -0.0300448, 0.0315408, -0.0423317, 0.0533670
2: 0.0019298, 0.0407174, -0.0153920, 0.0983312, -0.0964014, 0.0561095
3: -0.0348904, 0.0214866, -0.0527199, 0.0393237, -0.0742141, 0.0742064
4: -0.0198965, 0.0120266, -0.0551848, 0.0161622, -0.0360588, 0.0672114
5: -0.0037908, 0.0223690, -0.0190932, 0.0282152, -0.0320060, 0.0414622
6: -0.0147785, 0.0373392, -0.0177132, 0.0564424, -0.0712209, 0.0550524
7: -0.0258353, 0.0084135, -0.0484307, 0.0261273, -0.0519625, 0.0568443
8: 0.9090434, 0.9901649, 0.8414806, 0.9912996, -0.0822561, 0.1486843
9: -0.0182422, 0.0325506, -0.0244842, 0.1125271, -0.1307693, 0.0570348

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1309417, upper bound: 0.1411827
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411806
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0016469, 0.0240765, -0.0540253, 0.0484324, -0.0500793, 0.0781017
1: -0.0100552, 0.0217439, -0.0408621, 0.0351249, -0.0451801, 0.0626060
2: 0.0028583, 0.0391192, -0.0250597, 0.1267601, -0.1239018, 0.0641789
3: -0.0332908, 0.0193161, -0.0628187, 0.0462371, -0.0795279, 0.0821348
4: -0.0189128, 0.0109020, -0.0751395, 0.0176215, -0.0365343, 0.0860414
5: -0.0032642, 0.0207770, -0.0276314, 0.0303174, -0.0335816, 0.0484084
6: -0.0152594, 0.0353205, -0.0203774, 0.0658632, -0.0811226, 0.0556979
7: -0.0246563, 0.0086383, -0.0576211, 0.0363162, -0.0609725, 0.0662594
8: 0.9145586, 0.9904979, 0.8125240, 0.9911965, -0.0766379, 0.1779739
9: -0.0181429, 0.0298088, -0.0291325, 0.1456473, -0.1637902, 0.0589412

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1310072, upper bound: 0.1412868
time: 1.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305550, upper bound: 0.1412839
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0016469, 0.0240765, -0.0376672, 0.0402532, -0.0419001, 0.0617437
1: -0.0100552, 0.0217439, -0.0300448, 0.0315408, -0.0415960, 0.0517887
2: 0.0028583, 0.0391192, -0.0153920, 0.0983312, -0.0954729, 0.0545113
3: -0.0332908, 0.0193161, -0.0527199, 0.0393237, -0.0726145, 0.0720360
4: -0.0189128, 0.0109020, -0.0551848, 0.0161622, -0.0350750, 0.0660867
5: -0.0032642, 0.0207770, -0.0190932, 0.0282152, -0.0314794, 0.0398702
6: -0.0152594, 0.0353205, -0.0177132, 0.0564424, -0.0717018, 0.0530337
7: -0.0246563, 0.0086383, -0.0484307, 0.0261273, -0.0507836, 0.0570691
8: 0.9145586, 0.9904979, 0.8414806, 0.9912996, -0.0767410, 0.1490173
9: -0.0181429, 0.0298088, -0.0244842, 0.1125271, -0.1306699, 0.0542929

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1310072, upper bound: 0.1416215
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305550, upper bound: 0.1416182
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0008707, 0.0198998, -0.0739256, 0.0493153
1: -0.0408645, 0.0351376, -0.0080284, 0.0173411, -0.0582056, 0.0431659
2: -0.0250656, 0.1267675, 0.0056564, 0.0347844, -0.0598500, 0.1211111
3: -0.0628321, 0.0462464, -0.0291138, 0.0132635, -0.0760956, 0.0753603
4: -0.0751459, 0.0176258, -0.0162181, 0.0077863, -0.0829322, 0.0338440
5: -0.0276357, 0.0303247, -0.0016230, 0.0160990, -0.0437347, 0.0319478
6: -0.0203846, 0.0658701, -0.0146490, 0.0297791, -0.0501637, 0.0805191
7: -0.0576240, 0.0363257, -0.0208000, 0.0070211, -0.0646451, 0.0571257
8: 0.8125090, 0.9912007, 0.9298151, 0.9897382, -0.1772292, 0.0613856
9: -0.0291410, 0.1456623, -0.0169754, 0.0221925, -0.0513335, 0.1626377

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379811, upper bound: 0.1221972
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1224051
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0006787, 0.0187909, -0.0728166, 0.0491234
1: -0.0408645, 0.0351376, -0.0074538, 0.0163851, -0.0572496, 0.0425914
2: -0.0250656, 0.1267675, 0.0060388, 0.0335214, -0.0585870, 0.1207287
3: -0.0628321, 0.0462464, -0.0278158, 0.0115854, -0.0744175, 0.0740623
4: -0.0751459, 0.0176258, -0.0155632, 0.0068907, -0.0820366, 0.0331890
5: -0.0276357, 0.0303247, -0.0012751, 0.0154698, -0.0431055, 0.0315998
6: -0.0203846, 0.0658701, -0.0152177, 0.0281398, -0.0485244, 0.0810878
7: -0.0576240, 0.0363257, -0.0197580, 0.0076004, -0.0652244, 0.0560837
8: 0.8125090, 0.9912007, 0.9341831, 0.9902020, -0.1776930, 0.0570176
9: -0.0291410, 0.1456623, -0.0169410, 0.0204402, -0.0495812, 0.1626032

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379811, upper bound: 0.1224054
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1225943
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0008707, 0.0198998, -0.0575673, 0.0411316
1: -0.0300462, 0.0315486, -0.0080284, 0.0173411, -0.0473873, 0.0395770
2: -0.0153962, 0.0983359, 0.0056564, 0.0347844, -0.0501806, 0.0926795
3: -0.0527286, 0.0393295, -0.0291138, 0.0132635, -0.0659922, 0.0684433
4: -0.0551888, 0.0161649, -0.0162181, 0.0077863, -0.0629751, 0.0323830
5: -0.0190961, 0.0282199, -0.0016230, 0.0160990, -0.0351951, 0.0298429
6: -0.0177170, 0.0564468, -0.0146490, 0.0297791, -0.0474961, 0.0710958
7: -0.0484326, 0.0261334, -0.0208000, 0.0070211, -0.0554537, 0.0469334
8: 0.8414712, 0.9913020, 0.9298151, 0.9897382, -0.1482670, 0.0614870
9: -0.0244890, 0.1125363, -0.0169754, 0.0221925, -0.0466815, 0.1295117

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380220, upper bound: 0.1224901
time: 1.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1227179
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0006787, 0.0187909, -0.0564584, 0.0409396
1: -0.0300462, 0.0315486, -0.0074538, 0.0163851, -0.0464314, 0.0390024
2: -0.0153962, 0.0983359, 0.0060388, 0.0335214, -0.0489176, 0.0922971
3: -0.0527286, 0.0393295, -0.0278158, 0.0115854, -0.0643141, 0.0671453
4: -0.0551888, 0.0161649, -0.0155632, 0.0068907, -0.0620796, 0.0317280
5: -0.0190961, 0.0282199, -0.0012751, 0.0154698, -0.0345659, 0.0294950
6: -0.0177170, 0.0564468, -0.0152177, 0.0281398, -0.0458568, 0.0716645
7: -0.0484326, 0.0261334, -0.0197580, 0.0076004, -0.0560330, 0.0458914
8: 0.8414712, 0.9913020, 0.9341831, 0.9902020, -0.1487308, 0.0571190
9: -0.0244890, 0.1125363, -0.0169410, 0.0204402, -0.0449292, 0.1294773

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380220, upper bound: 0.1229431
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1231548
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0018666, 0.0255524, -0.0795781, 0.0503112
1: -0.0408645, 0.0351376, -0.0107909, 0.0233222, -0.0641867, 0.0459285
2: -0.0250656, 0.1267675, 0.0019298, 0.0407174, -0.0657830, 0.1248377
3: -0.0628321, 0.0462464, -0.0348904, 0.0214866, -0.0843187, 0.0811369
4: -0.0751459, 0.0176258, -0.0198965, 0.0120266, -0.0871725, 0.0375224
5: -0.0276357, 0.0303247, -0.0037908, 0.0223690, -0.0500047, 0.0341155
6: -0.0203846, 0.0658701, -0.0147785, 0.0373392, -0.0577238, 0.0806486
7: -0.0576240, 0.0363257, -0.0258353, 0.0084135, -0.0660376, 0.0621610
8: 0.8125090, 0.9912007, 0.9090434, 0.9901649, -0.1776559, 0.0821573
9: -0.0291410, 0.1456623, -0.0182422, 0.0325506, -0.0616917, 0.1639045

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1423145, upper bound: 0.1301601
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1425383, upper bound: 0.1304240
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0540257, 0.0484446, -0.0016469, 0.0240765, -0.0781022, 0.0500915
1: -0.0408645, 0.0351376, -0.0100552, 0.0217439, -0.0626084, 0.0451928
2: -0.0250656, 0.1267675, 0.0028583, 0.0391192, -0.0641848, 0.1239092
3: -0.0628321, 0.0462464, -0.0332908, 0.0193161, -0.0821482, 0.0795372
4: -0.0751459, 0.0176258, -0.0189128, 0.0109020, -0.0860478, 0.0365386
5: -0.0276357, 0.0303247, -0.0032642, 0.0207770, -0.0484127, 0.0335889
6: -0.0203846, 0.0658701, -0.0152594, 0.0353205, -0.0557051, 0.0811295
7: -0.0576240, 0.0363257, -0.0246563, 0.0086383, -0.0662624, 0.0609820
8: 0.8125090, 0.9912007, 0.9145586, 0.9904979, -0.1779889, 0.0766421
9: -0.0291410, 0.1456623, -0.0181429, 0.0298088, -0.0589498, 0.1638051

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1423145, upper bound: 0.1302162
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1425383, upper bound: 0.1304942
time: 1.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0018666, 0.0255524, -0.0632199, 0.0421275
1: -0.0300462, 0.0315486, -0.0107909, 0.0233222, -0.0533685, 0.0423395
2: -0.0153962, 0.0983359, 0.0019298, 0.0407174, -0.0561136, 0.0964061
3: -0.0527286, 0.0393295, -0.0348904, 0.0214866, -0.0742152, 0.0742199
4: -0.0551888, 0.0161649, -0.0198965, 0.0120266, -0.0672154, 0.0360614
5: -0.0190961, 0.0282199, -0.0037908, 0.0223690, -0.0414651, 0.0320106
6: -0.0177170, 0.0564468, -0.0147785, 0.0373392, -0.0550562, 0.0712253
7: -0.0484326, 0.0261334, -0.0258353, 0.0084135, -0.0568461, 0.0519687
8: 0.8414712, 0.9913020, 0.9090434, 0.9901649, -0.1486937, 0.0822586
9: -0.0244890, 0.1125363, -0.0182422, 0.0325506, -0.0570397, 0.1307785

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1424190, upper bound: 0.1304634
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1426106, upper bound: 0.1307118
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0376675, 0.0402609, -0.0016469, 0.0240765, -0.0617439, 0.0419078
1: -0.0300462, 0.0315486, -0.0100552, 0.0217439, -0.0517901, 0.0416038
2: -0.0153962, 0.0983359, 0.0028583, 0.0391192, -0.0545154, 0.0954776
3: -0.0527286, 0.0393295, -0.0332908, 0.0193161, -0.0720447, 0.0726203
4: -0.0551888, 0.0161649, -0.0189128, 0.0109020, -0.0660908, 0.0350777
5: -0.0190961, 0.0282199, -0.0032642, 0.0207770, -0.0398731, 0.0314841
6: -0.0177170, 0.0564468, -0.0152594, 0.0353205, -0.0530375, 0.0717062
7: -0.0484326, 0.0261334, -0.0246563, 0.0086383, -0.0570709, 0.0507897
8: 0.8414712, 0.9913020, 0.9145586, 0.9904979, -0.1490267, 0.0767434
9: -0.0244890, 0.1125363, -0.0181429, 0.0298088, -0.0542978, 0.1306792

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1424190, upper bound: 0.1306928
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1426106, upper bound: 0.1309724
time: 2.04 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

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

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1350729, upper bound: 0.1405008
time: 1.42 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405008
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1350729, upper bound: 0.1405153
time: 1.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405149
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

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

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354427, upper bound: 0.1412783
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1353880, upper bound: 0.1412840
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354427, upper bound: 0.1417744
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1353880, upper bound: 0.1417861
time: 2.30 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1436566, upper bound: 0.1338676
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1340467
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1436566, upper bound: 0.1340974
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1342546
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1437858, upper bound: 0.1342922
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1344536
time: 2.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1437858, upper bound: 0.1348329
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1349715
time: 2.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

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

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471705, upper bound: 0.1448961
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473196, upper bound: 0.1450693
time: 1.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 183

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471705, upper bound: 0.1450537
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473196, upper bound: 0.1452355
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1475582, upper bound: 0.1453992
time: 2.06 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1477143, upper bound: 0.1455570
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1475582, upper bound: 0.1460319
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1477143, upper bound: 0.1461895
time: 2.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.20 seconds
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1379433
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379473
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1224142, upper bound: 0.1379746
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1222235, upper bound: 0.1379788
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1225916, upper bound: 0.1382508
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1224089, upper bound: 0.1382508
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1225916, upper bound: 0.1387187
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1224089, upper bound: 0.1387198
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1309417, upper bound: 0.1411440
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411387
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1309417, upper bound: 0.1411827
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1305145, upper bound: 0.1411806
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1310072, upper bound: 0.1412868
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1305550, upper bound: 0.1412839
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1310072, upper bound: 0.1416215
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1305550, upper bound: 0.1416182
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1379811, upper bound: 0.1221972
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1224051
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1379811, upper bound: 0.1224054
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1381247, upper bound: 0.1225943
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1380220, upper bound: 0.1224901
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1227179
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1380220, upper bound: 0.1229431
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1381548, upper bound: 0.1231548
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1423145, upper bound: 0.1301601
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1425383, upper bound: 0.1304240
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1423145, upper bound: 0.1302162
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1425383, upper bound: 0.1304942
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1424190, upper bound: 0.1304634
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1426106, upper bound: 0.1307118
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1424190, upper bound: 0.1306928
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1426106, upper bound: 0.1309724
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1350729, upper bound: 0.1405008
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405008
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1350729, upper bound: 0.1405153
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1405149
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1354427, upper bound: 0.1412783
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1353880, upper bound: 0.1412840
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1354427, upper bound: 0.1417744
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1353880, upper bound: 0.1417861
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1436566, upper bound: 0.1338676
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1340467
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1436566, upper bound: 0.1340974
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1438782, upper bound: 0.1342546
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1437858, upper bound: 0.1342922
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1344536
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1437858, upper bound: 0.1348329
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1439685, upper bound: 0.1349715
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1471705, upper bound: 0.1448961
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1473196, upper bound: 0.1450693
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1471705, upper bound: 0.1450537
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1473196, upper bound: 0.1452355
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1475582, upper bound: 0.1453992
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1477143, upper bound: 0.1455570
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1475582, upper bound: 0.1460319
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.20
Output dim: 8, lower bound: -0.1477143, upper bound: 0.1461895

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0007255, 0.0185341, -0.0524759, 0.0475339, -0.0482595, 0.0710100
1: -0.0074363, 0.0162442, -0.0398181, 0.0346652, -0.0421015, 0.0560622
2: 0.0061794, 0.0333988, -0.0240615, 0.1240271, -0.1178477, 0.0574603
3: -0.0274709, 0.0115269, -0.0617415, 0.0455026, -0.0729735, 0.0732684
4: -0.0154255, 0.0068615, -0.0731925, 0.0174440, -0.0328695, 0.0800540
5: -0.0010335, 0.0154944, -0.0267699, 0.0300486, -0.0310821, 0.0422644
6: -0.0141299, 0.0280419, -0.0199964, 0.0649016, -0.0790315, 0.0480384
7: -0.0199523, 0.0066679, -0.0567209, 0.0352578, -0.0552100, 0.0633888
8: 0.9340628, 0.9897153, 0.8153608, 0.9911544, -0.0570916, 0.1743545
9: -0.0164730, 0.0203785, -0.0286014, 0.1424675, -0.1589405, 0.0489799

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1221816, upper bound: 0.1377536
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224017, upper bound: 0.1379433
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0008433, 0.0199433, -0.0529753, 0.0478245, -0.0486678, 0.0729186
1: -0.0080161, 0.0174116, -0.0401546, 0.0348141, -0.0428302, 0.0575662
2: 0.0056414, 0.0348078, -0.0243841, 0.1249075, -0.1192661, 0.0591919
3: -0.0292257, 0.0132736, -0.0620886, 0.0457394, -0.0749651, 0.0753622
4: -0.0162456, 0.0077591, -0.0738202, 0.0175014, -0.0337470, 0.0815793
5: -0.0016268, 0.0160642, -0.0270478, 0.0301356, -0.0317624, 0.0431119
6: -0.0143212, 0.0297787, -0.0201164, 0.0652117, -0.0795329, 0.0498951
7: -0.0207167, 0.0068729, -0.0570109, 0.0355974, -0.0563140, 0.0638838
8: 0.9300352, 0.9897074, 0.8144474, 0.9911669, -0.0611317, 0.1752599
9: -0.0169669, 0.0221489, -0.0287710, 0.1434909, -0.1604578, 0.0509199

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1220034, upper bound: 0.1377542
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222156, upper bound: 0.1379473
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0007255, 0.0185341, -0.0361533, 0.0393874, -0.0401129, 0.0546874
1: -0.0074363, 0.0162442, -0.0290306, 0.0310887, -0.0385249, 0.0452748
2: 0.0061794, 0.0333988, -0.0144257, 0.0956662, -0.0894868, 0.0478245
3: -0.0274709, 0.0115269, -0.0516839, 0.0386069, -0.0660778, 0.0632107
4: -0.0154255, 0.0068615, -0.0532942, 0.0159881, -0.0314137, 0.0601556
5: -0.0010335, 0.0154944, -0.0182573, 0.0279495, -0.0289830, 0.0337518
6: -0.0141299, 0.0280419, -0.0174445, 0.0555062, -0.0696361, 0.0454864
7: -0.0199523, 0.0066679, -0.0475535, 0.0251960, -0.0451482, 0.0542214
8: 0.9340628, 0.9897153, 0.8442495, 0.9912589, -0.0571961, 0.1454658
9: -0.0164730, 0.0203785, -0.0239953, 0.1094170, -0.1258901, 0.0443738

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224619, upper bound: 0.1377987
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1227176, upper bound: 0.1379746
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0008433, 0.0199433, -0.0366430, 0.0396683, -0.0405116, 0.0565862
1: -0.0080161, 0.0174116, -0.0293588, 0.0312360, -0.0392520, 0.0467704
2: 0.0056414, 0.0348078, -0.0147387, 0.0965284, -0.0908870, 0.0495465
3: -0.0292257, 0.0132736, -0.0520198, 0.0388387, -0.0680644, 0.0652934
4: -0.0162456, 0.0077591, -0.0539060, 0.0160446, -0.0322902, 0.0616651
5: -0.0016268, 0.0160642, -0.0185280, 0.0280360, -0.0296628, 0.0345921
6: -0.0143212, 0.0297787, -0.0175256, 0.0558096, -0.0701308, 0.0473043
7: -0.0207167, 0.0068729, -0.0478374, 0.0254935, -0.0462102, 0.0547103
8: 0.9300352, 0.9897074, 0.8433545, 0.9912711, -0.0612359, 0.1463529
9: -0.0169669, 0.0221489, -0.0241522, 0.1104216, -0.1273885, 0.0463012

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222866, upper bound: 0.1377996
time: 1.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225370, upper bound: 0.1379788
time: 2.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0005356, 0.0174531, -0.0524759, 0.0475339, -0.0480695, 0.0699290
1: -0.0068692, 0.0153094, -0.0398181, 0.0346652, -0.0415344, 0.0551275
2: 0.0065546, 0.0321588, -0.0240615, 0.1240271, -0.1174726, 0.0562203
3: -0.0262104, 0.0098755, -0.0617415, 0.0455026, -0.0717129, 0.0716170
4: -0.0147856, 0.0059808, -0.0731925, 0.0174440, -0.0322296, 0.0791732
5: -0.0007190, 0.0148735, -0.0267699, 0.0300486, -0.0307676, 0.0416434
6: -0.0146943, 0.0264273, -0.0199964, 0.0649016, -0.0795960, 0.0464238
7: -0.0189206, 0.0072527, -0.0567209, 0.0352578, -0.0541784, 0.0639736
8: 0.9383700, 0.9901789, 0.8153608, 0.9911544, -0.0527844, 0.1748181
9: -0.0164445, 0.0186969, -0.0286014, 0.1424675, -0.1589120, 0.0472983

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1223899, upper bound: 0.1380855
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225864, upper bound: 0.1382508
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0006505, 0.0188606, -0.0529753, 0.0478245, -0.0484750, 0.0718358
1: -0.0074290, 0.0164821, -0.0401546, 0.0348141, -0.0422431, 0.0566367
2: 0.0060065, 0.0335209, -0.0243841, 0.1249075, -0.1189010, 0.0579050
3: -0.0279165, 0.0116459, -0.0620886, 0.0457394, -0.0736559, 0.0737344
4: -0.0155779, 0.0068703, -0.0738202, 0.0175014, -0.0330792, 0.0806905
5: -0.0012776, 0.0154287, -0.0270478, 0.0301356, -0.0314132, 0.0424765
6: -0.0148937, 0.0281489, -0.0201164, 0.0652117, -0.0801054, 0.0482653
7: -0.0196854, 0.0074638, -0.0570109, 0.0355974, -0.0552828, 0.0644747
8: 0.9344307, 0.9901603, 0.8144474, 0.9911669, -0.0567362, 0.1757129
9: -0.0169174, 0.0204018, -0.0287710, 0.1434909, -0.1604083, 0.0491729

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1222204, upper bound: 0.1380858
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1224036, upper bound: 0.1382508
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0005356, 0.0174531, -0.0361533, 0.0393874, -0.0399230, 0.0536064
1: -0.0068692, 0.0153094, -0.0290306, 0.0310887, -0.0379579, 0.0443400
2: 0.0065546, 0.0321588, -0.0144257, 0.0956662, -0.0891116, 0.0465845
3: -0.0262104, 0.0098755, -0.0516839, 0.0386069, -0.0648172, 0.0615594
4: -0.0147856, 0.0059808, -0.0532942, 0.0159881, -0.0307737, 0.0592749
5: -0.0007190, 0.0148735, -0.0182573, 0.0279495, -0.0286685, 0.0331308
6: -0.0146943, 0.0264273, -0.0174445, 0.0555062, -0.0702005, 0.0438718
7: -0.0189206, 0.0072527, -0.0475535, 0.0251960, -0.0441166, 0.0548062
8: 0.9383700, 0.9901789, 0.8442495, 0.9912589, -0.0528889, 0.1459294
9: -0.0164445, 0.0186969, -0.0239953, 0.1094170, -0.1258616, 0.0426922

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1227966, upper bound: 0.1385101
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1230201, upper bound: 0.1387187
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0006505, 0.0188606, -0.0366430, 0.0396683, -0.0403188, 0.0555035
1: -0.0074290, 0.0164821, -0.0293588, 0.0312360, -0.0386649, 0.0458409
2: 0.0060065, 0.0335209, -0.0147387, 0.0965284, -0.0905219, 0.0482596
3: -0.0279165, 0.0116459, -0.0520198, 0.0388387, -0.0667551, 0.0636657
4: -0.0155779, 0.0068703, -0.0539060, 0.0160446, -0.0316225, 0.0607763
5: -0.0012776, 0.0154287, -0.0185280, 0.0280360, -0.0293136, 0.0339566
6: -0.0148937, 0.0281489, -0.0175256, 0.0558096, -0.0707033, 0.0456745
7: -0.0196854, 0.0074638, -0.0478374, 0.0254935, -0.0451790, 0.0553012
8: 0.9344307, 0.9901603, 0.8433545, 0.9912711, -0.0568405, 0.1468058
9: -0.0169174, 0.0204018, -0.0241522, 0.1104216, -0.1273390, 0.0445541

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1225697, upper bound: 0.1385105
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1227664, upper bound: 0.1387198
time: 1.54 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0016705, 0.0235792, -0.0524759, 0.0475339, -0.0492045, 0.0760551
1: -0.0099943, 0.0212966, -0.0398181, 0.0346652, -0.0446595, 0.0611147
2: 0.0032168, 0.0388687, -0.0240615, 0.1240271, -0.1208103, 0.0629302
3: -0.0327386, 0.0189410, -0.0617415, 0.0455026, -0.0782412, 0.0806825
4: -0.0186430, 0.0107790, -0.0731925, 0.0174440, -0.0360870, 0.0839715
5: -0.0029578, 0.0205506, -0.0267699, 0.0300486, -0.0330064, 0.0473205
6: -0.0141407, 0.0350338, -0.0199964, 0.0649016, -0.0790423, 0.0550303
7: -0.0246641, 0.0076440, -0.0567209, 0.0352578, -0.0599219, 0.0643649
8: 0.9147544, 0.9899706, 0.8153608, 0.9911544, -0.0764000, 0.1746098
9: -0.0176094, 0.0294052, -0.0286014, 0.1424675, -0.1600768, 0.0580066

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1306992, upper bound: 0.1408940
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1309415, upper bound: 0.1411440
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0017889, 0.0251851, -0.0529753, 0.0478245, -0.0496134, 0.0781604
1: -0.0105527, 0.0230072, -0.0401546, 0.0348141, -0.0453669, 0.0631618
2: 0.0021480, 0.0402471, -0.0243841, 0.1249075, -0.1227595, 0.0646312
3: -0.0345387, 0.0209242, -0.0620886, 0.0457394, -0.0802781, 0.0830128
4: -0.0196393, 0.0117172, -0.0738202, 0.0175014, -0.0371407, 0.0855373
5: -0.0036395, 0.0219487, -0.0270478, 0.0301356, -0.0337750, 0.0489965
6: -0.0143910, 0.0367453, -0.0201164, 0.0652117, -0.0796027, 0.0568617
7: -0.0254618, 0.0081613, -0.0570109, 0.0355974, -0.0610592, 0.0651722
8: 0.9107576, 0.9901012, 0.8144474, 0.9911669, -0.0804093, 0.1756538
9: -0.0181048, 0.0318282, -0.0287710, 0.1434909, -0.1615957, 0.0605992

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1302965, upper bound: 0.1408911
time: 2.12 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305144, upper bound: 0.1411386
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0016705, 0.0235792, -0.0361533, 0.0393874, -0.0410579, 0.0597325
1: -0.0099943, 0.0212966, -0.0290306, 0.0310887, -0.0410830, 0.0503272
2: 0.0032168, 0.0388687, -0.0144257, 0.0956662, -0.0924494, 0.0532944
3: -0.0327386, 0.0189410, -0.0516839, 0.0386069, -0.0713455, 0.0706249
4: -0.0186430, 0.0107790, -0.0532942, 0.0159881, -0.0346311, 0.0640731
5: -0.0029578, 0.0205506, -0.0182573, 0.0279495, -0.0309074, 0.0388079
6: -0.0141407, 0.0350338, -0.0174445, 0.0555062, -0.0696469, 0.0524783
7: -0.0246641, 0.0076440, -0.0475535, 0.0251960, -0.0498601, 0.0551975
8: 0.9147544, 0.9899706, 0.8442495, 0.9912589, -0.0765045, 0.1457211
9: -0.0176094, 0.0294052, -0.0239953, 0.1094170, -0.1270264, 0.0534005

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1309609, upper bound: 0.1409707
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1312149, upper bound: 0.1411827
time: 1.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0017889, 0.0251851, -0.0366430, 0.0396683, -0.0414572, 0.0618281
1: -0.0105527, 0.0230072, -0.0293588, 0.0312360, -0.0417887, 0.0523660
2: 0.0021480, 0.0402471, -0.0147387, 0.0965284, -0.0943804, 0.0549858
3: -0.0345387, 0.0209242, -0.0520198, 0.0388387, -0.0733774, 0.0729440
4: -0.0196393, 0.0117172, -0.0539060, 0.0160446, -0.0356839, 0.0656232
5: -0.0036395, 0.0219487, -0.0185280, 0.0280360, -0.0316754, 0.0404766
6: -0.0143910, 0.0367453, -0.0175256, 0.0558096, -0.0702006, 0.0542709
7: -0.0254618, 0.0081613, -0.0478374, 0.0254935, -0.0509553, 0.0559987
8: 0.9107576, 0.9901012, 0.8433545, 0.9912711, -0.0805135, 0.1467467
9: -0.0181048, 0.0318282, -0.0241522, 0.1104216, -0.1285264, 0.0559805

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305508, upper bound: 0.1409683
time: 2.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1307813, upper bound: 0.1411806
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0014535, 0.0221213, -0.0524759, 0.0475339, -0.0489874, 0.0745972
1: -0.0092692, 0.0197226, -0.0398181, 0.0346652, -0.0439344, 0.0595407
2: 0.0041365, 0.0372919, -0.0240615, 0.1240271, -0.1198906, 0.0613534
3: -0.0311495, 0.0167863, -0.0617415, 0.0455026, -0.0766520, 0.0785278
4: -0.0176634, 0.0096709, -0.0731925, 0.0174440, -0.0351074, 0.0828634
5: -0.0024441, 0.0189685, -0.0267699, 0.0300486, -0.0324927, 0.0457384
6: -0.0146341, 0.0330249, -0.0199964, 0.0649016, -0.0795357, 0.0530213
7: -0.0235000, 0.0078877, -0.0567209, 0.0352578, -0.0587578, 0.0646087
8: 0.9201922, 0.9903095, 0.8153608, 0.9911544, -0.0709622, 0.1749488
9: -0.0175104, 0.0267349, -0.0286014, 0.1424675, -0.1599778, 0.0553363

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1307560, upper bound: 0.1410616
time: 2.37 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1310072, upper bound: 0.1412868
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0015732, 0.0237563, -0.0529753, 0.0478245, -0.0493977, 0.0767316
1: -0.0098613, 0.0214836, -0.0401546, 0.0348141, -0.0446754, 0.0616382
2: 0.0030476, 0.0387425, -0.0243841, 0.1249075, -0.1218598, 0.0631266
3: -0.0330199, 0.0188173, -0.0620886, 0.0457394, -0.0787593, 0.0809059
4: -0.0187060, 0.0106305, -0.0738202, 0.0175014, -0.0362073, 0.0844507
5: -0.0031247, 0.0204359, -0.0270478, 0.0301356, -0.0332602, 0.0474836
6: -0.0148888, 0.0348182, -0.0201164, 0.0652117, -0.0801005, 0.0549346
7: -0.0242951, 0.0084041, -0.0570109, 0.0355974, -0.0598924, 0.0654150
8: 0.9159893, 0.9904320, 0.8144474, 0.9911669, -0.0751776, 0.1759846
9: -0.0180057, 0.0292046, -0.0287710, 0.1434909, -0.1614966, 0.0579757

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1303345, upper bound: 0.1410613
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1305546, upper bound: 0.1412839
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0014535, 0.0221213, -0.0361533, 0.0393874, -0.0408409, 0.0582745
1: -0.0092692, 0.0197226, -0.0290306, 0.0310887, -0.0403579, 0.0487532
2: 0.0041365, 0.0372919, -0.0144257, 0.0956662, -0.0915296, 0.0517176
3: -0.0311495, 0.0167863, -0.0516839, 0.0386069, -0.0697563, 0.0684702
4: -0.0176634, 0.0096709, -0.0532942, 0.0159881, -0.0336515, 0.0629651
5: -0.0024441, 0.0189685, -0.0182573, 0.0279495, -0.0303936, 0.0372258
6: -0.0146341, 0.0330249, -0.0174445, 0.0555062, -0.0701403, 0.0504693
7: -0.0235000, 0.0078877, -0.0475535, 0.0251960, -0.0486960, 0.0554413
8: 0.9201922, 0.9903095, 0.8442495, 0.9912589, -0.0710667, 0.1460600
9: -0.0175104, 0.0267349, -0.0239953, 0.1094170, -0.1269274, 0.0507302

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1310625, upper bound: 0.1414100
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1313337, upper bound: 0.1416215
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0015732, 0.0237563, -0.0366430, 0.0396683, -0.0412415, 0.0603993
1: -0.0098613, 0.0214836, -0.0293588, 0.0312360, -0.0410973, 0.0508424
2: 0.0030476, 0.0387425, -0.0147387, 0.0965284, -0.0934807, 0.0534812
3: -0.0330199, 0.0188173, -0.0520198, 0.0388387, -0.0718586, 0.0708371
4: -0.0187060, 0.0106305, -0.0539060, 0.0160446, -0.0347506, 0.0645366
5: -0.0031247, 0.0204359, -0.0185280, 0.0280360, -0.0311606, 0.0389638
6: -0.0148888, 0.0348182, -0.0175256, 0.0558096, -0.0706984, 0.0523437
7: -0.0242951, 0.0084041, -0.0478374, 0.0254935, -0.0497886, 0.0562415
8: 0.9159893, 0.9904320, 0.8433545, 0.9912711, -0.0752819, 0.1470776
9: -0.0180057, 0.0292046, -0.0241522, 0.1104216, -0.1284273, 0.0533569

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1306289, upper bound: 0.1414100
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1308487, upper bound: 0.1416182
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0008567, 0.0197781, -0.0640162, 0.0435861
1: -0.0342613, 0.0321695, -0.0079757, 0.0172442, -0.0515055, 0.0401452
2: -0.0187495, 0.1094280, 0.0057032, 0.0346615, -0.0534109, 0.1037248
3: -0.0559870, 0.0415918, -0.0289688, 0.0131093, -0.0690963, 0.0705607
4: -0.0628106, 0.0164810, -0.0161476, 0.0077036, -0.0705142, 0.0326286
5: -0.0221853, 0.0285809, -0.0015685, 0.0160455, -0.0382308, 0.0301494
6: -0.0180209, 0.0597589, -0.0145612, 0.0296252, -0.0476461, 0.0743201
7: -0.0519036, 0.0298092, -0.0207230, 0.0069684, -0.0588719, 0.0505323
8: 0.8305457, 0.9909523, 0.9301941, 0.9897332, -0.1591875, 0.0607582
9: -0.0258906, 0.1254902, -0.0169320, 0.0220267, -0.0479173, 0.1424222

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377536, upper bound: 0.1221816
time: 2.19 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377542, upper bound: 0.1220034
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0008707, 0.0198998, -0.0687216, 0.0462788
1: -0.0373475, 0.0335700, -0.0080284, 0.0173411, -0.0546885, 0.0415984
2: -0.0216974, 0.1175703, 0.0056564, 0.0347844, -0.0564817, 0.1119139
3: -0.0591903, 0.0437662, -0.0291138, 0.0132635, -0.0724538, 0.0728801
4: -0.0685832, 0.0170204, -0.0162181, 0.0077863, -0.0763695, 0.0332386
5: -0.0247323, 0.0294066, -0.0016230, 0.0160990, -0.0408313, 0.0310296
6: -0.0190791, 0.0626204, -0.0146490, 0.0297791, -0.0488582, 0.0772694
7: -0.0545923, 0.0327939, -0.0208000, 0.0070211, -0.0616134, 0.0535939
8: 0.8220751, 0.9910482, 0.9298151, 0.9897382, -0.1676631, 0.0612332
9: -0.0273315, 0.1349577, -0.0169754, 0.0221925, -0.0495240, 0.1519331

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379433, upper bound: 0.1224017
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379473, upper bound: 0.1222156
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0006656, 0.0186768, -0.0629148, 0.0433950
1: -0.0342613, 0.0321695, -0.0074045, 0.0162945, -0.0505558, 0.0395740
2: -0.0187495, 0.1094280, 0.0060830, 0.0334064, -0.0521559, 0.1033449
3: -0.0559870, 0.0415918, -0.0276802, 0.0114410, -0.0674280, 0.0692721
4: -0.0628106, 0.0164810, -0.0154971, 0.0068130, -0.0696236, 0.0319782
5: -0.0221853, 0.0285809, -0.0012254, 0.0154197, -0.0376050, 0.0298063
6: -0.0180209, 0.0597589, -0.0151285, 0.0279960, -0.0460168, 0.0748873
7: -0.0519036, 0.0298092, -0.0196855, 0.0075476, -0.0594512, 0.0494947
8: 0.8305457, 0.9909523, 0.9345381, 0.9901966, -0.1596509, 0.0564142
9: -0.0258906, 0.1254902, -0.0169001, 0.0202866, -0.0461772, 0.1423903

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380855, upper bound: 0.1223899
time: 2.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380858, upper bound: 0.1222204
time: 2.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0006787, 0.0187909, -0.0676127, 0.0460869
1: -0.0373475, 0.0335700, -0.0074538, 0.0163851, -0.0537326, 0.0410239
2: -0.0216974, 0.1175703, 0.0060388, 0.0335214, -0.0552188, 0.1115315
3: -0.0591903, 0.0437662, -0.0278158, 0.0115854, -0.0707757, 0.0715821
4: -0.0685832, 0.0170204, -0.0155632, 0.0068907, -0.0754740, 0.0325836
5: -0.0247323, 0.0294066, -0.0012751, 0.0154698, -0.0402021, 0.0306817
6: -0.0190791, 0.0626204, -0.0152177, 0.0281398, -0.0472189, 0.0778380
7: -0.0545923, 0.0327939, -0.0197580, 0.0076004, -0.0621927, 0.0525519
8: 0.8220751, 0.9910482, 0.9341831, 0.9902020, -0.1681269, 0.0568652
9: -0.0273315, 0.1349577, -0.0169410, 0.0204402, -0.0477717, 0.1518986

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382508, upper bound: 0.1225864
time: 1.41 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382508, upper bound: 0.1224036
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0292268, 0.0358424, -0.0008567, 0.0197781, -0.0490049, 0.0366990
1: -0.0250228, 0.0289604, -0.0079757, 0.0172442, -0.0422670, 0.0369361
2: -0.0106620, 0.0832981, 0.0057032, 0.0346615, -0.0453235, 0.0775949
3: -0.0472686, 0.0353324, -0.0289688, 0.0131093, -0.0603779, 0.0643013
4: -0.0458468, 0.0151690, -0.0161476, 0.0077036, -0.0535504, 0.0313165
5: -0.0147463, 0.0266969, -0.0015685, 0.0160455, -0.0307918, 0.0282654
6: -0.0162500, 0.0514486, -0.0145612, 0.0296252, -0.0458752, 0.0660098
7: -0.0435328, 0.0212903, -0.0207230, 0.0069684, -0.0505011, 0.0420133
8: 0.8569218, 0.9910862, 0.9301941, 0.9897332, -0.1328114, 0.0608921
9: -0.0222331, 0.0952172, -0.0169320, 0.0220267, -0.0442598, 0.1121492

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377987, upper bound: 0.1224619
time: 2.40 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377996, upper bound: 0.1222866
time: 2.33 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0323218, 0.0372962, -0.0008707, 0.0198998, -0.0522216, 0.0381669
1: -0.0265895, 0.0299472, -0.0080284, 0.0173411, -0.0439306, 0.0379756
2: -0.0121145, 0.0889088, 0.0056564, 0.0347844, -0.0468989, 0.0832524
3: -0.0491322, 0.0367937, -0.0291138, 0.0132635, -0.0623957, 0.0659075
4: -0.0487463, 0.0155466, -0.0162181, 0.0077863, -0.0565326, 0.0317647
5: -0.0162063, 0.0272790, -0.0016230, 0.0160990, -0.0323053, 0.0289021
6: -0.0167509, 0.0531899, -0.0146490, 0.0297791, -0.0465300, 0.0678390
7: -0.0453369, 0.0228843, -0.0208000, 0.0070211, -0.0523580, 0.0436843
8: 0.8512510, 0.9911513, 0.9298151, 0.9897382, -0.1384872, 0.0613362
9: -0.0228249, 0.1015424, -0.0169754, 0.0221925, -0.0450174, 0.1185179

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379746, upper bound: 0.1227176
time: 2.61 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379788, upper bound: 0.1225370
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0292268, 0.0358424, -0.0006656, 0.0186768, -0.0479036, 0.0365080
1: -0.0250228, 0.0289604, -0.0074045, 0.0162945, -0.0413173, 0.0363649
2: -0.0106620, 0.0832981, 0.0060830, 0.0334064, -0.0440684, 0.0772150
3: -0.0472686, 0.0353324, -0.0276802, 0.0114410, -0.0587096, 0.0630127
4: -0.0458468, 0.0151690, -0.0154971, 0.0068130, -0.0526598, 0.0306661
5: -0.0147463, 0.0266969, -0.0012254, 0.0154197, -0.0301660, 0.0279223
6: -0.0162500, 0.0514486, -0.0151285, 0.0279960, -0.0442459, 0.0665770
7: -0.0435328, 0.0212903, -0.0196855, 0.0075476, -0.0510804, 0.0409757
8: 0.8569218, 0.9910862, 0.9345381, 0.9901966, -0.1332748, 0.0565481
9: -0.0222331, 0.0952172, -0.0169001, 0.0202866, -0.0425197, 0.1121173

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385072, upper bound: 0.1229121
time: 1.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385073, upper bound: 0.1227131
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0323218, 0.0372962, -0.0006787, 0.0187909, -0.0511126, 0.0379749
1: -0.0265895, 0.0299472, -0.0074538, 0.0163851, -0.0429747, 0.0374011
2: -0.0121145, 0.0889088, 0.0060388, 0.0335214, -0.0456360, 0.0828700
3: -0.0491322, 0.0367937, -0.0278158, 0.0115854, -0.0607176, 0.0646095
4: -0.0487463, 0.0155466, -0.0155632, 0.0068907, -0.0556371, 0.0311098
5: -0.0162063, 0.0272790, -0.0012751, 0.0154698, -0.0316761, 0.0285541
6: -0.0167509, 0.0531899, -0.0152177, 0.0281398, -0.0448907, 0.0684076
7: -0.0453369, 0.0228843, -0.0197580, 0.0076004, -0.0529373, 0.0426423
8: 0.8512510, 0.9911513, 0.9341831, 0.9902020, -0.1389510, 0.0569682
9: -0.0228249, 0.1015424, -0.0169410, 0.0204402, -0.0432651, 0.1184834

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387037, upper bound: 0.1231501
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387061, upper bound: 0.1229238
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0018338, 0.0252464, -0.0694844, 0.0445632
1: -0.0342613, 0.0321695, -0.0106659, 0.0230073, -0.0572686, 0.0428354
2: -0.0187495, 0.1094280, 0.0021292, 0.0404291, -0.0591786, 0.1072988
3: -0.0559870, 0.0415918, -0.0345580, 0.0210890, -0.0770760, 0.0761498
4: -0.0628106, 0.0164810, -0.0197015, 0.0118304, -0.0746410, 0.0361826
5: -0.0221853, 0.0285809, -0.0036621, 0.0220841, -0.0442694, 0.0322430
6: -0.0180209, 0.0597589, -0.0146713, 0.0369796, -0.0550005, 0.0744302
7: -0.0519036, 0.0298092, -0.0256457, 0.0082953, -0.0601988, 0.0554550
8: 0.8305457, 0.9909523, 0.9099458, 0.9901356, -0.1595899, 0.0810065
9: -0.0258906, 0.1254902, -0.0181483, 0.0320562, -0.0579468, 0.1436384

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1419908, upper bound: 0.1301044
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1419935, upper bound: 0.1296962
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0018666, 0.0255524, -0.0743742, 0.0472748
1: -0.0373475, 0.0335700, -0.0107909, 0.0233222, -0.0606697, 0.0443609
2: -0.0216974, 0.1175703, 0.0019298, 0.0407174, -0.0624148, 0.1156405
3: -0.0591903, 0.0437662, -0.0348904, 0.0214866, -0.0806769, 0.0786567
4: -0.0685832, 0.0170204, -0.0198965, 0.0120266, -0.0806098, 0.0369170
5: -0.0247323, 0.0294066, -0.0037908, 0.0223690, -0.0471013, 0.0331974
6: -0.0190791, 0.0626204, -0.0147785, 0.0373392, -0.0564183, 0.0773988
7: -0.0545923, 0.0327939, -0.0258353, 0.0084135, -0.0630058, 0.0586292
8: 0.8220751, 0.9910482, 0.9090434, 0.9901649, -0.1680898, 0.0820048
9: -0.0273315, 0.1349577, -0.0182422, 0.0325506, -0.0598822, 0.1531999

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1422330, upper bound: 0.1304121
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1422361, upper bound: 0.1299922
time: 1.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0016156, 0.0237838, -0.0680218, 0.0443450
1: -0.0342613, 0.0321695, -0.0099361, 0.0214421, -0.0557033, 0.0421056
2: -0.0187495, 0.1094280, 0.0030489, 0.0388441, -0.0575936, 0.1063791
3: -0.0559870, 0.0415918, -0.0329711, 0.0189339, -0.0749209, 0.0745629
4: -0.0628106, 0.0164810, -0.0187259, 0.0107149, -0.0735255, 0.0352070
5: -0.0221853, 0.0285809, -0.0031425, 0.0205033, -0.0426886, 0.0317234
6: -0.0180209, 0.0597589, -0.0151532, 0.0349730, -0.0529939, 0.0749121
7: -0.0519036, 0.0298092, -0.0244746, 0.0085221, -0.0604257, 0.0542839
8: 0.8305457, 0.9909523, 0.9154189, 0.9904697, -0.1599240, 0.0755334
9: -0.0258906, 0.1254902, -0.0180523, 0.0293403, -0.0552309, 0.1435425

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1421125, upper bound: 0.1301547
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1421132, upper bound: 0.1297451
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0016469, 0.0240765, -0.0728982, 0.0470551
1: -0.0373475, 0.0335700, -0.0100552, 0.0217439, -0.0590913, 0.0436252
2: -0.0216974, 0.1175703, 0.0028583, 0.0391192, -0.0608166, 0.1147120
3: -0.0591903, 0.0437662, -0.0332908, 0.0193161, -0.0785064, 0.0770570
4: -0.0685832, 0.0170204, -0.0189128, 0.0109020, -0.0794852, 0.0359332
5: -0.0247323, 0.0294066, -0.0032642, 0.0207770, -0.0455093, 0.0326708
6: -0.0190791, 0.0626204, -0.0152594, 0.0353205, -0.0543996, 0.0778798
7: -0.0545923, 0.0327939, -0.0246563, 0.0086383, -0.0632306, 0.0574502
8: 0.8220751, 0.9910482, 0.9145586, 0.9904979, -0.1684228, 0.0764896
9: -0.0273315, 0.1349577, -0.0181429, 0.0298088, -0.0571403, 0.1531005

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1423432, upper bound: 0.1304756
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1423457, upper bound: 0.1300373
time: 1.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0292268, 0.0358424, -0.0018338, 0.0252464, -0.0544732, 0.0376762
1: -0.0250228, 0.0289604, -0.0106659, 0.0230073, -0.0480300, 0.0396264
2: -0.0106620, 0.0832981, 0.0021292, 0.0404291, -0.0510911, 0.0811689
3: -0.0472686, 0.0353324, -0.0345580, 0.0210890, -0.0683576, 0.0698904
4: -0.0458468, 0.0151690, -0.0197015, 0.0118304, -0.0576771, 0.0348705
5: -0.0147463, 0.0266969, -0.0036621, 0.0220841, -0.0368305, 0.0303590
6: -0.0162500, 0.0514486, -0.0146713, 0.0369796, -0.0532296, 0.0661199
7: -0.0435328, 0.0212903, -0.0256457, 0.0082953, -0.0518281, 0.0469360
8: 0.8569218, 0.9910862, 0.9099458, 0.9901356, -0.1332138, 0.0811403
9: -0.0222331, 0.0952172, -0.0181483, 0.0320562, -0.0542894, 0.1133655

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1420873, upper bound: 0.1303891
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1420905, upper bound: 0.1300095
time: 1.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0323218, 0.0372962, -0.0018666, 0.0255524, -0.0578742, 0.0391628
1: -0.0265895, 0.0299472, -0.0107909, 0.0233222, -0.0499118, 0.0407381
2: -0.0121145, 0.0889088, 0.0019298, 0.0407174, -0.0528320, 0.0869790
3: -0.0491322, 0.0367937, -0.0348904, 0.0214866, -0.0706187, 0.0716841
4: -0.0487463, 0.0155466, -0.0198965, 0.0120266, -0.0607730, 0.0354431
5: -0.0162063, 0.0272790, -0.0037908, 0.0223690, -0.0385753, 0.0310698
6: -0.0167509, 0.0531899, -0.0147785, 0.0373392, -0.0540901, 0.0679684
7: -0.0453369, 0.0228843, -0.0258353, 0.0084135, -0.0537504, 0.0487196
8: 0.8512510, 0.9911513, 0.9090434, 0.9901649, -0.1389139, 0.0821078
9: -0.0228249, 0.1015424, -0.0182422, 0.0325506, -0.0553756, 0.1197847

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1422917, upper bound: 0.1306844
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1422998, upper bound: 0.1302817
time: 1.37 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0292268, 0.0358424, -0.0016156, 0.0237838, -0.0530106, 0.0374580
1: -0.0250228, 0.0289604, -0.0099361, 0.0214421, -0.0464648, 0.0388965
2: -0.0106620, 0.0832981, 0.0030489, 0.0388441, -0.0495061, 0.0802491
3: -0.0472686, 0.0353324, -0.0329711, 0.0189339, -0.0662025, 0.0683035
4: -0.0458468, 0.0151690, -0.0187259, 0.0107149, -0.0565617, 0.0338949
5: -0.0147463, 0.0266969, -0.0031425, 0.0205033, -0.0352497, 0.0298394
6: -0.0162500, 0.0514486, -0.0151532, 0.0349730, -0.0512230, 0.0666017
7: -0.0435328, 0.0212903, -0.0244746, 0.0085221, -0.0520549, 0.0457649
8: 0.8569218, 0.9910862, 0.9154189, 0.9904697, -0.1335479, 0.0756673
9: -0.0222331, 0.0952172, -0.0180523, 0.0293403, -0.0515735, 0.1132695

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1425093, upper bound: 0.1306197
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1425126, upper bound: 0.1302207
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0323218, 0.0372962, -0.0016469, 0.0240765, -0.0563982, 0.0389431
1: -0.0265895, 0.0299472, -0.0100552, 0.0217439, -0.0483334, 0.0400024
2: -0.0121145, 0.0889088, 0.0028583, 0.0391192, -0.0512338, 0.0860505
3: -0.0491322, 0.0367937, -0.0332908, 0.0193161, -0.0684482, 0.0700845
4: -0.0487463, 0.0155466, -0.0189128, 0.0109020, -0.0596483, 0.0344594
5: -0.0162063, 0.0272790, -0.0032642, 0.0207770, -0.0369833, 0.0305432
6: -0.0167509, 0.0531899, -0.0152594, 0.0353205, -0.0520714, 0.0684493
7: -0.0453369, 0.0228843, -0.0246563, 0.0086383, -0.0539752, 0.0475406
8: 0.8512510, 0.9911513, 0.9145586, 0.9904979, -0.1392469, 0.0765927
9: -0.0228249, 0.1015424, -0.0181429, 0.0298088, -0.0526337, 0.1196853

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1427015, upper bound: 0.1309327
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1427028, upper bound: 0.1305003
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0021067, 0.0226430, -0.0524763, 0.0475461, -0.0496528, 0.0751194
1: -0.0101733, 0.0202153, -0.0398204, 0.0346778, -0.0448511, 0.0600356
2: 0.0039695, 0.0386867, -0.0240675, 0.1240345, -0.1200650, 0.0627542
3: -0.0310013, 0.0188877, -0.0617548, 0.0455119, -0.0765133, 0.0806425
4: -0.0180117, 0.0110294, -0.0731988, 0.0174483, -0.0354600, 0.0842282
5: -0.0024120, 0.0204676, -0.0267742, 0.0300559, -0.0324679, 0.0472418
6: -0.0147673, 0.0350947, -0.0200035, 0.0649085, -0.0796758, 0.0550982
7: -0.0261750, 0.0076698, -0.0567239, 0.0352673, -0.0614424, 0.0643937
8: 0.9121118, 0.9899802, 0.8153460, 0.9911584, -0.0790466, 0.1746342
9: -0.0165162, 0.0295205, -0.0286096, 0.1424822, -0.1589984, 0.0581301

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1348831, upper bound: 0.1402267
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1350729, upper bound: 0.1404866
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0022120, 0.0240952, -0.0529757, 0.0478367, -0.0500486, 0.0770710
1: -0.0106865, 0.0217594, -0.0401569, 0.0348267, -0.0455132, 0.0619164
2: 0.0030123, 0.0399501, -0.0243901, 0.1249149, -0.1219026, 0.0643402
3: -0.0326759, 0.0206355, -0.0621019, 0.0457487, -0.0784246, 0.0827374
4: -0.0189229, 0.0118769, -0.0738265, 0.0175057, -0.0364286, 0.0857034
5: -0.0030103, 0.0217384, -0.0270521, 0.0301428, -0.0331532, 0.0487905
6: -0.0149828, 0.0366543, -0.0201235, 0.0652186, -0.0802014, 0.0567777
7: -0.0268758, 0.0081157, -0.0570138, 0.0356069, -0.0624827, 0.0651296
8: 0.9084176, 0.9900976, 0.8144326, 0.9911709, -0.0827534, 0.1756650
9: -0.0170030, 0.0317037, -0.0287792, 0.1435056, -0.1605087, 0.0604830

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1347767, upper bound: 0.1402267
time: 1.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1349645, upper bound: 0.1404863
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0021067, 0.0226430, -0.0361535, 0.0393951, -0.0415017, 0.0587966
1: -0.0101733, 0.0202153, -0.0290321, 0.0310964, -0.0412697, 0.0492474
2: 0.0039695, 0.0386867, -0.0144299, 0.0956708, -0.0917014, 0.0531166
3: -0.0310013, 0.0188877, -0.0516926, 0.0386126, -0.0696139, 0.0705803
4: -0.0180117, 0.0110294, -0.0532982, 0.0159908, -0.0340025, 0.0643276
5: -0.0024120, 0.0204676, -0.0182603, 0.0279541, -0.0303662, 0.0387279
6: -0.0147673, 0.0350947, -0.0174481, 0.0555105, -0.0702778, 0.0525428
7: -0.0261750, 0.0076698, -0.0475553, 0.0252021, -0.0513772, 0.0552251
8: 0.9121118, 0.9899802, 0.8442402, 0.9912613, -0.0791495, 0.1457400
9: -0.0165162, 0.0295205, -0.0240001, 0.1094263, -0.1259424, 0.0535206

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1351899, upper bound: 0.1402406
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354072, upper bound: 0.1405005
time: 1.38 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0022120, 0.0240952, -0.0366432, 0.0396761, -0.0418880, 0.0607384
1: -0.0106865, 0.0217594, -0.0293602, 0.0312437, -0.0419302, 0.0511197
2: 0.0030123, 0.0399501, -0.0147429, 0.0965330, -0.0935207, 0.0546930
3: -0.0326759, 0.0206355, -0.0520285, 0.0388444, -0.0715203, 0.0726641
4: -0.0189229, 0.0118769, -0.0539100, 0.0160473, -0.0349702, 0.0657869
5: -0.0030103, 0.0217384, -0.0185309, 0.0280405, -0.0310509, 0.0402693
6: -0.0149828, 0.0366543, -0.0175293, 0.0558140, -0.0707967, 0.0541836
7: -0.0268758, 0.0081157, -0.0478392, 0.0254996, -0.0523754, 0.0559550
8: 0.9084176, 0.9900976, 0.8433452, 0.9912735, -0.0828559, 0.1467525
9: -0.0170030, 0.0317037, -0.0241571, 0.1104309, -0.1274339, 0.0558608

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1350662, upper bound: 0.1402405
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1352903, upper bound: 0.1404998
time: 1.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0018603, 0.0212831, -0.0524763, 0.0475461, -0.0494064, 0.0737594
1: -0.0094697, 0.0187419, -0.0398204, 0.0346778, -0.0441475, 0.0585622
2: 0.0048193, 0.0371824, -0.0240675, 0.1240345, -0.1192152, 0.0612499
3: -0.0295685, 0.0167865, -0.0617548, 0.0455119, -0.0750805, 0.0785413
4: -0.0171059, 0.0099592, -0.0731988, 0.0174483, -0.0345542, 0.0831580
5: -0.0019715, 0.0189420, -0.0267742, 0.0300559, -0.0320274, 0.0457162
6: -0.0152194, 0.0331261, -0.0200035, 0.0649085, -0.0801280, 0.0531297
7: -0.0249141, 0.0078765, -0.0567239, 0.0352673, -0.0601814, 0.0646003
8: 0.9174590, 0.9902773, 0.8153460, 0.9911584, -0.0736994, 0.1749313
9: -0.0164673, 0.0269324, -0.0286096, 0.1424822, -0.1589495, 0.0555420

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1352737, upper bound: 0.1410616
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354427, upper bound: 0.1412587
time: 1.99 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0019611, 0.0227174, -0.0529757, 0.0478367, -0.0497977, 0.0756931
1: -0.0099748, 0.0202797, -0.0401569, 0.0348267, -0.0448015, 0.0604366
2: 0.0038689, 0.0384339, -0.0243901, 0.1249149, -0.1210460, 0.0628240
3: -0.0312116, 0.0185331, -0.0621019, 0.0457487, -0.0769603, 0.0806350
4: -0.0180026, 0.0107829, -0.0738265, 0.0175057, -0.0355083, 0.0846094
5: -0.0025477, 0.0201981, -0.0270521, 0.0301428, -0.0326905, 0.0472502
6: -0.0154277, 0.0346740, -0.0201235, 0.0652186, -0.0806463, 0.0547975
7: -0.0256090, 0.0083204, -0.0570138, 0.0356069, -0.0612159, 0.0653342
8: 0.9138816, 0.9903927, 0.8144326, 0.9911709, -0.0772893, 0.1759601
9: -0.0169338, 0.0290747, -0.0287792, 0.1435056, -0.1604394, 0.0578539

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1352277, upper bound: 0.1410659
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1353880, upper bound: 0.1412669
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0018603, 0.0212831, -0.0361535, 0.0393951, -0.0412554, 0.0574366
1: -0.0094697, 0.0187419, -0.0290321, 0.0310964, -0.0405661, 0.0477740
2: 0.0048193, 0.0371824, -0.0144299, 0.0956708, -0.0908516, 0.0516123
3: -0.0295685, 0.0167865, -0.0516926, 0.0386126, -0.0681811, 0.0684791
4: -0.0171059, 0.0099592, -0.0532982, 0.0159908, -0.0330966, 0.0632574
5: -0.0019715, 0.0189420, -0.0182603, 0.0279541, -0.0299257, 0.0372023
6: -0.0152194, 0.0331261, -0.0174481, 0.0555105, -0.0707300, 0.0505743
7: -0.0249141, 0.0078765, -0.0475553, 0.0252021, -0.0501162, 0.0554318
8: 0.9174590, 0.9902773, 0.8442402, 0.9912613, -0.0738023, 0.1460372
9: -0.0164673, 0.0269324, -0.0240001, 0.1094263, -0.1258936, 0.0509325

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1358256, upper bound: 0.1415553
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1359943, upper bound: 0.1417495
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0019611, 0.0227174, -0.0366432, 0.0396761, -0.0416371, 0.0593606
1: -0.0099748, 0.0202797, -0.0293602, 0.0312437, -0.0412185, 0.0496399
2: 0.0038689, 0.0384339, -0.0147429, 0.0965330, -0.0926641, 0.0531768
3: -0.0312116, 0.0185331, -0.0520285, 0.0388444, -0.0700560, 0.0705616
4: -0.0180026, 0.0107829, -0.0539100, 0.0160473, -0.0340499, 0.0646930
5: -0.0025477, 0.0201981, -0.0185309, 0.0280405, -0.0305882, 0.0387290
6: -0.0154277, 0.0346740, -0.0175293, 0.0558140, -0.0712416, 0.0522033
7: -0.0256090, 0.0083204, -0.0478392, 0.0254996, -0.0511086, 0.0561596
8: 0.9138816, 0.9903927, 0.8433452, 0.9912735, -0.0773919, 0.1470476
9: -0.0169338, 0.0290747, -0.0241571, 0.1104309, -0.1273647, 0.0532318

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1357776, upper bound: 0.1415635
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1359312, upper bound: 0.1417640
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0022376, 0.0240012, -0.0682392, 0.0449670
1: -0.0342613, 0.0321695, -0.0107144, 0.0216263, -0.0558876, 0.0428839
2: -0.0187495, 0.1094280, 0.0030815, 0.0399494, -0.0586989, 0.1063465
3: -0.0559870, 0.0415918, -0.0324919, 0.0206453, -0.0766323, 0.0740838
4: -0.0628106, 0.0164810, -0.0188753, 0.0118838, -0.0746944, 0.0353564
5: -0.0221853, 0.0285809, -0.0029775, 0.0217218, -0.0439071, 0.0315583
6: -0.0180209, 0.0597589, -0.0152222, 0.0366958, -0.0547167, 0.0749810
7: -0.0519036, 0.0298092, -0.0269655, 0.0082006, -0.0601041, 0.0567747
8: 0.8305457, 0.9909523, 0.9082133, 0.9901152, -0.1595696, 0.0827391
9: -0.0258906, 0.1254902, -0.0169544, 0.0316712, -0.0575618, 0.1424446

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1435783, upper bound: 0.1338279
time: 2.12 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1435784, upper bound: 0.1337013
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0022585, 0.0241961, -0.0730179, 0.0476667
1: -0.0373475, 0.0335700, -0.0107930, 0.0218282, -0.0591756, 0.0443631
2: -0.0216974, 0.1175703, 0.0029539, 0.0401321, -0.0618294, 0.1146164
3: -0.0591903, 0.0437662, -0.0327061, 0.0208967, -0.0800870, 0.0764723
4: -0.0685832, 0.0170204, -0.0190003, 0.0120081, -0.0805913, 0.0360208
5: -0.0247323, 0.0294066, -0.0030589, 0.0219027, -0.0466349, 0.0324655
6: -0.0190791, 0.0626204, -0.0153214, 0.0369249, -0.0560040, 0.0779418
7: -0.0545923, 0.0327939, -0.0270843, 0.0082864, -0.0628787, 0.0598782
8: 0.8220751, 0.9910482, 0.9076468, 0.9901348, -0.1680596, 0.0834014
9: -0.0273315, 0.1349577, -0.0170160, 0.0319851, -0.0593167, 0.1519736

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1437918, upper bound: 0.1340063
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1437923, upper bound: 0.1338912
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0019895, 0.0226302, -0.0668682, 0.0447189
1: -0.0342613, 0.0321695, -0.0100070, 0.0201448, -0.0544061, 0.0421765
2: -0.0187495, 0.1094280, 0.0039368, 0.0384384, -0.0571879, 0.1054912
3: -0.0559870, 0.0415918, -0.0310487, 0.0185313, -0.0745183, 0.0726406
4: -0.0628106, 0.0164810, -0.0179641, 0.0108087, -0.0736193, 0.0344452
5: -0.0221853, 0.0285809, -0.0025207, 0.0201963, -0.0423816, 0.0311016
6: -0.0180209, 0.0597589, -0.0156725, 0.0347212, -0.0527420, 0.0754314
7: -0.0519036, 0.0298092, -0.0257011, 0.0083910, -0.0602945, 0.0555103
8: 0.8305457, 0.9909523, 0.9135737, 0.9904108, -0.1598651, 0.0773786
9: -0.0258906, 0.1254902, -0.0169015, 0.0290602, -0.0549508, 0.1423917

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439862, upper bound: 0.1340560
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1439956, upper bound: 0.1339822
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0020111, 0.0228301, -0.0716519, 0.0474192
1: -0.0373475, 0.0335700, -0.0100880, 0.0203515, -0.0576990, 0.0436581
2: -0.0216974, 0.1175703, 0.0038061, 0.0386262, -0.0603236, 0.1137642
3: -0.0591903, 0.0437662, -0.0312670, 0.0187905, -0.0779808, 0.0750333
4: -0.0685832, 0.0170204, -0.0180915, 0.0109365, -0.0795197, 0.0351120
5: -0.0247323, 0.0294066, -0.0026025, 0.0203822, -0.0451145, 0.0320091
6: -0.0190791, 0.0626204, -0.0157730, 0.0349574, -0.0540366, 0.0783933
7: -0.0545923, 0.0327939, -0.0258241, 0.0084808, -0.0630731, 0.0586180
8: 0.8220751, 0.9910482, 0.9129899, 0.9904315, -0.1683564, 0.0780584
9: -0.0273315, 0.1349577, -0.0169642, 0.0293804, -0.0567119, 0.1519218

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1441708, upper bound: 0.1342127
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1441779, upper bound: 0.1341551
time: 2.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0292268, 0.0358424, -0.0022376, 0.0240012, -0.0532280, 0.0380799
1: -0.0250228, 0.0289604, -0.0107144, 0.0216263, -0.0466491, 0.0396749
2: -0.0106620, 0.0832981, 0.0030815, 0.0399494, -0.0506114, 0.0802166
3: -0.0472686, 0.0353324, -0.0324919, 0.0206453, -0.0679139, 0.0678244
4: -0.0458468, 0.0151690, -0.0188753, 0.0118838, -0.0577306, 0.0340443
5: -0.0147463, 0.0266969, -0.0029775, 0.0217218, -0.0364682, 0.0296743
6: -0.0162500, 0.0514486, -0.0152222, 0.0366958, -0.0529458, 0.0666707
7: -0.0435328, 0.0212903, -0.0269655, 0.0082006, -0.0517334, 0.0482557
8: 0.8569218, 0.9910862, 0.9082133, 0.9901152, -0.1331934, 0.0828729
9: -0.0222331, 0.0952172, -0.0169544, 0.0316712, -0.0539043, 0.1121716

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1436689, upper bound: 0.1342521
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1436693, upper bound: 0.1341409
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0323218, 0.0372962, -0.0022585, 0.0241961, -0.0565178, 0.0395547
1: -0.0265895, 0.0299472, -0.0107930, 0.0218282, -0.0484177, 0.0407403
2: -0.0121145, 0.0889088, 0.0029539, 0.0401321, -0.0522466, 0.0859549
3: -0.0491322, 0.0367937, -0.0327061, 0.0208967, -0.0700289, 0.0694998
4: -0.0487463, 0.0155466, -0.0190003, 0.0120081, -0.0607545, 0.0345469
5: -0.0162063, 0.0272790, -0.0030589, 0.0219027, -0.0381090, 0.0303379
6: -0.0167509, 0.0531899, -0.0153214, 0.0369249, -0.0536758, 0.0685113
7: -0.0453369, 0.0228843, -0.0270843, 0.0082864, -0.0536233, 0.0499686
8: 0.8512510, 0.9911513, 0.9076468, 0.9901348, -0.1388838, 0.0835044
9: -0.0228249, 0.1015424, -0.0170160, 0.0319851, -0.0548101, 0.1185584

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438729, upper bound: 0.1344126
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1438730, upper bound: 0.1343098
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0292268, 0.0358424, -0.0019895, 0.0226302, -0.0518569, 0.0378319
1: -0.0250228, 0.0289604, -0.0100070, 0.0201448, -0.0451676, 0.0389675
2: -0.0106620, 0.0832981, 0.0039368, 0.0384384, -0.0491004, 0.0793613
3: -0.0472686, 0.0353324, -0.0310487, 0.0185313, -0.0657998, 0.0663812
4: -0.0458468, 0.0151690, -0.0179641, 0.0108087, -0.0566554, 0.0331331
5: -0.0147463, 0.0266969, -0.0025207, 0.0201963, -0.0349426, 0.0292175
6: -0.0162500, 0.0514486, -0.0156725, 0.0347212, -0.0509711, 0.0671211
7: -0.0435328, 0.0212903, -0.0257011, 0.0083910, -0.0519238, 0.0469914
8: 0.8569218, 0.9910862, 0.9135737, 0.9904108, -0.1334890, 0.0775124
9: -0.0222331, 0.0952172, -0.0169015, 0.0290602, -0.0512934, 0.1121187

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1446946, upper bound: 0.1347849
time: 1.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1446946, upper bound: 0.1347256
time: 2.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0323218, 0.0372962, -0.0020111, 0.0228301, -0.0551519, 0.0393073
1: -0.0265895, 0.0299472, -0.0100880, 0.0203515, -0.0469410, 0.0400353
2: -0.0121145, 0.0889088, 0.0038061, 0.0386262, -0.0507407, 0.0851027
3: -0.0491322, 0.0367937, -0.0312670, 0.0187905, -0.0679226, 0.0680607
4: -0.0487463, 0.0155466, -0.0180915, 0.0109365, -0.0596828, 0.0336381
5: -0.0162063, 0.0272790, -0.0026025, 0.0203822, -0.0365885, 0.0298815
6: -0.0167509, 0.0531899, -0.0157730, 0.0349574, -0.0517083, 0.0689629
7: -0.0453369, 0.0228843, -0.0258241, 0.0084808, -0.0538177, 0.0487083
8: 0.8512510, 0.9911513, 0.9129899, 0.9904315, -0.1391805, 0.0781614
9: -0.0228249, 0.1015424, -0.0169642, 0.0293804, -0.0522054, 0.1185066

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1449011, upper bound: 0.1349223
time: 2.95 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1449012, upper bound: 0.1348603
time: 3.06 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0518425, 0.0471908, -0.0914288, 0.0945719
1: -0.0342613, 0.0321695, -0.0393951, 0.0345008, -0.0687621, 0.0715646
2: -0.0187495, 0.1094280, -0.0236673, 0.1229203, -0.1416698, 0.1330953
3: -0.0559870, 0.0415918, -0.0613239, 0.0452183, -0.1012053, 0.1029158
4: -0.0628106, 0.0164810, -0.0724070, 0.0173790, -0.0801896, 0.0888881
5: -0.0221853, 0.0285809, -0.0264261, 0.0299519, -0.0521372, 0.0550070
6: -0.0180209, 0.0597589, -0.0198468, 0.0645211, -0.0825420, 0.0796057
7: -0.0519036, 0.0298092, -0.0563577, 0.0348339, -0.0867375, 0.0861669
8: 0.8305457, 0.9909523, 0.8165002, 0.9911419, -0.1605963, 0.1744521
9: -0.0258906, 0.1254902, -0.0283954, 0.1411783, -0.1670689, 0.1538855

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469324, upper bound: 0.1448907
time: 1.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469426, upper bound: 0.1446921
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0540257, 0.0484446, -0.0972664, 0.0994339
1: -0.0373475, 0.0335700, -0.0408645, 0.0351376, -0.0724850, 0.0744345
2: -0.0216974, 0.1175703, -0.0250656, 0.1267675, -0.1484649, 0.1426359
3: -0.0591903, 0.0437662, -0.0628321, 0.0462464, -0.1054367, 0.1065983
4: -0.0685832, 0.0170204, -0.0751459, 0.0176258, -0.0862091, 0.0921663
5: -0.0247323, 0.0294066, -0.0276357, 0.0303247, -0.0550570, 0.0570422
6: -0.0190791, 0.0626204, -0.0203846, 0.0658701, -0.0849492, 0.0830050
7: -0.0545923, 0.0327939, -0.0576240, 0.0363257, -0.0909180, 0.0904180
8: 0.8220751, 0.9910482, 0.8125090, 0.9912007, -0.1691256, 0.1785392
9: -0.0273315, 0.1349577, -0.0291410, 0.1456623, -0.1729938, 0.1640987

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 183

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471826, upper bound: 0.1449286
time: 2.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471826, upper bound: 0.1450693
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0442380, 0.0427294, -0.0354902, 0.0390314, -0.0832694, 0.0782196
1: -0.0342613, 0.0321695, -0.0285906, 0.0309144, -0.0651756, 0.0607601
2: -0.0187495, 0.1094280, -0.0140191, 0.0945078, -0.1132573, 0.1234471
3: -0.0559870, 0.0415918, -0.0512520, 0.0383068, -0.0942939, 0.0928438
4: -0.0628106, 0.0164810, -0.0524775, 0.0159190, -0.0787296, 0.0689586
5: -0.0221853, 0.0285809, -0.0179006, 0.0278471, -0.0500324, 0.0464815
6: -0.0180209, 0.0597589, -0.0173285, 0.0551095, -0.0731304, 0.0770873
7: -0.0519036, 0.0298092, -0.0471741, 0.0247922, -0.0766958, 0.0769833
8: 0.8305457, 0.9909523, 0.8454440, 0.9912450, -0.1606994, 0.1455083
9: -0.0258906, 0.1254902, -0.0237918, 0.1080636, -0.1339542, 0.1492819

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471569, upper bound: 0.1450489
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471700, upper bound: 0.1448334
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0488218, 0.0454082, -0.0376675, 0.0402609, -0.0890827, 0.0830757
1: -0.0373475, 0.0335700, -0.0300462, 0.0315486, -0.0688960, 0.0636162
2: -0.0216974, 0.1175703, -0.0153962, 0.0983359, -0.1200333, 0.1329665
3: -0.0591903, 0.0437662, -0.0527286, 0.0393295, -0.0985198, 0.0964949
4: -0.0685832, 0.0170204, -0.0551888, 0.0161649, -0.0847481, 0.0722093
5: -0.0247323, 0.0294066, -0.0190961, 0.0282199, -0.0529521, 0.0485027
6: -0.0190791, 0.0626204, -0.0177170, 0.0564468, -0.0755259, 0.0803374
7: -0.0545923, 0.0327939, -0.0484326, 0.0261334, -0.0807257, 0.0812265
8: 0.8220751, 0.9910482, 0.8414712, 0.9913020, -0.1692269, 0.1495770
9: -0.0273315, 0.1349577, -0.0244890, 0.1125363, -0.1398678, 0.1594467

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1474084, upper bound: 0.1450914
time: 1.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1474084, upper bound: 0.1452355
time: 1.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0292268, 0.0358424, -0.0518425, 0.0471908, -0.0764176, 0.0876849
1: -0.0250228, 0.0289604, -0.0393951, 0.0345008, -0.0595236, 0.0683555
2: -0.0106620, 0.0832981, -0.0236673, 0.1229203, -0.1335823, 0.1069653
3: -0.0472686, 0.0353324, -0.0613239, 0.0452183, -0.0924869, 0.0966564
4: -0.0458468, 0.0151690, -0.0724070, 0.0173790, -0.0632257, 0.0875760
5: -0.0147463, 0.0266969, -0.0264261, 0.0299519, -0.0446983, 0.0531229
6: -0.0162500, 0.0514486, -0.0198468, 0.0645211, -0.0807711, 0.0712954
7: -0.0435328, 0.0212903, -0.0563577, 0.0348339, -0.0783667, 0.0776479
8: 0.8569218, 0.9910862, 0.8165002, 0.9911419, -0.1342201, 0.1745860
9: -0.0222331, 0.0952172, -0.0283954, 0.1411783, -0.1634114, 0.1236125

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 140

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472733, upper bound: 0.1453946
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472779, upper bound: 0.1451652
time: 1.52 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.39 + 598.04 = 602.43 seconds
