## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0055854


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044784, -0.0038492, -0.0044784, -0.0038492, -0.0005995, 0.0005995)
1: (0.0010607, 0.0045450, 0.0010607, 0.0045450, -0.0033195, 0.0033195)
2: (0.0048122, 0.0125964, 0.0048122, 0.0125964, -0.0074161, 0.0074161)
3: (0.0020262, 0.0053065, 0.0020262, 0.0053065, -0.0031252, 0.0031252)
4: (1.0046113, 1.0173376, 1.0046113, 1.0173376, -0.0121244, 0.0121244)
5: (0.0031385, 0.0056142, 0.0031385, 0.0056142, -0.0023587, 0.0023587)
6: (-0.0130490, -0.0098272, -0.0130490, -0.0098272, -0.0030695, 0.0030695)
7: (-0.0104679, -0.0100569, -0.0104679, -0.0100569, -0.0003915, 0.0003915)
8: (-0.0040643, -0.0018383, -0.0040643, -0.0018383, -0.0021208, 0.0021208)
9: (-0.0089681, 0.0021759, -0.0089681, 0.0021759, -0.0106170, 0.0106170)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.87 + 2.18 = 4.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0062060, upper bound: 0.0062060

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0059414
time: 1.37 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0059416, upper bound: 0.0059416
time: 1.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.81 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.81
Output dim: 4, lower bound: -0.0057839, upper bound: 0.0059414
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.81
Output dim: 4, lower bound: -0.0059416, upper bound: 0.0059416

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0044778, -0.0038874, -0.0044783, -0.0038595, -0.0005881, 0.0005632
1: 0.0012727, 0.0045417, 0.0011183, 0.0045441, -0.0031185, 0.0032565
2: 0.0048194, 0.0121228, 0.0048141, 0.0124678, -0.0072754, 0.0069670
3: 0.0022258, 0.0053034, 0.0020804, 0.0053057, -0.0029359, 0.0030659
4: 1.0053854, 1.0173256, 1.0048213, 1.0173343, -0.0113902, 0.0118944
5: 0.0032891, 0.0056119, 0.0031793, 0.0056136, -0.0022159, 0.0023139
6: -0.0130460, -0.0100232, -0.0130483, -0.0098804, -0.0030112, 0.0028836
7: -0.0104675, -0.0100819, -0.0104678, -0.0100637, -0.0003841, 0.0003678
8: -0.0039289, -0.0018403, -0.0040275, -0.0018388, -0.0019923, 0.0020805
9: -0.0089578, 0.0014980, -0.0089654, 0.0019919, -0.0104156, 0.0099742

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0053622, upper bound: 0.0057142
time: 1.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055861, upper bound: 0.0057496
time: 1.17 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0044937, -0.0038744, -0.0044783, -0.0038562, -0.0006034, 0.0005705
1: 0.0012005, 0.0046294, 0.0010995, 0.0045443, -0.0031591, 0.0033410
2: 0.0046236, 0.0122842, 0.0048137, 0.0125098, -0.0074641, 0.0070577
3: 0.0021578, 0.0053860, 0.0020627, 0.0053058, -0.0029742, 0.0031454
4: 1.0051216, 1.0176458, 1.0047528, 1.0173348, -0.0115386, 0.0122029
5: 0.0032378, 0.0056742, 0.0031660, 0.0056137, -0.0022447, 0.0023739
6: -0.0131271, -0.0099564, -0.0130484, -0.0098630, -0.0030893, 0.0029212
7: -0.0104778, -0.0100734, -0.0104678, -0.0100615, -0.0003941, 0.0003726
8: -0.0039750, -0.0017843, -0.0040395, -0.0018387, -0.0020183, 0.0021345
9: -0.0092381, 0.0017290, -0.0089659, 0.0020520, -0.0106858, 0.0101040

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0055071, upper bound: 0.0057143
time: 1.24 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0057497, upper bound: 0.0057497
time: 1.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.95 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 4, lower bound: -0.0053622, upper bound: 0.0057142
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 4, lower bound: -0.0055861, upper bound: 0.0057496
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 4, lower bound: -0.0055071, upper bound: 0.0057143
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.95
Output dim: 4, lower bound: -0.0057497, upper bound: 0.0057497

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0044631, -0.0038914, -0.0044336, -0.0038650, -0.0005148, 0.0005086
1: 0.0012945, 0.0044601, 0.0011485, 0.0042966, -0.0028159, 0.0028505
2: 0.0050019, 0.0120741, 0.0053671, 0.0124003, -0.0063683, 0.0062911
3: 0.0022463, 0.0052265, 0.0021088, 0.0050726, -0.0026511, 0.0026836
4: 1.0054650, 1.0170273, 1.0049318, 1.0164301, -0.0102851, 0.0104114
5: 0.0033046, 0.0055539, 0.0032008, 0.0054377, -0.0020009, 0.0020254
6: -0.0129705, -0.0100434, -0.0128194, -0.0099084, -0.0026358, 0.0026038
7: -0.0104579, -0.0100845, -0.0104386, -0.0100673, -0.0003362, 0.0003321
8: -0.0039149, -0.0018925, -0.0040082, -0.0019970, -0.0017990, 0.0018211
9: -0.0086965, 0.0014283, -0.0081736, 0.0018952, -0.0091170, 0.0090064

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0054771
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055251
time: 1.22 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0044770, -0.0038878, -0.0044675, -0.0038638, -0.0005737, 0.0005421
1: 0.0012745, 0.0045370, 0.0011418, 0.0044846, -0.0030015, 0.0031766
2: 0.0048300, 0.0121188, 0.0049470, 0.0124153, -0.0070969, 0.0067056
3: 0.0022275, 0.0052990, 0.0021025, 0.0052497, -0.0028258, 0.0029906
4: 1.0053920, 1.0173082, 1.0049074, 1.0171170, -0.0109628, 0.0116025
5: 0.0032904, 0.0056085, 0.0031961, 0.0055713, -0.0021327, 0.0022572
6: -0.0130417, -0.0100249, -0.0129932, -0.0099022, -0.0029374, 0.0027754
7: -0.0104669, -0.0100821, -0.0104608, -0.0100665, -0.0003747, 0.0003540
8: -0.0039277, -0.0018434, -0.0040125, -0.0018768, -0.0019176, 0.0020295
9: -0.0089426, 0.0014922, -0.0087751, 0.0019167, -0.0101601, 0.0095999

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0054246, upper bound: 0.0055078
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0054246, upper bound: 0.0055598
time: 1.24 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0044785, -0.0038781, -0.0044336, -0.0038617, -0.0005375, 0.0005166
1: 0.0012208, 0.0045453, 0.0011302, 0.0042967, -0.0028601, 0.0029759
2: 0.0048114, 0.0122387, 0.0053667, 0.0124411, -0.0066484, 0.0063899
3: 0.0021769, 0.0053068, 0.0020916, 0.0050728, -0.0026927, 0.0028016
4: 1.0051960, 1.0173386, 1.0048651, 1.0164309, -0.0104467, 0.0108693
5: 0.0032522, 0.0056145, 0.0031878, 0.0054378, -0.0020323, 0.0021145
6: -0.0130493, -0.0099753, -0.0128195, -0.0098915, -0.0027517, 0.0026447
7: -0.0104679, -0.0100758, -0.0104386, -0.0100651, -0.0003510, 0.0003374
8: -0.0039620, -0.0018381, -0.0040199, -0.0019969, -0.0018273, 0.0019012
9: -0.0089692, 0.0016638, -0.0081742, 0.0019537, -0.0095180, 0.0091479

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0052773, upper bound: 0.0055253
time: 1.27 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0053177, upper bound: 0.0055253
time: 1.29 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0044929, -0.0038747, -0.0044676, -0.0038603, -0.0005913, 0.0005495
1: 0.0012022, 0.0046249, 0.0011225, 0.0044848, -0.0030425, 0.0032741
2: 0.0046335, 0.0122802, 0.0049467, 0.0124584, -0.0073147, 0.0067972
3: 0.0021594, 0.0053818, 0.0020844, 0.0052498, -0.0028644, 0.0030824
4: 1.0051280, 1.0176295, 1.0048368, 1.0171176, -0.0111126, 0.0119586
5: 0.0032390, 0.0056710, 0.0031823, 0.0055715, -0.0021618, 0.0023264
6: -0.0131230, -0.0099581, -0.0129934, -0.0098843, -0.0030275, 0.0028133
7: -0.0104773, -0.0100736, -0.0104608, -0.0100642, -0.0003862, 0.0003589
8: -0.0039739, -0.0017872, -0.0040248, -0.0018767, -0.0019438, 0.0020917
9: -0.0092238, 0.0017233, -0.0087756, 0.0019784, -0.0104718, 0.0097310

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055599, upper bound: 0.0055078
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0055601, upper bound: 0.0055601
time: 1.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.15 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0054771
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0051933, upper bound: 0.0055251
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0054246, upper bound: 0.0055078
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0054246, upper bound: 0.0055598
NS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0052773, upper bound: 0.0055253
NS_A2_B1_B2, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0053177, upper bound: 0.0055253
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0055599, upper bound: 0.0055078
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.15
Output dim: 4, lower bound: -0.0055601, upper bound: 0.0055601

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.06 + 27.10 = 31.16 seconds
