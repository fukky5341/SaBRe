## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000711535


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040660, -0.0004446, -0.0040660, -0.0004446, -0.0031298, 0.0031298)
1: (-0.0040850, -0.0030640, -0.0040850, -0.0030640, -0.0008824, 0.0008824)
2: (0.0084198, 0.0159529, 0.0084198, 0.0159529, -0.0065105, 0.0065105)
3: (0.0027415, 0.0037384, 0.0027415, 0.0037384, -0.0008616, 0.0008616)
4: (-0.0058303, -0.0002006, -0.0058303, -0.0002006, -0.0048656, 0.0048656)
5: (0.9938864, 0.9954505, 0.9938864, 0.9954505, -0.0013518, 0.0013518)
6: (0.0023343, 0.0037541, 0.0023343, 0.0037541, -0.0012270, 0.0012270)
7: (-0.0146701, -0.0093719, -0.0146701, -0.0093719, -0.0045790, 0.0045790)
8: (-0.0018987, 0.0022249, -0.0018987, 0.0022249, -0.0035639, 0.0035639)
9: (-0.0042017, -0.0038459, -0.0042017, -0.0038459, -0.0003075, 0.0003075)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 2.78 = 4.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0008367, upper bound: 0.0008367

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 220
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 220

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0007855
time: 2.18 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0008098
time: 1.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.22
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0007855
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.22
Output dim: 5, lower bound: -0.0008107, upper bound: 0.0008098

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0039324, -0.0004647, -0.0040499, -0.0004476, -0.0030757, 0.0030513
1: -0.0040473, -0.0030697, -0.0040805, -0.0030649, -0.0008672, 0.0008603
2: 0.0086977, 0.0159112, 0.0084533, 0.0159468, -0.0063981, 0.0063473
3: 0.0027783, 0.0037329, 0.0027460, 0.0037376, -0.0008467, 0.0008400
4: -0.0057991, -0.0004083, -0.0058258, -0.0002256, -0.0047436, 0.0047816
5: 0.9938951, 0.9953928, 0.9938877, 0.9954436, -0.0013179, 0.0013285
6: 0.0023422, 0.0037017, 0.0023355, 0.0037478, -0.0011963, 0.0012058
7: -0.0146408, -0.0095674, -0.0146658, -0.0093955, -0.0044643, 0.0045000
8: -0.0017466, 0.0022021, -0.0018804, 0.0022216, -0.0035024, 0.0034745
9: -0.0041997, -0.0038591, -0.0042014, -0.0038475, -0.0002998, 0.0003022

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007925, upper bound: 0.0007590
time: 1.90 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007925, upper bound: 0.0007674
time: 1.81 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0039271, -0.0002519, -0.0040139, -0.0004537, -0.0030525, 0.0031318
1: -0.0040459, -0.0030097, -0.0040703, -0.0030666, -0.0008606, 0.0008830
2: 0.0087086, 0.0163538, 0.0085281, 0.0159340, -0.0063498, 0.0065148
3: 0.0027797, 0.0037915, 0.0027559, 0.0037359, -0.0008403, 0.0008621
4: -0.0061299, -0.0004164, -0.0058162, -0.0002815, -0.0048687, 0.0047454
5: 0.9938031, 0.9953905, 0.9938903, 0.9954280, -0.0013527, 0.0013184
6: 0.0022588, 0.0036997, 0.0023379, 0.0037337, -0.0012278, 0.0011967
7: -0.0149521, -0.0095750, -0.0146568, -0.0094481, -0.0045820, 0.0044660
8: -0.0017406, 0.0024444, -0.0018394, 0.0022146, -0.0034759, 0.0035662
9: -0.0042206, -0.0038596, -0.0042008, -0.0038510, -0.0003077, 0.0002999

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007930, upper bound: 0.0007815
time: 1.80 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007925, upper bound: 0.0007925
time: 1.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.17 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 5, lower bound: -0.0007925, upper bound: 0.0007590
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 5, lower bound: -0.0007925, upper bound: 0.0007674
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 5, lower bound: -0.0007930, upper bound: 0.0007815
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.17
Output dim: 5, lower bound: -0.0007925, upper bound: 0.0007925

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0039303, -0.0005018, -0.0040425, -0.0005753, -0.0029409, 0.0030024
1: -0.0040468, -0.0030801, -0.0040784, -0.0031009, -0.0008292, 0.0008465
2: 0.0087020, 0.0158340, 0.0084686, 0.0156811, -0.0061177, 0.0062455
3: 0.0027789, 0.0037227, 0.0027480, 0.0037024, -0.0008096, 0.0008265
4: -0.0057415, -0.0004115, -0.0056272, -0.0002370, -0.0046675, 0.0045720
5: 0.9939110, 0.9953920, 0.9939429, 0.9954403, -0.0012968, 0.0012702
6: 0.0023567, 0.0037009, 0.0023856, 0.0037449, -0.0011771, 0.0011530
7: -0.0145865, -0.0095704, -0.0144790, -0.0094062, -0.0043927, 0.0043028
8: -0.0017442, 0.0021598, -0.0018720, 0.0020762, -0.0033488, 0.0034188
9: -0.0041961, -0.0038593, -0.0041889, -0.0038482, -0.0002950, 0.0002889

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007756, upper bound: 0.0007586
time: 1.86 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007756, upper bound: 0.0007586
time: 2.01 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0039309, -0.0004984, -0.0040944, -0.0005438, -0.0029713, 0.0031094
1: -0.0040469, -0.0030792, -0.0040930, -0.0030920, -0.0008377, 0.0008767
2: 0.0087008, 0.0158410, 0.0083607, 0.0157467, -0.0061810, 0.0064682
3: 0.0027787, 0.0037236, 0.0027337, 0.0037111, -0.0008180, 0.0008560
4: -0.0057467, -0.0004106, -0.0056762, -0.0001564, -0.0048339, 0.0046193
5: 0.9939097, 0.9953922, 0.9939292, 0.9954627, -0.0013430, 0.0012834
6: 0.0023554, 0.0037011, 0.0023732, 0.0037652, -0.0012190, 0.0011649
7: -0.0145915, -0.0095696, -0.0145251, -0.0093303, -0.0045493, 0.0043473
8: -0.0017449, 0.0021637, -0.0019311, 0.0021120, -0.0033835, 0.0035407
9: -0.0041964, -0.0038592, -0.0041920, -0.0038431, -0.0003055, 0.0002919

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007761, upper bound: 0.0007674
time: 1.87 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007761, upper bound: 0.0007674
time: 2.24 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0039253, -0.0002882, -0.0040066, -0.0005813, -0.0029200, 0.0030840
1: -0.0040453, -0.0030199, -0.0040683, -0.0031026, -0.0008233, 0.0008695
2: 0.0087124, 0.0162783, 0.0085433, 0.0156686, -0.0060743, 0.0064154
3: 0.0027802, 0.0037815, 0.0027579, 0.0037008, -0.0008038, 0.0008490
4: -0.0060735, -0.0004193, -0.0056179, -0.0002929, -0.0047945, 0.0045395
5: 0.9938188, 0.9953898, 0.9939454, 0.9954248, -0.0013321, 0.0012612
6: 0.0022730, 0.0036989, 0.0023879, 0.0037308, -0.0012091, 0.0011448
7: -0.0148990, -0.0095777, -0.0144702, -0.0094588, -0.0045121, 0.0042722
8: -0.0017385, 0.0024031, -0.0018311, 0.0020693, -0.0033251, 0.0035118
9: -0.0042171, -0.0038597, -0.0041883, -0.0038518, -0.0003030, 0.0002869

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007393
time: 1.68 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007490, upper bound: 0.0007372
time: 2.05 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0039257, -0.0002860, -0.0040588, -0.0005508, -0.0029467, 0.0031848
1: -0.0040455, -0.0030193, -0.0040830, -0.0030939, -0.0008308, 0.0008979
2: 0.0087115, 0.0162829, 0.0084348, 0.0157321, -0.0061297, 0.0066251
3: 0.0027801, 0.0037821, 0.0027435, 0.0037092, -0.0008112, 0.0008767
4: -0.0060769, -0.0004186, -0.0056654, -0.0002118, -0.0049512, 0.0045810
5: 0.9938179, 0.9953899, 0.9939322, 0.9954474, -0.0013756, 0.0012727
6: 0.0022722, 0.0036991, 0.0023759, 0.0037513, -0.0012486, 0.0011553
7: -0.0149022, -0.0095771, -0.0145149, -0.0093824, -0.0046596, 0.0043112
8: -0.0017390, 0.0024056, -0.0018905, 0.0021041, -0.0033554, 0.0036266
9: -0.0042173, -0.0038597, -0.0041913, -0.0038466, -0.0003129, 0.0002895

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007504
time: 2.08 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007482
time: 1.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.81 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007756, upper bound: 0.0007586
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007756, upper bound: 0.0007586
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007761, upper bound: 0.0007674
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007761, upper bound: 0.0007674
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007393
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007490, upper bound: 0.0007372
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007414, upper bound: 0.0007504
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.81
Output dim: 5, lower bound: -0.0007486, upper bound: 0.0007482

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0039303, -0.0005018, -0.0039249, -0.0005923, -0.0028334, 0.0029166
1: -0.0040468, -0.0030801, -0.0040452, -0.0031056, -0.0007989, 0.0008223
2: 0.0087020, 0.0158340, 0.0087132, 0.0156458, -0.0058941, 0.0060671
3: 0.0027789, 0.0037227, 0.0027804, 0.0036978, -0.0007800, 0.0008029
4: -0.0057415, -0.0004115, -0.0056008, -0.0004198, -0.0045342, 0.0044049
5: 0.9939110, 0.9953920, 0.9939501, 0.9953895, -0.0012597, 0.0012238
6: 0.0023567, 0.0037009, 0.0023922, 0.0036988, -0.0011434, 0.0011109
7: -0.0145865, -0.0095704, -0.0144541, -0.0095783, -0.0042672, 0.0041455
8: -0.0017442, 0.0021598, -0.0017381, 0.0020568, -0.0032265, 0.0033211
9: -0.0041961, -0.0038593, -0.0041872, -0.0038598, -0.0002865, 0.0002784

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007216
time: 1.86 seconds

## Relational analysis of NS_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007389, upper bound: 0.0007207
time: 2.12 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0039303, -0.0005018, -0.0039205, -0.0003773, -0.0029662, 0.0028610
1: -0.0040468, -0.0030801, -0.0040440, -0.0030450, -0.0008363, 0.0008066
2: 0.0087020, 0.0158340, 0.0087224, 0.0160931, -0.0061703, 0.0059514
3: 0.0027789, 0.0037227, 0.0027816, 0.0037570, -0.0008165, 0.0007876
4: -0.0057415, -0.0004115, -0.0059351, -0.0004267, -0.0044477, 0.0046113
5: 0.9939110, 0.9953920, 0.9938573, 0.9953877, -0.0012357, 0.0012812
6: 0.0023567, 0.0037009, 0.0023079, 0.0036971, -0.0011216, 0.0011629
7: -0.0145865, -0.0095704, -0.0147687, -0.0095847, -0.0041858, 0.0043397
8: -0.0017442, 0.0021598, -0.0017331, 0.0023016, -0.0033776, 0.0032578
9: -0.0041961, -0.0038593, -0.0042083, -0.0038602, -0.0002811, 0.0002914

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007394, upper bound: 0.0007111
time: 2.13 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007393, upper bound: 0.0007203
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0039309, -0.0004984, -0.0039779, -0.0005621, -0.0028603, 0.0030242
1: -0.0040469, -0.0030792, -0.0040602, -0.0030971, -0.0008064, 0.0008526
2: 0.0087008, 0.0158410, 0.0086030, 0.0157085, -0.0059499, 0.0062910
3: 0.0027787, 0.0037236, 0.0027658, 0.0037061, -0.0007874, 0.0008325
4: -0.0057467, -0.0004106, -0.0056477, -0.0003375, -0.0047015, 0.0044466
5: 0.9939097, 0.9953922, 0.9939372, 0.9954125, -0.0013062, 0.0012354
6: 0.0023554, 0.0037011, 0.0023804, 0.0037196, -0.0011857, 0.0011214
7: -0.0145915, -0.0095696, -0.0144982, -0.0095008, -0.0044247, 0.0041847
8: -0.0017449, 0.0021637, -0.0017984, 0.0020911, -0.0032570, 0.0034437
9: -0.0041964, -0.0038592, -0.0041901, -0.0038546, -0.0002971, 0.0002810

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007390, upper bound: 0.0007213
time: 1.59 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007389, upper bound: 0.0007301
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0039309, -0.0004984, -0.0039713, -0.0003508, -0.0028613, 0.0029551
1: -0.0040469, -0.0030792, -0.0040583, -0.0030376, -0.0008067, 0.0008332
2: 0.0087008, 0.0158410, 0.0086167, 0.0161481, -0.0059521, 0.0061472
3: 0.0027787, 0.0037236, 0.0027676, 0.0037642, -0.0007877, 0.0008135
4: -0.0057467, -0.0004106, -0.0059762, -0.0003477, -0.0045940, 0.0044483
5: 0.9939097, 0.9953922, 0.9938459, 0.9954097, -0.0012764, 0.0012359
6: 0.0023554, 0.0037011, 0.0022975, 0.0037170, -0.0011585, 0.0011218
7: -0.0145915, -0.0095696, -0.0148074, -0.0095104, -0.0043235, 0.0041863
8: -0.0017449, 0.0021637, -0.0017909, 0.0023318, -0.0032582, 0.0033650
9: -0.0041964, -0.0038592, -0.0042109, -0.0038552, -0.0002903, 0.0002811

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007390, upper bound: 0.0007213
time: 2.13 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007389, upper bound: 0.0007301
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0039253, -0.0002882, -0.0039255, -0.0005926, -0.0029115, 0.0030094
1: -0.0040453, -0.0030199, -0.0040454, -0.0031057, -0.0008209, 0.0008485
2: 0.0087124, 0.0162783, 0.0087121, 0.0156451, -0.0060565, 0.0062601
3: 0.0027802, 0.0037815, 0.0027802, 0.0036977, -0.0008015, 0.0008284
4: -0.0060735, -0.0004193, -0.0056003, -0.0004190, -0.0046784, 0.0045263
5: 0.9938188, 0.9953898, 0.9939502, 0.9953898, -0.0012998, 0.0012575
6: 0.0022730, 0.0036989, 0.0023923, 0.0036990, -0.0011798, 0.0011415
7: -0.0148990, -0.0095777, -0.0144537, -0.0095774, -0.0044029, 0.0042597
8: -0.0017385, 0.0024031, -0.0017387, 0.0020565, -0.0033153, 0.0034268
9: -0.0042171, -0.0038597, -0.0041872, -0.0038597, -0.0002956, 0.0002860

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007171
time: 1.64 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007204
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038751, -0.0002993, -0.0038405, -0.0003541, -0.0030112, 0.0028778
1: -0.0040312, -0.0030230, -0.0040215, -0.0030385, -0.0008490, 0.0008114
2: 0.0088168, 0.0162552, 0.0088887, 0.0161412, -0.0062640, 0.0059864
3: 0.0027941, 0.0037784, 0.0028036, 0.0037633, -0.0008289, 0.0007922
4: -0.0060563, -0.0004972, -0.0059711, -0.0005510, -0.0044738, 0.0046813
5: 0.9938236, 0.9953681, 0.9938473, 0.9953531, -0.0012430, 0.0013006
6: 0.0022774, 0.0036793, 0.0022988, 0.0036657, -0.0011282, 0.0011806
7: -0.0148828, -0.0096511, -0.0148026, -0.0097017, -0.0042104, 0.0044056
8: -0.0016814, 0.0023904, -0.0016420, 0.0023280, -0.0034289, 0.0032769
9: -0.0042160, -0.0038647, -0.0042106, -0.0038681, -0.0002827, 0.0002958

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007147
time: 2.00 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007186
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0039257, -0.0002860, -0.0039789, -0.0005615, -0.0029386, 0.0031104
1: -0.0040455, -0.0030193, -0.0040605, -0.0030970, -0.0008285, 0.0008770
2: 0.0087115, 0.0162829, 0.0086010, 0.0157099, -0.0061128, 0.0064704
3: 0.0027801, 0.0037821, 0.0027655, 0.0037063, -0.0008089, 0.0008562
4: -0.0060769, -0.0004186, -0.0056487, -0.0003360, -0.0048355, 0.0045683
5: 0.9938179, 0.9953899, 0.9939368, 0.9954129, -0.0013435, 0.0012692
6: 0.0022722, 0.0036991, 0.0023801, 0.0037199, -0.0012195, 0.0011521
7: -0.0149022, -0.0095771, -0.0144992, -0.0094993, -0.0045508, 0.0042993
8: -0.0017390, 0.0024056, -0.0017995, 0.0020919, -0.0033462, 0.0035419
9: -0.0042173, -0.0038597, -0.0041902, -0.0038545, -0.0003056, 0.0002887

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007282
time: 1.92 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007305
time: 1.89 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038756, -0.0002969, -0.0038885, -0.0003236, -0.0030398, 0.0029694
1: -0.0040313, -0.0030224, -0.0040350, -0.0030299, -0.0008570, 0.0008372
2: 0.0088158, 0.0162603, 0.0087889, 0.0162047, -0.0063235, 0.0061769
3: 0.0027939, 0.0037791, 0.0027904, 0.0037717, -0.0008368, 0.0008174
4: -0.0060601, -0.0004965, -0.0060185, -0.0004764, -0.0046162, 0.0047258
5: 0.9938225, 0.9953682, 0.9938341, 0.9953739, -0.0012825, 0.0013130
6: 0.0022764, 0.0036794, 0.0022869, 0.0036845, -0.0011641, 0.0011918
7: -0.0148864, -0.0096504, -0.0148473, -0.0096315, -0.0043444, 0.0044475
8: -0.0016819, 0.0023932, -0.0016967, 0.0023628, -0.0034615, 0.0033812
9: -0.0042162, -0.0038646, -0.0042136, -0.0038634, -0.0002917, 0.0002986

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007262
time: 1.33 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007284
time: 1.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.27 seconds
NS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007216
NS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007389, upper bound: 0.0007207
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007394, upper bound: 0.0007111
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007393, upper bound: 0.0007203
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007390, upper bound: 0.0007213
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007389, upper bound: 0.0007301
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007390, upper bound: 0.0007213
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007389, upper bound: 0.0007301
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007171
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007204
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007147
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007186
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007282
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007305
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007262
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007284

## BFS NS instance: NS_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0039303, -0.0005018, -0.0038446, -0.0006034, -0.0028249, 0.0028439
1: -0.0040468, -0.0030801, -0.0040226, -0.0031088, -0.0007965, 0.0008018
2: 0.0087020, 0.0158340, 0.0088802, 0.0156227, -0.0058765, 0.0059158
3: 0.0027789, 0.0037227, 0.0028025, 0.0036947, -0.0007777, 0.0007829
4: -0.0057415, -0.0004115, -0.0055836, -0.0005447, -0.0044211, 0.0043917
5: 0.9939110, 0.9953920, 0.9939550, 0.9953550, -0.0012283, 0.0012201
6: 0.0023567, 0.0037009, 0.0023966, 0.0036673, -0.0011149, 0.0011075
7: -0.0145865, -0.0095704, -0.0144379, -0.0096957, -0.0041607, 0.0041331
8: -0.0017442, 0.0021598, -0.0016467, 0.0020442, -0.0032168, 0.0032383
9: -0.0041961, -0.0038593, -0.0041861, -0.0038677, -0.0002794, 0.0002775

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007200
time: 2.24 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007316
time: 2.24 seconds

## BFS NS instance: NS_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038770, -0.0005128, -0.0037616, -0.0003627, -0.0028926, 0.0028181
1: -0.0040317, -0.0030832, -0.0039992, -0.0030409, -0.0008155, 0.0007945
2: 0.0088130, 0.0158112, 0.0090529, 0.0161235, -0.0060172, 0.0058623
3: 0.0027936, 0.0037197, 0.0028253, 0.0037610, -0.0007963, 0.0007758
4: -0.0057244, -0.0004944, -0.0059578, -0.0006737, -0.0043811, 0.0044969
5: 0.9939158, 0.9953689, 0.9938510, 0.9953191, -0.0012172, 0.0012494
6: 0.0023611, 0.0036800, 0.0023022, 0.0036348, -0.0011049, 0.0011340
7: -0.0145705, -0.0096484, -0.0147901, -0.0098172, -0.0041231, 0.0042321
8: -0.0016835, 0.0021473, -0.0015522, 0.0023183, -0.0032938, 0.0032090
9: -0.0041950, -0.0038645, -0.0042097, -0.0038758, -0.0002769, 0.0002842

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007191, upper bound: 0.0007122
time: 2.19 seconds

## Relational analysis of NS_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007118
time: 2.03 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038501, -0.0005127, -0.0039205, -0.0003773, -0.0028937, 0.0028527
1: -0.0040242, -0.0030832, -0.0040440, -0.0030450, -0.0008158, 0.0008043
2: 0.0088688, 0.0158114, 0.0087224, 0.0160931, -0.0060194, 0.0059342
3: 0.0028009, 0.0037197, 0.0027816, 0.0037570, -0.0007966, 0.0007853
4: -0.0057246, -0.0005361, -0.0059351, -0.0004267, -0.0044348, 0.0044985
5: 0.9939158, 0.9953573, 0.9938573, 0.9953877, -0.0012321, 0.0012498
6: 0.0023610, 0.0036695, 0.0023079, 0.0036971, -0.0011184, 0.0011345
7: -0.0145706, -0.0096877, -0.0147687, -0.0095847, -0.0041737, 0.0042336
8: -0.0016529, 0.0021475, -0.0017331, 0.0023016, -0.0032950, 0.0032484
9: -0.0041950, -0.0038671, -0.0042083, -0.0038602, -0.0002803, 0.0002843

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007107
time: 1.99 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007111
time: 2.13 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037658, -0.0002720, -0.0038705, -0.0003872, -0.0027465, 0.0029342
1: -0.0040004, -0.0030154, -0.0040299, -0.0030478, -0.0007743, 0.0008273
2: 0.0090441, 0.0163120, 0.0088265, 0.0160724, -0.0057132, 0.0061038
3: 0.0028241, 0.0037859, 0.0027953, 0.0037542, -0.0007561, 0.0008077
4: -0.0060987, -0.0006672, -0.0059197, -0.0005045, -0.0045616, 0.0042697
5: 0.9938118, 0.9953209, 0.9938616, 0.9953661, -0.0012674, 0.0011863
6: 0.0022667, 0.0036364, 0.0023118, 0.0036774, -0.0011504, 0.0010768
7: -0.0149227, -0.0098110, -0.0147542, -0.0096579, -0.0042930, 0.0040183
8: -0.0015569, 0.0024215, -0.0016761, 0.0022904, -0.0031274, 0.0033412
9: -0.0042186, -0.0038754, -0.0042073, -0.0038651, -0.0002883, 0.0002698

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0006975
time: 1.90 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007016
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038507, -0.0005092, -0.0039779, -0.0005621, -0.0027877, 0.0030159
1: -0.0040243, -0.0030822, -0.0040602, -0.0030971, -0.0007860, 0.0008503
2: 0.0088676, 0.0158185, 0.0086030, 0.0157085, -0.0057991, 0.0062738
3: 0.0028008, 0.0037206, 0.0027658, 0.0037061, -0.0007674, 0.0008302
4: -0.0057299, -0.0005352, -0.0056477, -0.0003375, -0.0046886, 0.0043339
5: 0.9939143, 0.9953575, 0.9939372, 0.9954125, -0.0013026, 0.0012041
6: 0.0023597, 0.0036697, 0.0023804, 0.0037196, -0.0011824, 0.0010929
7: -0.0145756, -0.0096868, -0.0144982, -0.0095008, -0.0044125, 0.0040786
8: -0.0016536, 0.0021514, -0.0017984, 0.0020911, -0.0031744, 0.0034343
9: -0.0041953, -0.0038671, -0.0041901, -0.0038546, -0.0002963, 0.0002739

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007306, upper bound: 0.0007301
time: 2.24 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007306, upper bound: 0.0007297
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037661, -0.0002687, -0.0039242, -0.0005731, -0.0027604, 0.0030827
1: -0.0040005, -0.0030144, -0.0040450, -0.0031002, -0.0007783, 0.0008691
2: 0.0090435, 0.0163190, 0.0087146, 0.0156856, -0.0057421, 0.0064126
3: 0.0028241, 0.0037868, 0.0027805, 0.0037030, -0.0007599, 0.0008486
4: -0.0061039, -0.0006667, -0.0056306, -0.0004209, -0.0047924, 0.0042913
5: 0.9938104, 0.9953210, 0.9939418, 0.9953893, -0.0013315, 0.0011923
6: 0.0022654, 0.0036365, 0.0023847, 0.0036985, -0.0012086, 0.0010822
7: -0.0149276, -0.0098106, -0.0144821, -0.0095793, -0.0045102, 0.0040386
8: -0.0015573, 0.0024253, -0.0017373, 0.0020786, -0.0031432, 0.0035103
9: -0.0042190, -0.0038754, -0.0041891, -0.0038598, -0.0003028, 0.0002712

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007216, upper bound: 0.0007188
time: 1.98 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007216
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038507, -0.0005092, -0.0039713, -0.0003508, -0.0027852, 0.0029468
1: -0.0040243, -0.0030822, -0.0040583, -0.0030376, -0.0007853, 0.0008308
2: 0.0088676, 0.0158185, 0.0086167, 0.0161481, -0.0057938, 0.0061299
3: 0.0028008, 0.0037206, 0.0027676, 0.0037642, -0.0007667, 0.0008112
4: -0.0057299, -0.0005352, -0.0059762, -0.0003477, -0.0045811, 0.0043299
5: 0.9939143, 0.9953575, 0.9938459, 0.9954097, -0.0012728, 0.0012030
6: 0.0023597, 0.0036697, 0.0022975, 0.0037170, -0.0011553, 0.0010919
7: -0.0145756, -0.0096868, -0.0148074, -0.0095104, -0.0043113, 0.0040749
8: -0.0016536, 0.0021514, -0.0017909, 0.0023318, -0.0031715, 0.0033555
9: -0.0041953, -0.0038671, -0.0042109, -0.0038552, -0.0002895, 0.0002736

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007172, upper bound: 0.0007009
time: 2.26 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007309, upper bound: 0.0007013
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037661, -0.0002687, -0.0039194, -0.0003599, -0.0027728, 0.0030220
1: -0.0040005, -0.0030144, -0.0040437, -0.0030401, -0.0007818, 0.0008520
2: 0.0090435, 0.0163190, 0.0087246, 0.0161291, -0.0057680, 0.0062864
3: 0.0028241, 0.0037868, 0.0027819, 0.0037617, -0.0007633, 0.0008319
4: -0.0061039, -0.0006667, -0.0059620, -0.0004284, -0.0046981, 0.0043106
5: 0.9938104, 0.9953210, 0.9938499, 0.9953873, -0.0013053, 0.0011976
6: 0.0022654, 0.0036365, 0.0023011, 0.0036966, -0.0011848, 0.0010871
7: -0.0149276, -0.0098106, -0.0147941, -0.0095863, -0.0044214, 0.0040568
8: -0.0015573, 0.0024253, -0.0017318, 0.0023214, -0.0031574, 0.0034412
9: -0.0042190, -0.0038754, -0.0042100, -0.0038603, -0.0002969, 0.0002724

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007071
time: 1.80 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007103
time: 1.96 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0039235, -0.0003162, -0.0039181, -0.0007041, -0.0027853, 0.0029621
1: -0.0040448, -0.0030278, -0.0040433, -0.0031372, -0.0007853, 0.0008351
2: 0.0087162, 0.0162201, 0.0087273, 0.0154132, -0.0057939, 0.0061618
3: 0.0027807, 0.0037738, 0.0027822, 0.0036670, -0.0007667, 0.0008154
4: -0.0060300, -0.0004221, -0.0054270, -0.0004304, -0.0046050, 0.0043300
5: 0.9938309, 0.9953889, 0.9939985, 0.9953867, -0.0012794, 0.0012030
6: 0.0022840, 0.0036982, 0.0024361, 0.0036961, -0.0011613, 0.0010920
7: -0.0148581, -0.0095804, -0.0142905, -0.0095882, -0.0043338, 0.0040750
8: -0.0017364, 0.0023712, -0.0017304, 0.0019295, -0.0031716, 0.0033730
9: -0.0042143, -0.0038599, -0.0041762, -0.0038604, -0.0002910, 0.0002736

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007151
time: 1.99 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007172
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0039234, -0.0003189, -0.0039485, -0.0006885, -0.0027038, 0.0030159
1: -0.0040448, -0.0030286, -0.0040519, -0.0031328, -0.0007623, 0.0008503
2: 0.0087163, 0.0162145, 0.0086641, 0.0154456, -0.0056244, 0.0062738
3: 0.0027808, 0.0037730, 0.0027738, 0.0036713, -0.0007443, 0.0008302
4: -0.0060258, -0.0004221, -0.0054512, -0.0003831, -0.0046886, 0.0042033
5: 0.9938321, 0.9953890, 0.9939917, 0.9953998, -0.0013026, 0.0011678
6: 0.0022850, 0.0036982, 0.0024299, 0.0037081, -0.0011824, 0.0010600
7: -0.0148541, -0.0095804, -0.0143134, -0.0095437, -0.0044125, 0.0039558
8: -0.0017364, 0.0023681, -0.0017650, 0.0019472, -0.0030788, 0.0034343
9: -0.0042140, -0.0038599, -0.0041777, -0.0038575, -0.0002963, 0.0002656

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007187
time: 1.98 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007212
time: 1.61 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0038733, -0.0003264, -0.0038330, -0.0004634, -0.0029087, 0.0028420
1: -0.0040307, -0.0030307, -0.0040193, -0.0030693, -0.0008201, 0.0008013
2: 0.0088205, 0.0161989, 0.0089044, 0.0159138, -0.0060508, 0.0059120
3: 0.0027945, 0.0037710, 0.0028057, 0.0037332, -0.0008007, 0.0007824
4: -0.0060142, -0.0005000, -0.0058011, -0.0005627, -0.0044183, 0.0045220
5: 0.9938353, 0.9953673, 0.9938945, 0.9953499, -0.0012275, 0.0012563
6: 0.0022880, 0.0036786, 0.0023417, 0.0036628, -0.0011142, 0.0011404
7: -0.0148432, -0.0096537, -0.0146427, -0.0097127, -0.0041581, 0.0042557
8: -0.0016794, 0.0023596, -0.0016334, 0.0022035, -0.0033122, 0.0032362
9: -0.0042133, -0.0038648, -0.0041998, -0.0038688, -0.0002792, 0.0002858

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B1_B1

### Relational analysis result of NS_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007147
time: 1.85 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2

### Relational analysis result of NS_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007147
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038733, -0.0003282, -0.0038638, -0.0004445, -0.0029241, 0.0028832
1: -0.0040307, -0.0030312, -0.0040280, -0.0030640, -0.0008244, 0.0008129
2: 0.0088207, 0.0161950, 0.0088403, 0.0159531, -0.0060828, 0.0059977
3: 0.0027946, 0.0037705, 0.0027972, 0.0037384, -0.0008050, 0.0007937
4: -0.0060113, -0.0005002, -0.0058305, -0.0005148, -0.0044823, 0.0045459
5: 0.9938361, 0.9953673, 0.9938864, 0.9953632, -0.0012453, 0.0012630
6: 0.0022887, 0.0036785, 0.0023343, 0.0036748, -0.0011304, 0.0011464
7: -0.0148404, -0.0096538, -0.0146703, -0.0096676, -0.0042183, 0.0042782
8: -0.0016793, 0.0023575, -0.0016685, 0.0022250, -0.0033297, 0.0032831
9: -0.0042131, -0.0038649, -0.0042017, -0.0038658, -0.0002833, 0.0002873

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B2_B2_B1

### Relational analysis result of NS_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007182
time: 1.77 seconds

## Relational analysis of NS_A2_B1_B2_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007182
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0039239, -0.0003138, -0.0039717, -0.0006699, -0.0027221, 0.0030635
1: -0.0040450, -0.0030271, -0.0040584, -0.0031275, -0.0007675, 0.0008637
2: 0.0087153, 0.0162252, 0.0086159, 0.0154844, -0.0056624, 0.0063727
3: 0.0027806, 0.0037744, 0.0027675, 0.0036764, -0.0007493, 0.0008433
4: -0.0060338, -0.0004214, -0.0054802, -0.0003471, -0.0047626, 0.0042318
5: 0.9938298, 0.9953892, 0.9939837, 0.9954097, -0.0013232, 0.0011757
6: 0.0022830, 0.0036984, 0.0024226, 0.0037171, -0.0012011, 0.0010672
7: -0.0148616, -0.0095797, -0.0143406, -0.0095098, -0.0044821, 0.0039826
8: -0.0017370, 0.0023740, -0.0017913, 0.0019684, -0.0030996, 0.0034884
9: -0.0042145, -0.0038599, -0.0041796, -0.0038552, -0.0003010, 0.0002674

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007274
time: 1.62 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007282
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0039239, -0.0003169, -0.0039953, -0.0006598, -0.0027362, 0.0031082
1: -0.0040449, -0.0030280, -0.0040651, -0.0031247, -0.0007714, 0.0008763
2: 0.0087154, 0.0162186, 0.0085667, 0.0155054, -0.0056919, 0.0064657
3: 0.0027806, 0.0037736, 0.0027610, 0.0036792, -0.0007532, 0.0008556
4: -0.0060289, -0.0004215, -0.0054959, -0.0003104, -0.0048321, 0.0042538
5: 0.9938313, 0.9953892, 0.9939793, 0.9954200, -0.0013425, 0.0011818
6: 0.0022843, 0.0036984, 0.0024187, 0.0037264, -0.0012186, 0.0010727
7: -0.0148570, -0.0095798, -0.0143554, -0.0094752, -0.0045475, 0.0040033
8: -0.0017369, 0.0023704, -0.0018183, 0.0019799, -0.0031158, 0.0035394
9: -0.0042142, -0.0038599, -0.0041806, -0.0038529, -0.0003054, 0.0002688

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007289
time: 1.94 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007309
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0038738, -0.0003236, -0.0038811, -0.0004307, -0.0029392, 0.0029344
1: -0.0040308, -0.0030299, -0.0040329, -0.0030601, -0.0008287, 0.0008273
2: 0.0088196, 0.0162046, 0.0088044, 0.0159820, -0.0061142, 0.0061041
3: 0.0027944, 0.0037717, 0.0027924, 0.0037423, -0.0008091, 0.0008078
4: -0.0060184, -0.0004993, -0.0058520, -0.0004880, -0.0045618, 0.0045694
5: 0.9938341, 0.9953675, 0.9938803, 0.9953706, -0.0012674, 0.0012695
6: 0.0022869, 0.0036787, 0.0023289, 0.0036816, -0.0011504, 0.0011523
7: -0.0148472, -0.0096531, -0.0146906, -0.0096424, -0.0042932, 0.0043003
8: -0.0016799, 0.0023627, -0.0016882, 0.0022408, -0.0033469, 0.0033414
9: -0.0042136, -0.0038648, -0.0042031, -0.0038641, -0.0002883, 0.0002888

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007263
time: 1.95 seconds

## Relational analysis of NS_A2_B2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007259
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038737, -0.0003260, -0.0039081, -0.0004165, -0.0029535, 0.0029685
1: -0.0040308, -0.0030306, -0.0040405, -0.0030561, -0.0008327, 0.0008369
2: 0.0088198, 0.0161998, 0.0087481, 0.0160114, -0.0061439, 0.0061751
3: 0.0027945, 0.0037711, 0.0027850, 0.0037461, -0.0008130, 0.0008172
4: -0.0060148, -0.0004995, -0.0058741, -0.0004459, -0.0046149, 0.0045915
5: 0.9938352, 0.9953674, 0.9938743, 0.9953823, -0.0012821, 0.0012757
6: 0.0022878, 0.0036787, 0.0023233, 0.0036922, -0.0011638, 0.0011579
7: -0.0148438, -0.0096532, -0.0147113, -0.0096028, -0.0043431, 0.0043212
8: -0.0016798, 0.0023601, -0.0017190, 0.0022569, -0.0033632, 0.0033802
9: -0.0042134, -0.0038648, -0.0042045, -0.0038614, -0.0002916, 0.0002902

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007284
time: 1.76 seconds

## Relational analysis of NS_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007280
time: 2.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.54 seconds
NS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007200
NS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007301, upper bound: 0.0007316
NS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007191, upper bound: 0.0007122
NS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007118
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007107
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007410, upper bound: 0.0007111
NS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0006975
NS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007016
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007306, upper bound: 0.0007301
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007306, upper bound: 0.0007297
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007216, upper bound: 0.0007188
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007212, upper bound: 0.0007216
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007172, upper bound: 0.0007009
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007309, upper bound: 0.0007013
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007071
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007284, upper bound: 0.0007103
NS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007151
NS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007172
NS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007187
NS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007212
NS_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007147
NS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007147
NS_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007182
NS_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007182
NS_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007274
NS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007214, upper bound: 0.0007282
NS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007289
NS_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007210, upper bound: 0.0007309
NS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007263
NS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007259
NS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007099, upper bound: 0.0007284
NS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.54
Output dim: 5, lower bound: -0.0007103, upper bound: 0.0007280

## BFS NS instance: NS_A1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038501, -0.0005127, -0.0038446, -0.0006034, -0.0027524, 0.0028355
1: -0.0040242, -0.0030832, -0.0040226, -0.0031088, -0.0007760, 0.0007994
2: 0.0088688, 0.0158114, 0.0088802, 0.0156227, -0.0057256, 0.0058985
3: 0.0028009, 0.0037197, 0.0028025, 0.0036947, -0.0007577, 0.0007806
4: -0.0057246, -0.0005361, -0.0055836, -0.0005447, -0.0044082, 0.0042789
5: 0.9939158, 0.9953573, 0.9939550, 0.9953550, -0.0012247, 0.0011888
6: 0.0023610, 0.0036695, 0.0023966, 0.0036673, -0.0011117, 0.0010791
7: -0.0145706, -0.0096877, -0.0144379, -0.0096957, -0.0041486, 0.0040270
8: -0.0016529, 0.0021475, -0.0016467, 0.0020442, -0.0031342, 0.0032289
9: -0.0041950, -0.0038671, -0.0041861, -0.0038677, -0.0002786, 0.0002704

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007052
time: 2.17 seconds

## Relational analysis of NS_A1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007097, upper bound: 0.0007055
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037658, -0.0002720, -0.0038446, -0.0006034, -0.0026783, 0.0029536
1: -0.0040004, -0.0030154, -0.0040226, -0.0031088, -0.0007551, 0.0008327
2: 0.0090441, 0.0163120, 0.0088802, 0.0156227, -0.0055713, 0.0061440
3: 0.0028241, 0.0037859, 0.0028025, 0.0036947, -0.0007373, 0.0008131
4: -0.0060987, -0.0006672, -0.0055836, -0.0005447, -0.0045917, 0.0041637
5: 0.9938118, 0.9953209, 0.9939550, 0.9953550, -0.0012757, 0.0011568
6: 0.0022667, 0.0036364, 0.0023966, 0.0036673, -0.0011580, 0.0010500
7: -0.0149227, -0.0098110, -0.0144379, -0.0096957, -0.0043213, 0.0039185
8: -0.0015569, 0.0024215, -0.0016467, 0.0020442, -0.0030497, 0.0033633
9: -0.0042186, -0.0038754, -0.0041861, -0.0038677, -0.0002902, 0.0002631

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007069
time: 1.89 seconds

## Relational analysis of NS_A1_B1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007097, upper bound: 0.0007115
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038697, -0.0006242, -0.0037597, -0.0003904, -0.0028571, 0.0025812
1: -0.0040297, -0.0031147, -0.0039987, -0.0030487, -0.0008055, 0.0007277
2: 0.0088280, 0.0155793, 0.0090569, 0.0160657, -0.0059433, 0.0053694
3: 0.0027955, 0.0036890, 0.0028258, 0.0037533, -0.0007865, 0.0007106
4: -0.0055511, -0.0005057, -0.0059146, -0.0006767, -0.0040128, 0.0044416
5: 0.9939640, 0.9953658, 0.9938630, 0.9953182, -0.0011149, 0.0012340
6: 0.0024048, 0.0036771, 0.0023131, 0.0036340, -0.0010120, 0.0011201
7: -0.0144074, -0.0096590, -0.0147495, -0.0098200, -0.0037765, 0.0041801
8: -0.0016752, 0.0020204, -0.0015500, 0.0022867, -0.0032534, 0.0029392
9: -0.0041840, -0.0038652, -0.0042070, -0.0038760, -0.0002536, 0.0002807

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006894, upper bound: 0.0006770
time: 2.09 seconds

## Relational analysis of NS_A1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006828, upper bound: 0.0006772
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038989, -0.0006088, -0.0037596, -0.0003907, -0.0028980, 0.0025961
1: -0.0040379, -0.0031103, -0.0039986, -0.0030488, -0.0008171, 0.0007319
2: 0.0087674, 0.0156114, 0.0090570, 0.0160651, -0.0060284, 0.0054004
3: 0.0027875, 0.0036932, 0.0028259, 0.0037533, -0.0007978, 0.0007147
4: -0.0055752, -0.0004603, -0.0059142, -0.0006768, -0.0040359, 0.0045053
5: 0.9939574, 0.9953783, 0.9938631, 0.9953182, -0.0011213, 0.0012517
6: 0.0023987, 0.0036886, 0.0023132, 0.0036340, -0.0010178, 0.0011362
7: -0.0144300, -0.0096164, -0.0147490, -0.0098201, -0.0037983, 0.0042400
8: -0.0017084, 0.0020380, -0.0015499, 0.0022863, -0.0033000, 0.0029562
9: -0.0041856, -0.0038623, -0.0042070, -0.0038760, -0.0002550, 0.0002847

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006912, upper bound: 0.0006774
time: 1.88 seconds

## Relational analysis of NS_A1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006772
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038501, -0.0005127, -0.0038450, -0.0003872, -0.0027496, 0.0027814
1: -0.0040242, -0.0030832, -0.0040227, -0.0030478, -0.0007752, 0.0007842
2: 0.0088688, 0.0158114, 0.0088795, 0.0160724, -0.0057197, 0.0057859
3: 0.0028009, 0.0037197, 0.0028024, 0.0037542, -0.0007569, 0.0007657
4: -0.0057246, -0.0005361, -0.0059196, -0.0005441, -0.0043240, 0.0042745
5: 0.9939158, 0.9953573, 0.9938616, 0.9953550, -0.0012013, 0.0011876
6: 0.0023610, 0.0036695, 0.0023118, 0.0036674, -0.0010905, 0.0010780
7: -0.0145706, -0.0096877, -0.0147542, -0.0096952, -0.0040694, 0.0040228
8: -0.0016529, 0.0021475, -0.0016471, 0.0022903, -0.0031310, 0.0031672
9: -0.0041950, -0.0038671, -0.0042073, -0.0038676, -0.0002733, 0.0002701

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0006884
time: 1.99 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0006920
time: 2.10 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038501, -0.0005127, -0.0037611, -0.0001489, -0.0030203, 0.0027108
1: -0.0040242, -0.0030832, -0.0039991, -0.0029806, -0.0008515, 0.0007643
2: 0.0088688, 0.0158114, 0.0090540, 0.0165681, -0.0062829, 0.0056391
3: 0.0028009, 0.0037197, 0.0028254, 0.0038198, -0.0008314, 0.0007462
4: -0.0057246, -0.0005361, -0.0062901, -0.0006745, -0.0042143, 0.0046955
5: 0.9939158, 0.9953573, 0.9937586, 0.9953189, -0.0011709, 0.0013045
6: 0.0023610, 0.0036695, 0.0022184, 0.0036346, -0.0010628, 0.0011841
7: -0.0145706, -0.0096877, -0.0151028, -0.0098179, -0.0039661, 0.0044190
8: -0.0016529, 0.0021475, -0.0015516, 0.0025617, -0.0034393, 0.0030868
9: -0.0041950, -0.0038671, -0.0042307, -0.0038759, -0.0002663, 0.0002967

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0006917
time: 1.98 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0006920
time: 2.04 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037639, -0.0002995, -0.0038632, -0.0004944, -0.0026417, 0.0028988
1: -0.0039999, -0.0030231, -0.0040278, -0.0030780, -0.0007448, 0.0008173
2: 0.0090481, 0.0162549, 0.0088417, 0.0158494, -0.0054952, 0.0060301
3: 0.0028247, 0.0037784, 0.0027974, 0.0037247, -0.0007272, 0.0007980
4: -0.0060560, -0.0006701, -0.0057530, -0.0005159, -0.0045065, 0.0041068
5: 0.9938236, 0.9953200, 0.9939078, 0.9953629, -0.0012520, 0.0011410
6: 0.0022774, 0.0036357, 0.0023538, 0.0036746, -0.0011365, 0.0010357
7: -0.0148825, -0.0098138, -0.0145973, -0.0096686, -0.0042411, 0.0038649
8: -0.0015548, 0.0023902, -0.0016678, 0.0021683, -0.0030081, 0.0033009
9: -0.0042160, -0.0038756, -0.0041968, -0.0038658, -0.0002848, 0.0002595

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006893, upper bound: 0.0006668
time: 1.86 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006886, upper bound: 0.0006626
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037638, -0.0003005, -0.0038934, -0.0004791, -0.0026559, 0.0029330
1: -0.0039998, -0.0030234, -0.0040364, -0.0030737, -0.0007488, 0.0008269
2: 0.0090483, 0.0162527, 0.0087787, 0.0158813, -0.0055247, 0.0061012
3: 0.0028247, 0.0037781, 0.0027890, 0.0037289, -0.0007311, 0.0008074
4: -0.0060544, -0.0006703, -0.0057768, -0.0004688, -0.0045597, 0.0041288
5: 0.9938241, 0.9953200, 0.9939013, 0.9953760, -0.0012668, 0.0011471
6: 0.0022778, 0.0036356, 0.0023478, 0.0036864, -0.0011499, 0.0010412
7: -0.0148810, -0.0098139, -0.0146197, -0.0096243, -0.0042912, 0.0038857
8: -0.0015547, 0.0023890, -0.0017022, 0.0021857, -0.0030242, 0.0033398
9: -0.0042159, -0.0038756, -0.0041983, -0.0038629, -0.0002881, 0.0002609

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006696
time: 1.96 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006886, upper bound: 0.0006661
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038507, -0.0005092, -0.0038998, -0.0005726, -0.0027797, 0.0029426
1: -0.0040243, -0.0030822, -0.0040382, -0.0031001, -0.0007837, 0.0008296
2: 0.0088676, 0.0158185, 0.0087655, 0.0156867, -0.0057823, 0.0061213
3: 0.0028008, 0.0037206, 0.0027873, 0.0037032, -0.0007652, 0.0008101
4: -0.0057299, -0.0005352, -0.0056314, -0.0004589, -0.0045747, 0.0043213
5: 0.9939143, 0.9953575, 0.9939417, 0.9953788, -0.0012710, 0.0012006
6: 0.0023597, 0.0036697, 0.0023845, 0.0036889, -0.0011537, 0.0010898
7: -0.0145756, -0.0096868, -0.0144829, -0.0096150, -0.0043053, 0.0040668
8: -0.0016536, 0.0021514, -0.0017095, 0.0020792, -0.0031652, 0.0033508
9: -0.0041953, -0.0038671, -0.0041891, -0.0038622, -0.0002891, 0.0002731

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007101
time: 1.54 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007156, upper bound: 0.0007096
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038507, -0.0005092, -0.0038121, -0.0003322, -0.0028969, 0.0028562
1: -0.0040243, -0.0030822, -0.0040134, -0.0030323, -0.0008167, 0.0008053
2: 0.0088676, 0.0158185, 0.0089478, 0.0161868, -0.0060261, 0.0059414
3: 0.0028008, 0.0037206, 0.0028114, 0.0037694, -0.0007975, 0.0007863
4: -0.0057299, -0.0005352, -0.0060051, -0.0005952, -0.0044403, 0.0045035
5: 0.9939143, 0.9953575, 0.9938378, 0.9953409, -0.0012336, 0.0012512
6: 0.0023597, 0.0036697, 0.0022903, 0.0036546, -0.0011198, 0.0011357
7: -0.0145756, -0.0096868, -0.0148346, -0.0097433, -0.0041788, 0.0042383
8: -0.0016536, 0.0021514, -0.0016097, 0.0023529, -0.0032987, 0.0032523
9: -0.0041953, -0.0038671, -0.0042127, -0.0038709, -0.0002806, 0.0002846

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007093
time: 2.08 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007156, upper bound: 0.0007093
time: 2.04 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037642, -0.0002960, -0.0039170, -0.0006796, -0.0025232, 0.0030481
1: -0.0039999, -0.0030221, -0.0040430, -0.0031303, -0.0007114, 0.0008594
2: 0.0090474, 0.0162620, 0.0087297, 0.0154642, -0.0052487, 0.0063407
3: 0.0028246, 0.0037793, 0.0027825, 0.0036737, -0.0006946, 0.0008391
4: -0.0060614, -0.0006696, -0.0054651, -0.0004321, -0.0047387, 0.0039225
5: 0.9938222, 0.9953203, 0.9939879, 0.9953862, -0.0013165, 0.0010898
6: 0.0022761, 0.0036358, 0.0024265, 0.0036957, -0.0011950, 0.0009892
7: -0.0148876, -0.0098133, -0.0143264, -0.0095898, -0.0044596, 0.0036915
8: -0.0015551, 0.0023941, -0.0017291, 0.0019574, -0.0028731, 0.0034709
9: -0.0042163, -0.0038756, -0.0041786, -0.0038606, -0.0002995, 0.0002479

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006890
time: 1.91 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006864, upper bound: 0.0006824
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037641, -0.0002973, -0.0039405, -0.0006689, -0.0025366, 0.0030788
1: -0.0039999, -0.0030225, -0.0040496, -0.0031273, -0.0007152, 0.0008680
2: 0.0090477, 0.0162594, 0.0086807, 0.0154863, -0.0052767, 0.0064045
3: 0.0028246, 0.0037790, 0.0027761, 0.0036767, -0.0006983, 0.0008475
4: -0.0060594, -0.0006698, -0.0054816, -0.0003956, -0.0047863, 0.0039435
5: 0.9938228, 0.9953201, 0.9939833, 0.9953964, -0.0013298, 0.0010956
6: 0.0022766, 0.0036358, 0.0024223, 0.0037049, -0.0012070, 0.0009945
7: -0.0148857, -0.0098135, -0.0143420, -0.0095554, -0.0045045, 0.0037113
8: -0.0015550, 0.0023927, -0.0017559, 0.0019695, -0.0028885, 0.0035058
9: -0.0042162, -0.0038756, -0.0041797, -0.0038582, -0.0003025, 0.0002492

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006864, upper bound: 0.0006913
time: 2.07 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006864, upper bound: 0.0006860
time: 2.04 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0038434, -0.0006195, -0.0039695, -0.0003774, -0.0027491, 0.0027076
1: -0.0040222, -0.0031133, -0.0040578, -0.0030451, -0.0007751, 0.0007634
2: 0.0088829, 0.0155891, 0.0086205, 0.0160929, -0.0057186, 0.0056323
3: 0.0028028, 0.0036903, 0.0027681, 0.0037569, -0.0007568, 0.0007454
4: -0.0055585, -0.0005466, -0.0059349, -0.0003505, -0.0042093, 0.0042737
5: 0.9939619, 0.9953543, 0.9938573, 0.9954088, -0.0011695, 0.0011874
6: 0.0024029, 0.0036668, 0.0023080, 0.0037163, -0.0010615, 0.0010778
7: -0.0144143, -0.0096976, -0.0147686, -0.0095131, -0.0039614, 0.0040221
8: -0.0016452, 0.0020258, -0.0017888, 0.0023015, -0.0031304, 0.0030831
9: -0.0041845, -0.0038678, -0.0042083, -0.0038554, -0.0002660, 0.0002701

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007017
time: 1.61 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007013
time: 2.17 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0038714, -0.0006066, -0.0039694, -0.0003801, -0.0027918, 0.0027217
1: -0.0040302, -0.0031097, -0.0040578, -0.0030458, -0.0007871, 0.0007673
2: 0.0088245, 0.0156160, 0.0086207, 0.0160871, -0.0058075, 0.0056617
3: 0.0027951, 0.0036938, 0.0027681, 0.0037562, -0.0007685, 0.0007492
4: -0.0055785, -0.0005030, -0.0059306, -0.0003507, -0.0042312, 0.0043402
5: 0.9939563, 0.9953665, 0.9938585, 0.9954088, -0.0011756, 0.0012058
6: 0.0023978, 0.0036778, 0.0023091, 0.0037162, -0.0010670, 0.0010945
7: -0.0144332, -0.0096566, -0.0147645, -0.0095132, -0.0039820, 0.0040846
8: -0.0016772, 0.0020405, -0.0017887, 0.0022984, -0.0031791, 0.0030992
9: -0.0041858, -0.0038650, -0.0042080, -0.0038554, -0.0002674, 0.0002743

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0007013
time: 2.05 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0007012
time: 2.36 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037642, -0.0002960, -0.0039122, -0.0004646, -0.0026691, 0.0029873
1: -0.0039999, -0.0030221, -0.0040416, -0.0030696, -0.0007525, 0.0008422
2: 0.0090474, 0.0162620, 0.0087397, 0.0159114, -0.0055524, 0.0062143
3: 0.0028246, 0.0037793, 0.0027839, 0.0037329, -0.0007348, 0.0008224
4: -0.0060614, -0.0006696, -0.0057993, -0.0004397, -0.0046442, 0.0041495
5: 0.9938222, 0.9953203, 0.9938951, 0.9953841, -0.0012903, 0.0011529
6: 0.0022761, 0.0036358, 0.0023422, 0.0036938, -0.0011712, 0.0010464
7: -0.0148876, -0.0098133, -0.0146410, -0.0095969, -0.0043707, 0.0039051
8: -0.0015551, 0.0023941, -0.0017236, 0.0022022, -0.0030394, 0.0034017
9: -0.0042163, -0.0038756, -0.0041997, -0.0038610, -0.0002935, 0.0002622

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006766
time: 2.05 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006886, upper bound: 0.0006725
time: 2.05 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037641, -0.0002973, -0.0039371, -0.0004552, -0.0026819, 0.0030183
1: -0.0039999, -0.0030225, -0.0040487, -0.0030670, -0.0007561, 0.0008510
2: 0.0090477, 0.0162594, 0.0086879, 0.0159310, -0.0055790, 0.0062787
3: 0.0028246, 0.0037790, 0.0027770, 0.0037355, -0.0007383, 0.0008309
4: -0.0060594, -0.0006698, -0.0058139, -0.0004009, -0.0046923, 0.0041694
5: 0.9938228, 0.9953201, 0.9938909, 0.9953949, -0.0013037, 0.0011584
6: 0.0022766, 0.0036358, 0.0023385, 0.0037036, -0.0011833, 0.0010515
7: -0.0148857, -0.0098135, -0.0146547, -0.0095604, -0.0044160, 0.0039239
8: -0.0015550, 0.0023927, -0.0017520, 0.0022129, -0.0030540, 0.0034370
9: -0.0042162, -0.0038756, -0.0042007, -0.0038586, -0.0002965, 0.0002635

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006782
time: 2.12 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006890, upper bound: 0.0006743
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038480, -0.0003263, -0.0039181, -0.0007041, -0.0027169, 0.0028455
1: -0.0040236, -0.0030307, -0.0040433, -0.0031372, -0.0007660, 0.0008022
2: 0.0088732, 0.0161990, 0.0087273, 0.0154132, -0.0056517, 0.0059192
3: 0.0028015, 0.0037710, 0.0027822, 0.0036670, -0.0007479, 0.0007833
4: -0.0060142, -0.0005394, -0.0054270, -0.0004304, -0.0044236, 0.0042237
5: 0.9938353, 0.9953563, 0.9939985, 0.9953867, -0.0012290, 0.0011735
6: 0.0022880, 0.0036686, 0.0024361, 0.0036961, -0.0011156, 0.0010652
7: -0.0148432, -0.0096908, -0.0142905, -0.0095882, -0.0041631, 0.0039750
8: -0.0016505, 0.0023596, -0.0017304, 0.0019295, -0.0030938, 0.0032402
9: -0.0042133, -0.0038673, -0.0041762, -0.0038604, -0.0002795, 0.0002669

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007150
time: 2.13 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007150
time: 2.39 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037630, -0.0000865, -0.0039181, -0.0007041, -0.0026425, 0.0031138
1: -0.0039996, -0.0029630, -0.0040433, -0.0031372, -0.0007450, 0.0008779
2: 0.0090500, 0.0166980, 0.0087273, 0.0154132, -0.0054969, 0.0064773
3: 0.0028249, 0.0038370, 0.0027822, 0.0036670, -0.0007274, 0.0008572
4: -0.0063872, -0.0006715, -0.0054270, -0.0004304, -0.0048408, 0.0041081
5: 0.9937317, 0.9953196, 0.9939985, 0.9953867, -0.0013449, 0.0011413
6: 0.0021939, 0.0036353, 0.0024361, 0.0036961, -0.0012208, 0.0010360
7: -0.0151942, -0.0098151, -0.0142905, -0.0095882, -0.0045557, 0.0038661
8: -0.0015537, 0.0026328, -0.0017304, 0.0019295, -0.0030090, 0.0035457
9: -0.0042369, -0.0038757, -0.0041762, -0.0038604, -0.0003059, 0.0002596

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007168
time: 1.96 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007168
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038480, -0.0003282, -0.0039485, -0.0006885, -0.0026330, 0.0028903
1: -0.0040235, -0.0030312, -0.0040519, -0.0031328, -0.0007424, 0.0008149
2: 0.0088733, 0.0161951, 0.0086641, 0.0154456, -0.0054773, 0.0060125
3: 0.0028015, 0.0037705, 0.0027738, 0.0036713, -0.0007248, 0.0007957
4: -0.0060113, -0.0005395, -0.0054512, -0.0003831, -0.0044934, 0.0040934
5: 0.9938362, 0.9953563, 0.9939917, 0.9953998, -0.0012484, 0.0011373
6: 0.0022887, 0.0036686, 0.0024299, 0.0037081, -0.0011332, 0.0010323
7: -0.0148405, -0.0096908, -0.0143134, -0.0095437, -0.0042287, 0.0038523
8: -0.0016505, 0.0023575, -0.0017650, 0.0019472, -0.0029983, 0.0032912
9: -0.0042131, -0.0038673, -0.0041777, -0.0038575, -0.0002840, 0.0002587

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007187
time: 1.55 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007195
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037629, -0.0000874, -0.0039485, -0.0006885, -0.0025695, 0.0031567
1: -0.0039996, -0.0029633, -0.0040519, -0.0031328, -0.0007245, 0.0008900
2: 0.0090503, 0.0166960, 0.0086641, 0.0154456, -0.0053452, 0.0065667
3: 0.0028250, 0.0038367, 0.0027738, 0.0036713, -0.0007073, 0.0008690
4: -0.0063857, -0.0006717, -0.0054512, -0.0003831, -0.0049075, 0.0039947
5: 0.9937321, 0.9953196, 0.9939917, 0.9953998, -0.0013635, 0.0011098
6: 0.0021943, 0.0036353, 0.0024299, 0.0037081, -0.0012376, 0.0010074
7: -0.0151928, -0.0098153, -0.0143134, -0.0095437, -0.0046185, 0.0037594
8: -0.0015536, 0.0026317, -0.0017650, 0.0019472, -0.0029260, 0.0035946
9: -0.0042368, -0.0038757, -0.0041777, -0.0038575, -0.0003101, 0.0002524

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007205
time: 1.90 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007208
time: 2.23 seconds

## BFS NS instance: NS_A2_B1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0038733, -0.0003264, -0.0037540, -0.0004721, -0.0027453, 0.0027961
1: -0.0040307, -0.0030307, -0.0039970, -0.0030718, -0.0007740, 0.0007883
2: 0.0088205, 0.0161989, 0.0090689, 0.0158957, -0.0057108, 0.0058165
3: 0.0027945, 0.0037710, 0.0028274, 0.0037308, -0.0007557, 0.0007697
4: -0.0060142, -0.0005000, -0.0057876, -0.0006856, -0.0043469, 0.0042679
5: 0.9938353, 0.9953673, 0.9938983, 0.9953157, -0.0012077, 0.0011858
6: 0.0022880, 0.0036786, 0.0023451, 0.0036318, -0.0010962, 0.0010763
7: -0.0148432, -0.0096537, -0.0146299, -0.0098284, -0.0040909, 0.0040166
8: -0.0016794, 0.0023596, -0.0015434, 0.0021936, -0.0031261, 0.0031840
9: -0.0042133, -0.0038648, -0.0041990, -0.0038766, -0.0002747, 0.0002697

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006786, upper bound: 0.0006773
time: 1.95 seconds

## Relational analysis of NS_A2_B1_B2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006770
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038733, -0.0003264, -0.0037535, -0.0002575, -0.0029040, 0.0027670
1: -0.0040307, -0.0030307, -0.0039969, -0.0030112, -0.0008187, 0.0007801
2: 0.0088205, 0.0161989, 0.0090699, 0.0163423, -0.0060409, 0.0057560
3: 0.0027945, 0.0037710, 0.0028276, 0.0037899, -0.0007994, 0.0007617
4: -0.0060142, -0.0005000, -0.0061213, -0.0006864, -0.0043017, 0.0045146
5: 0.9938353, 0.9953673, 0.9938055, 0.9953155, -0.0011951, 0.0012543
6: 0.0022880, 0.0036786, 0.0022610, 0.0036316, -0.0010848, 0.0011385
7: -0.0148432, -0.0096537, -0.0149440, -0.0098291, -0.0040483, 0.0042488
8: -0.0016794, 0.0023596, -0.0015428, 0.0024381, -0.0033068, 0.0031508
9: -0.0042133, -0.0038648, -0.0042201, -0.0038766, -0.0002718, 0.0002853

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006750, upper bound: 0.0006838
time: 2.18 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006752, upper bound: 0.0006766
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0038733, -0.0003282, -0.0037848, -0.0004530, -0.0027601, 0.0028374
1: -0.0040307, -0.0030312, -0.0040057, -0.0030664, -0.0007782, 0.0008000
2: 0.0088207, 0.0161950, 0.0090046, 0.0159356, -0.0057416, 0.0059024
3: 0.0027946, 0.0037705, 0.0028189, 0.0037361, -0.0007598, 0.0007811
4: -0.0060113, -0.0005002, -0.0058174, -0.0006376, -0.0044111, 0.0042909
5: 0.9938361, 0.9953673, 0.9938900, 0.9953291, -0.0012255, 0.0011921
6: 0.0022887, 0.0036785, 0.0023376, 0.0036439, -0.0011124, 0.0010821
7: -0.0148404, -0.0096538, -0.0146580, -0.0097832, -0.0041513, 0.0040382
8: -0.0016793, 0.0023575, -0.0015786, 0.0022154, -0.0031430, 0.0032310
9: -0.0042131, -0.0038649, -0.0042009, -0.0038735, -0.0002788, 0.0002712

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006782, upper bound: 0.0006802
time: 2.09 seconds

## Relational analysis of NS_A2_B1_B2_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006796
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038733, -0.0003282, -0.0037847, -0.0002393, -0.0029191, 0.0028099
1: -0.0040307, -0.0030312, -0.0040057, -0.0030061, -0.0008230, 0.0007922
2: 0.0088207, 0.0161950, 0.0090048, 0.0163801, -0.0060724, 0.0058452
3: 0.0027946, 0.0037705, 0.0028189, 0.0037949, -0.0008036, 0.0007735
4: -0.0060113, -0.0005002, -0.0061496, -0.0006378, -0.0043683, 0.0045382
5: 0.9938361, 0.9953673, 0.9937977, 0.9953290, -0.0012137, 0.0012608
6: 0.0022887, 0.0036785, 0.0022538, 0.0036438, -0.0011016, 0.0011445
7: -0.0148404, -0.0096538, -0.0149706, -0.0097834, -0.0041111, 0.0042709
8: -0.0016793, 0.0023575, -0.0015785, 0.0024588, -0.0033241, 0.0031997
9: -0.0042131, -0.0038649, -0.0042219, -0.0038736, -0.0002761, 0.0002868

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B2_B2_B2_B1

### Relational analysis result of NS_A2_B1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006751, upper bound: 0.0006868
time: 2.06 seconds

## Relational analysis of NS_A2_B1_B2_B2_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006804
time: 2.20 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038484, -0.0003236, -0.0039717, -0.0006699, -0.0026513, 0.0029465
1: -0.0040237, -0.0030299, -0.0040584, -0.0031275, -0.0007475, 0.0008307
2: 0.0088723, 0.0162047, 0.0086159, 0.0154844, -0.0055153, 0.0061294
3: 0.0028014, 0.0037717, 0.0027675, 0.0036764, -0.0007299, 0.0008111
4: -0.0060185, -0.0005387, -0.0054802, -0.0003471, -0.0045807, 0.0041218
5: 0.9938341, 0.9953565, 0.9939837, 0.9954097, -0.0012727, 0.0011452
6: 0.0022869, 0.0036688, 0.0024226, 0.0037171, -0.0011552, 0.0010395
7: -0.0148473, -0.0096901, -0.0143406, -0.0095098, -0.0043110, 0.0038791
8: -0.0016510, 0.0023628, -0.0017913, 0.0019684, -0.0030191, 0.0033552
9: -0.0042136, -0.0038673, -0.0041796, -0.0038552, -0.0002895, 0.0002605

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007266
time: 1.85 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007274
time: 2.19 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0037633, -0.0000837, -0.0039717, -0.0006699, -0.0025877, 0.0032107
1: -0.0039997, -0.0029623, -0.0040584, -0.0031275, -0.0007296, 0.0009052
2: 0.0090494, 0.0167037, 0.0086159, 0.0154844, -0.0053830, 0.0066788
3: 0.0028248, 0.0038378, 0.0027675, 0.0036764, -0.0007124, 0.0008838
4: -0.0063915, -0.0006711, -0.0054802, -0.0003471, -0.0049913, 0.0040229
5: 0.9937305, 0.9953199, 0.9939837, 0.9954097, -0.0013867, 0.0011177
6: 0.0021928, 0.0036354, 0.0024226, 0.0037171, -0.0012587, 0.0010145
7: -0.0151982, -0.0098147, -0.0143406, -0.0095098, -0.0046974, 0.0037860
8: -0.0015541, 0.0026359, -0.0017913, 0.0019684, -0.0029466, 0.0036560
9: -0.0042372, -0.0038757, -0.0041796, -0.0038552, -0.0003154, 0.0002542

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007282
time: 1.85 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007286
time: 2.25 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038484, -0.0003260, -0.0039953, -0.0006598, -0.0026655, 0.0029805
1: -0.0040237, -0.0030306, -0.0040651, -0.0031247, -0.0007515, 0.0008403
2: 0.0088724, 0.0161998, 0.0085667, 0.0155054, -0.0055448, 0.0062001
3: 0.0028014, 0.0037711, 0.0027610, 0.0036792, -0.0007338, 0.0008205
4: -0.0060149, -0.0005388, -0.0054959, -0.0003104, -0.0046336, 0.0041438
5: 0.9938351, 0.9953565, 0.9939793, 0.9954200, -0.0012873, 0.0011513
6: 0.0022878, 0.0036688, 0.0024187, 0.0037264, -0.0011685, 0.0010450
7: -0.0148438, -0.0096902, -0.0143554, -0.0094752, -0.0043607, 0.0038998
8: -0.0016510, 0.0023601, -0.0018183, 0.0019799, -0.0030352, 0.0033940
9: -0.0042134, -0.0038673, -0.0041806, -0.0038529, -0.0002928, 0.0002619

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007293
time: 1.77 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007293
time: 2.29 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0037632, -0.0000847, -0.0039953, -0.0006598, -0.0026018, 0.0032434
1: -0.0039996, -0.0029625, -0.0040651, -0.0031247, -0.0007335, 0.0009144
2: 0.0090497, 0.0167016, 0.0085667, 0.0155054, -0.0054122, 0.0067470
3: 0.0028249, 0.0038375, 0.0027610, 0.0036792, -0.0007162, 0.0008929
4: -0.0063899, -0.0006713, -0.0054959, -0.0003104, -0.0050423, 0.0040447
5: 0.9937310, 0.9953197, 0.9939793, 0.9954200, -0.0014009, 0.0011237
6: 0.0021932, 0.0036354, 0.0024187, 0.0037264, -0.0012716, 0.0010200
7: -0.0151967, -0.0098149, -0.0143554, -0.0094752, -0.0047453, 0.0038065
8: -0.0015539, 0.0026348, -0.0018183, 0.0019799, -0.0029626, 0.0036933
9: -0.0042371, -0.0038757, -0.0041806, -0.0038529, -0.0003186, 0.0002556

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 220

## Relational analysis of NS_A2_B2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007305
time: 1.97 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007305
time: 2.10 seconds

## BFS NS instance: NS_A2_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0038738, -0.0003236, -0.0038045, -0.0004395, -0.0027752, 0.0028980
1: -0.0040308, -0.0030299, -0.0040113, -0.0030626, -0.0007824, 0.0008170
2: 0.0088196, 0.0162046, 0.0089637, 0.0159637, -0.0057729, 0.0060284
3: 0.0027944, 0.0037717, 0.0028135, 0.0037398, -0.0007639, 0.0007978
4: -0.0060184, -0.0004993, -0.0058384, -0.0006070, -0.0045052, 0.0043143
5: 0.9938341, 0.9953675, 0.9938841, 0.9953377, -0.0012517, 0.0011986
6: 0.0022869, 0.0036787, 0.0023323, 0.0036516, -0.0011362, 0.0010880
7: -0.0148472, -0.0096531, -0.0146777, -0.0097544, -0.0042399, 0.0040602
8: -0.0016799, 0.0023627, -0.0016010, 0.0022308, -0.0031601, 0.0033000
9: -0.0042136, -0.0038648, -0.0042022, -0.0038716, -0.0002847, 0.0002726

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006781, upper bound: 0.0006864
time: 1.91 seconds

## Relational analysis of NS_A2_B2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006862
time: 1.83 seconds

## BFS NS instance: NS_A2_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038738, -0.0003236, -0.0038003, -0.0002274, -0.0029328, 0.0028707
1: -0.0040308, -0.0030299, -0.0040101, -0.0030028, -0.0008269, 0.0008094
2: 0.0088196, 0.0162046, 0.0089725, 0.0164049, -0.0061008, 0.0059717
3: 0.0027944, 0.0037717, 0.0028147, 0.0037982, -0.0008073, 0.0007903
4: -0.0060184, -0.0004993, -0.0061681, -0.0006136, -0.0044629, 0.0045594
5: 0.9938341, 0.9953675, 0.9937925, 0.9953358, -0.0012399, 0.0012667
6: 0.0022869, 0.0036787, 0.0022492, 0.0036499, -0.0011255, 0.0011498
7: -0.0148472, -0.0096531, -0.0149880, -0.0097606, -0.0042001, 0.0042909
8: -0.0016799, 0.0023627, -0.0015962, 0.0024724, -0.0033396, 0.0032689
9: -0.0042136, -0.0038648, -0.0042230, -0.0038720, -0.0002820, 0.0002881

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B2_B1_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006755, upper bound: 0.0006937
time: 2.11 seconds

## Relational analysis of NS_A2_B2_B2_B1_B2_B2

### Relational analysis result of NS_A2_B2_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006862
time: 2.21 seconds

## BFS NS instance: NS_A2_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0038737, -0.0003260, -0.0038287, -0.0004251, -0.0027885, 0.0029289
1: -0.0040308, -0.0030306, -0.0040181, -0.0030585, -0.0007862, 0.0008258
2: 0.0088198, 0.0161998, 0.0089135, 0.0159936, -0.0058005, 0.0060926
3: 0.0027945, 0.0037711, 0.0028069, 0.0037438, -0.0007676, 0.0008063
4: -0.0060148, -0.0004995, -0.0058608, -0.0005695, -0.0045532, 0.0043350
5: 0.9938352, 0.9953674, 0.9938779, 0.9953480, -0.0012650, 0.0012044
6: 0.0022878, 0.0036787, 0.0023267, 0.0036610, -0.0011483, 0.0010932
7: -0.0148438, -0.0096532, -0.0146988, -0.0097191, -0.0042851, 0.0040797
8: -0.0016798, 0.0023601, -0.0016285, 0.0022472, -0.0031752, 0.0033351
9: -0.0042134, -0.0038648, -0.0042036, -0.0038692, -0.0002877, 0.0002739

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006782, upper bound: 0.0006888
time: 2.13 seconds

## Relational analysis of NS_A2_B2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006885
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038737, -0.0003260, -0.0038281, -0.0002117, -0.0029468, 0.0029036
1: -0.0040308, -0.0030306, -0.0040179, -0.0029983, -0.0008308, 0.0008186
2: 0.0088198, 0.0161998, 0.0089147, 0.0164375, -0.0061299, 0.0060400
3: 0.0027945, 0.0037711, 0.0028070, 0.0038025, -0.0008112, 0.0007993
4: -0.0060148, -0.0004995, -0.0061925, -0.0005704, -0.0045139, 0.0045811
5: 0.9938352, 0.9953674, 0.9937857, 0.9953478, -0.0012541, 0.0012728
6: 0.0022878, 0.0036787, 0.0022430, 0.0036608, -0.0011383, 0.0011553
7: -0.0148438, -0.0096532, -0.0150110, -0.0097200, -0.0042481, 0.0043114
8: -0.0016798, 0.0023601, -0.0016278, 0.0024902, -0.0033555, 0.0033063
9: -0.0042134, -0.0038648, -0.0042246, -0.0038693, -0.0002853, 0.0002895

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006782, upper bound: 0.0006888
time: 2.08 seconds

## Relational analysis of NS_A2_B2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006752, upper bound: 0.0006881
time: 2.14 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.06 seconds
NS_A1_B1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007079, upper bound: 0.0007052
NS_A1_B1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007097, upper bound: 0.0007055
NS_A1_B1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0007069
NS_A1_B1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007097, upper bound: 0.0007115
NS_A1_B1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006894, upper bound: 0.0006770
NS_A1_B1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006828, upper bound: 0.0006772
NS_A1_B1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006912, upper bound: 0.0006774
NS_A1_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006772
NS_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0006884
NS_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0006920
NS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0006917
NS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0006920
NS_A1_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006893, upper bound: 0.0006668
NS_A1_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006886, upper bound: 0.0006626
NS_A1_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006696
NS_A1_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006886, upper bound: 0.0006661
NS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007101
NS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007156, upper bound: 0.0007096
NS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007135, upper bound: 0.0007093
NS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007156, upper bound: 0.0007093
NS_A1_B2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006859, upper bound: 0.0006890
NS_A1_B2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006864, upper bound: 0.0006824
NS_A1_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006864, upper bound: 0.0006913
NS_A1_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006864, upper bound: 0.0006860
NS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007017
NS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007270, upper bound: 0.0007013
NS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0007013
NS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007293, upper bound: 0.0007012
NS_A1_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006766
NS_A1_B2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006886, upper bound: 0.0006725
NS_A1_B2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006888, upper bound: 0.0006782
NS_A1_B2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006890, upper bound: 0.0006743
NS_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007150
NS_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007150
NS_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007168
NS_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007168
NS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007187
NS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007195
NS_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007205
NS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007208
NS_A2_B1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006786, upper bound: 0.0006773
NS_A2_B1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006770
NS_A2_B1_B2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006750, upper bound: 0.0006838
NS_A2_B1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006752, upper bound: 0.0006766
NS_A2_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006782, upper bound: 0.0006802
NS_A2_B1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006796
NS_A2_B1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006751, upper bound: 0.0006868
NS_A2_B1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006804
NS_A2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007266
NS_A2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007274
NS_A2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007282
NS_A2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007286
NS_A2_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007293
NS_A2_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007293
NS_A2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007013, upper bound: 0.0007305
NS_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0007017, upper bound: 0.0007305
NS_A2_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006781, upper bound: 0.0006864
NS_A2_B2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006862
NS_A2_B2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006755, upper bound: 0.0006937
NS_A2_B2_B2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006862
NS_A2_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006782, upper bound: 0.0006888
NS_A2_B2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006747, upper bound: 0.0006885
NS_A2_B2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006782, upper bound: 0.0006888
NS_A2_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 6.06
Output dim: 5, lower bound: -0.0006752, upper bound: 0.0006881

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0038483, -0.0005411, -0.0038375, -0.0004944, -0.0026448, 0.0027320
1: -0.0040236, -0.0030912, -0.0040206, -0.0030780, -0.0007457, 0.0007703
2: 0.0088726, 0.0157522, 0.0088950, 0.0158495, -0.0055017, 0.0056832
3: 0.0028014, 0.0037118, 0.0028044, 0.0037247, -0.0007281, 0.0007521
4: -0.0056803, -0.0005390, -0.0057530, -0.0005557, -0.0042472, 0.0041116
5: 0.9939281, 0.9953565, 0.9939079, 0.9953519, -0.0011800, 0.0011423
6: 0.0023722, 0.0036687, 0.0023538, 0.0036645, -0.0010711, 0.0010369
7: -0.0145290, -0.0096904, -0.0145974, -0.0097061, -0.0039971, 0.0038695
8: -0.0016508, 0.0021150, -0.0016386, 0.0021683, -0.0030116, 0.0031110
9: -0.0041922, -0.0038673, -0.0041968, -0.0038684, -0.0002684, 0.0002598

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007177, upper bound: 0.0006813
time: 2.20 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007177, upper bound: 0.0006813
time: 2.07 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038483, -0.0005432, -0.0038679, -0.0004793, -0.0026584, 0.0027861
1: -0.0040236, -0.0030918, -0.0040292, -0.0030738, -0.0007495, 0.0007855
2: 0.0088726, 0.0157479, 0.0088318, 0.0158808, -0.0055301, 0.0057956
3: 0.0028014, 0.0037113, 0.0027960, 0.0037289, -0.0007318, 0.0007670
4: -0.0056772, -0.0005389, -0.0057765, -0.0005085, -0.0043313, 0.0041328
5: 0.9939290, 0.9953566, 0.9939013, 0.9953650, -0.0012034, 0.0011482
6: 0.0023730, 0.0036688, 0.0023479, 0.0036764, -0.0010923, 0.0010422
7: -0.0145260, -0.0096903, -0.0146195, -0.0096617, -0.0040762, 0.0038895
8: -0.0016509, 0.0021127, -0.0016732, 0.0021855, -0.0030272, 0.0031725
9: -0.0041920, -0.0038673, -0.0041983, -0.0038654, -0.0002737, 0.0002612

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0006866
time: 1.98 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0006869
time: 2.07 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038428, -0.0006242, -0.0037592, -0.0001766, -0.0029847, 0.0024790
1: -0.0040221, -0.0031147, -0.0039985, -0.0029884, -0.0008415, 0.0006989
2: 0.0088841, 0.0155793, 0.0090580, 0.0165105, -0.0062088, 0.0051569
3: 0.0028030, 0.0036890, 0.0028260, 0.0038122, -0.0008216, 0.0006824
4: -0.0055512, -0.0005476, -0.0062470, -0.0006775, -0.0038539, 0.0046401
5: 0.9939640, 0.9953541, 0.9937707, 0.9953180, -0.0010707, 0.0012891
6: 0.0024047, 0.0036666, 0.0022293, 0.0036338, -0.0009719, 0.0011702
7: -0.0144074, -0.0096985, -0.0150623, -0.0098207, -0.0036270, 0.0043668
8: -0.0016445, 0.0020204, -0.0015494, 0.0025301, -0.0033987, 0.0028229
9: -0.0041840, -0.0038679, -0.0042280, -0.0038761, -0.0002435, 0.0002932

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007011, upper bound: 0.0006619
time: 1.53 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006946, upper bound: 0.0006618
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038708, -0.0006092, -0.0037591, -0.0001769, -0.0030252, 0.0024931
1: -0.0040300, -0.0031104, -0.0039985, -0.0029885, -0.0008529, 0.0007029
2: 0.0088257, 0.0156106, 0.0090582, 0.0165098, -0.0062931, 0.0051862
3: 0.0027952, 0.0036931, 0.0028260, 0.0038121, -0.0008328, 0.0006863
4: -0.0055746, -0.0005039, -0.0062465, -0.0006777, -0.0038758, 0.0047031
5: 0.9939574, 0.9953662, 0.9937708, 0.9953179, -0.0010768, 0.0013066
6: 0.0023988, 0.0036776, 0.0022294, 0.0036338, -0.0009774, 0.0011860
7: -0.0144294, -0.0096574, -0.0150618, -0.0098209, -0.0036476, 0.0044261
8: -0.0016765, 0.0020376, -0.0015492, 0.0025298, -0.0034448, 0.0028389
9: -0.0041855, -0.0038651, -0.0042280, -0.0038761, -0.0002449, 0.0002972

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007035, upper bound: 0.0006618
time: 2.13 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006618
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038434, -0.0006195, -0.0038980, -0.0006003, -0.0027303, 0.0026962
1: -0.0040222, -0.0031133, -0.0040377, -0.0031079, -0.0007698, 0.0007601
2: 0.0088829, 0.0155891, 0.0087692, 0.0156290, -0.0056796, 0.0056086
3: 0.0028028, 0.0036903, 0.0027878, 0.0036955, -0.0007516, 0.0007422
4: -0.0055585, -0.0005466, -0.0055883, -0.0004617, -0.0041915, 0.0042446
5: 0.9939619, 0.9953543, 0.9939536, 0.9953780, -0.0011645, 0.0011793
6: 0.0024029, 0.0036668, 0.0023954, 0.0036882, -0.0010570, 0.0010704
7: -0.0144143, -0.0096976, -0.0144423, -0.0096177, -0.0039447, 0.0039946
8: -0.0016452, 0.0020258, -0.0017074, 0.0020476, -0.0031090, 0.0030701
9: -0.0041845, -0.0038678, -0.0041864, -0.0038624, -0.0002649, 0.0002682

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007034, upper bound: 0.0007019
time: 1.80 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007028
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0038714, -0.0006066, -0.0038980, -0.0006038, -0.0026385, 0.0027103
1: -0.0040302, -0.0031097, -0.0040376, -0.0031089, -0.0007439, 0.0007641
2: 0.0088245, 0.0156160, 0.0087693, 0.0156219, -0.0054886, 0.0056380
3: 0.0027951, 0.0036938, 0.0027878, 0.0036946, -0.0007263, 0.0007461
4: -0.0055785, -0.0005030, -0.0055830, -0.0004617, -0.0042135, 0.0041018
5: 0.9939563, 0.9953665, 0.9939551, 0.9953780, -0.0011706, 0.0011396
6: 0.0023978, 0.0036778, 0.0023967, 0.0036882, -0.0010626, 0.0010344
7: -0.0144332, -0.0096566, -0.0144373, -0.0096177, -0.0039654, 0.0038603
8: -0.0016772, 0.0020405, -0.0017074, 0.0020437, -0.0030044, 0.0030863
9: -0.0041858, -0.0038650, -0.0041861, -0.0038624, -0.0002663, 0.0002592

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007056, upper bound: 0.0007026
time: 1.53 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007030, upper bound: 0.0007030
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038434, -0.0006195, -0.0038102, -0.0003593, -0.0028616, 0.0026202
1: -0.0040222, -0.0031133, -0.0040129, -0.0030400, -0.0008068, 0.0007387
2: 0.0088829, 0.0155891, 0.0089518, 0.0161304, -0.0059527, 0.0054505
3: 0.0028028, 0.0036903, 0.0028119, 0.0037619, -0.0007877, 0.0007213
4: -0.0055585, -0.0005466, -0.0059630, -0.0005981, -0.0040733, 0.0044487
5: 0.9939619, 0.9953543, 0.9938495, 0.9953401, -0.0011317, 0.0012360
6: 0.0024029, 0.0036668, 0.0023009, 0.0036538, -0.0010272, 0.0011219
7: -0.0144143, -0.0096976, -0.0147950, -0.0097461, -0.0038335, 0.0041867
8: -0.0016452, 0.0020258, -0.0016075, 0.0023221, -0.0032585, 0.0029836
9: -0.0041845, -0.0038678, -0.0042101, -0.0038710, -0.0002574, 0.0002811

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006909, upper bound: 0.0006794
time: 2.17 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006851, upper bound: 0.0006791
time: 2.40 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0038714, -0.0006066, -0.0038101, -0.0003609, -0.0029022, 0.0026342
1: -0.0040302, -0.0031097, -0.0040129, -0.0030404, -0.0008182, 0.0007427
2: 0.0088245, 0.0156160, 0.0089520, 0.0161270, -0.0060372, 0.0054797
3: 0.0027951, 0.0036938, 0.0028120, 0.0037615, -0.0007989, 0.0007252
4: -0.0055785, -0.0005030, -0.0059605, -0.0005983, -0.0040952, 0.0045119
5: 0.9939563, 0.9953665, 0.9938502, 0.9953400, -0.0011378, 0.0012535
6: 0.0023978, 0.0036778, 0.0023015, 0.0036538, -0.0010327, 0.0011378
7: -0.0144332, -0.0096566, -0.0147926, -0.0097462, -0.0038540, 0.0042462
8: -0.0016772, 0.0020405, -0.0016074, 0.0023203, -0.0033048, 0.0029996
9: -0.0041858, -0.0038650, -0.0042099, -0.0038711, -0.0002588, 0.0002851

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006931, upper bound: 0.0006794
time: 1.84 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006851, upper bound: 0.0006794
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038434, -0.0006195, -0.0038939, -0.0003862, -0.0027416, 0.0026367
1: -0.0040222, -0.0031133, -0.0040365, -0.0030476, -0.0007730, 0.0007434
2: 0.0088829, 0.0155891, 0.0087777, 0.0160744, -0.0057031, 0.0054849
3: 0.0028028, 0.0036903, 0.0027889, 0.0037545, -0.0007547, 0.0007258
4: -0.0055585, -0.0005466, -0.0059211, -0.0004680, -0.0040990, 0.0042622
5: 0.9939619, 0.9953543, 0.9938611, 0.9953762, -0.0011388, 0.0011842
6: 0.0024029, 0.0036668, 0.0023114, 0.0036866, -0.0010337, 0.0010749
7: -0.0144143, -0.0096976, -0.0147556, -0.0096236, -0.0038577, 0.0040112
8: -0.0016452, 0.0020258, -0.0017028, 0.0022914, -0.0031219, 0.0030024
9: -0.0041845, -0.0038678, -0.0042074, -0.0038628, -0.0002590, 0.0002693

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006946, upper bound: 0.0006735
time: 2.14 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006837, upper bound: 0.0006703
time: 2.23 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038434, -0.0006195, -0.0038060, -0.0001478, -0.0030095, 0.0025586
1: -0.0040222, -0.0031133, -0.0040117, -0.0029803, -0.0008485, 0.0007214
2: 0.0088829, 0.0155891, 0.0089606, 0.0165704, -0.0062604, 0.0053224
3: 0.0028028, 0.0036903, 0.0028131, 0.0038201, -0.0008285, 0.0007043
4: -0.0055585, -0.0005466, -0.0062918, -0.0006047, -0.0039776, 0.0046786
5: 0.9939619, 0.9953543, 0.9937582, 0.9953383, -0.0011051, 0.0012999
6: 0.0024029, 0.0036668, 0.0022180, 0.0036522, -0.0010031, 0.0011799
7: -0.0144143, -0.0096976, -0.0151044, -0.0097522, -0.0037434, 0.0044031
8: -0.0016452, 0.0020258, -0.0016027, 0.0025629, -0.0034270, 0.0029135
9: -0.0041845, -0.0038678, -0.0042309, -0.0038715, -0.0002514, 0.0002957

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007010, upper bound: 0.0006704
time: 2.12 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006837, upper bound: 0.0006703
time: 2.38 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0038714, -0.0006066, -0.0038938, -0.0003890, -0.0027842, 0.0026508
1: -0.0040302, -0.0031097, -0.0040365, -0.0030483, -0.0007850, 0.0007474
2: 0.0088245, 0.0156160, 0.0087779, 0.0160687, -0.0057918, 0.0055142
3: 0.0027951, 0.0036938, 0.0027889, 0.0037537, -0.0007664, 0.0007297
4: -0.0055785, -0.0005030, -0.0059169, -0.0004682, -0.0041210, 0.0043284
5: 0.9939563, 0.9953665, 0.9938624, 0.9953762, -0.0011449, 0.0012026
6: 0.0023978, 0.0036778, 0.0023125, 0.0036866, -0.0010393, 0.0010916
7: -0.0144332, -0.0096566, -0.0147516, -0.0096237, -0.0038783, 0.0040735
8: -0.0016772, 0.0020405, -0.0017027, 0.0022883, -0.0031704, 0.0030185
9: -0.0041858, -0.0038650, -0.0042072, -0.0038628, -0.0002604, 0.0002735

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006739
time: 2.04 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006711
time: 1.88 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0038714, -0.0006066, -0.0038058, -0.0001488, -0.0030495, 0.0025726
1: -0.0040302, -0.0031097, -0.0040117, -0.0029806, -0.0008598, 0.0007253
2: 0.0088245, 0.0156160, 0.0089609, 0.0165684, -0.0063436, 0.0053515
3: 0.0027951, 0.0036938, 0.0028131, 0.0038199, -0.0008395, 0.0007082
4: -0.0055785, -0.0005030, -0.0062903, -0.0006050, -0.0039993, 0.0047408
5: 0.9939563, 0.9953665, 0.9937586, 0.9953382, -0.0011111, 0.0013171
6: 0.0023978, 0.0036778, 0.0022183, 0.0036521, -0.0010086, 0.0011956
7: -0.0144332, -0.0096566, -0.0151030, -0.0097525, -0.0037638, 0.0044617
8: -0.0016772, 0.0020405, -0.0016025, 0.0025619, -0.0034725, 0.0029294
9: -0.0041858, -0.0038650, -0.0042308, -0.0038715, -0.0002527, 0.0002996

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007035, upper bound: 0.0006707
time: 2.32 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006707
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038480, -0.0003263, -0.0038372, -0.0007135, -0.0024479, 0.0027979
1: -0.0040236, -0.0030307, -0.0040205, -0.0031398, -0.0006901, 0.0007888
2: 0.0088732, 0.0161990, 0.0088958, 0.0153936, -0.0050920, 0.0058201
3: 0.0028015, 0.0037710, 0.0028045, 0.0036644, -0.0006738, 0.0007702
4: -0.0060142, -0.0005394, -0.0054123, -0.0005563, -0.0043496, 0.0038055
5: 0.9938353, 0.9953563, 0.9940026, 0.9953517, -0.0012085, 0.0010573
6: 0.0022880, 0.0036686, 0.0024398, 0.0036644, -0.0010969, 0.0009597
7: -0.0148432, -0.0096908, -0.0142768, -0.0097066, -0.0040935, 0.0035814
8: -0.0016505, 0.0023596, -0.0016382, 0.0019188, -0.0027874, 0.0031860
9: -0.0042133, -0.0038673, -0.0041753, -0.0038684, -0.0002749, 0.0002405

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007059
time: 2.35 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007035
time: 2.10 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038480, -0.0003263, -0.0038375, -0.0004944, -0.0026139, 0.0027708
1: -0.0040236, -0.0030307, -0.0040206, -0.0030780, -0.0007369, 0.0007812
2: 0.0088732, 0.0161990, 0.0088950, 0.0158495, -0.0054374, 0.0057638
3: 0.0028015, 0.0037710, 0.0028044, 0.0037247, -0.0007195, 0.0007627
4: -0.0060142, -0.0005394, -0.0057530, -0.0005557, -0.0043075, 0.0040635
5: 0.9938353, 0.9953563, 0.9939079, 0.9953519, -0.0011968, 0.0011290
6: 0.0022880, 0.0036686, 0.0023538, 0.0036645, -0.0010863, 0.0010248
7: -0.0148432, -0.0096908, -0.0145974, -0.0097061, -0.0040538, 0.0038242
8: -0.0016505, 0.0023596, -0.0016386, 0.0021683, -0.0029764, 0.0031551
9: -0.0042133, -0.0038673, -0.0041968, -0.0038684, -0.0002722, 0.0002568

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007058
time: 2.40 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007035
time: 2.38 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037630, -0.0000865, -0.0038372, -0.0007135, -0.0023876, 0.0030662
1: -0.0039996, -0.0029630, -0.0040205, -0.0031398, -0.0006732, 0.0008645
2: 0.0090500, 0.0166980, 0.0088958, 0.0153936, -0.0049668, 0.0063783
3: 0.0028249, 0.0038370, 0.0028045, 0.0036644, -0.0006573, 0.0008441
4: -0.0063872, -0.0006715, -0.0054123, -0.0005563, -0.0047667, 0.0037118
5: 0.9937317, 0.9953196, 0.9940026, 0.9953517, -0.0013243, 0.0010313
6: 0.0021939, 0.0036353, 0.0024398, 0.0036644, -0.0012021, 0.0009361
7: -0.0151942, -0.0098151, -0.0142768, -0.0097066, -0.0044860, 0.0034933
8: -0.0015537, 0.0026328, -0.0016382, 0.0019188, -0.0027188, 0.0034915
9: -0.0042369, -0.0038757, -0.0041753, -0.0038684, -0.0003012, 0.0002346

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006897
time: 2.04 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006843
time: 2.15 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037630, -0.0000865, -0.0038375, -0.0004944, -0.0025505, 0.0030349
1: -0.0039996, -0.0029630, -0.0040206, -0.0030780, -0.0007191, 0.0008557
2: 0.0090500, 0.0166980, 0.0088950, 0.0158495, -0.0053055, 0.0063133
3: 0.0028249, 0.0038370, 0.0028044, 0.0037247, -0.0007021, 0.0008355
4: -0.0063872, -0.0006715, -0.0057530, -0.0005557, -0.0047181, 0.0039650
5: 0.9937317, 0.9953196, 0.9939079, 0.9953519, -0.0013108, 0.0011016
6: 0.0021939, 0.0036353, 0.0023538, 0.0036645, -0.0011898, 0.0009999
7: -0.0151942, -0.0098151, -0.0145974, -0.0097061, -0.0044403, 0.0037315
8: -0.0015537, 0.0026328, -0.0016386, 0.0021683, -0.0029043, 0.0034559
9: -0.0042369, -0.0038757, -0.0041968, -0.0038684, -0.0002982, 0.0002506

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006894
time: 2.30 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006847
time: 2.54 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038480, -0.0003282, -0.0038655, -0.0006973, -0.0024623, 0.0028403
1: -0.0040235, -0.0030312, -0.0040285, -0.0031352, -0.0006942, 0.0008008
2: 0.0088733, 0.0161951, 0.0088368, 0.0154274, -0.0051221, 0.0059084
3: 0.0028015, 0.0037705, 0.0027967, 0.0036689, -0.0006778, 0.0007819
4: -0.0060113, -0.0005395, -0.0054376, -0.0005122, -0.0044156, 0.0038280
5: 0.9938362, 0.9953563, 0.9939956, 0.9953639, -0.0012268, 0.0010635
6: 0.0022887, 0.0036686, 0.0024334, 0.0036755, -0.0011135, 0.0009654
7: -0.0148405, -0.0096908, -0.0143005, -0.0096652, -0.0041556, 0.0036025
8: -0.0016505, 0.0023575, -0.0016704, 0.0019372, -0.0028039, 0.0032343
9: -0.0042131, -0.0038673, -0.0041769, -0.0038656, -0.0002790, 0.0002419

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007099
time: 2.22 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007077
time: 2.04 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038480, -0.0003282, -0.0038679, -0.0004793, -0.0026283, 0.0028149
1: -0.0040235, -0.0030312, -0.0040292, -0.0030738, -0.0007410, 0.0007936
2: 0.0088733, 0.0161951, 0.0088318, 0.0158808, -0.0054673, 0.0058556
3: 0.0028015, 0.0037705, 0.0027960, 0.0037289, -0.0007235, 0.0007749
4: -0.0060113, -0.0005395, -0.0057765, -0.0005085, -0.0043761, 0.0040859
5: 0.9938362, 0.9953563, 0.9939013, 0.9953650, -0.0012158, 0.0011352
6: 0.0022887, 0.0036686, 0.0023479, 0.0036764, -0.0011036, 0.0010304
7: -0.0148405, -0.0096908, -0.0146195, -0.0096617, -0.0041184, 0.0038453
8: -0.0016505, 0.0023575, -0.0016732, 0.0021855, -0.0029928, 0.0032054
9: -0.0042131, -0.0038673, -0.0041983, -0.0038654, -0.0002765, 0.0002582

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007103
time: 2.28 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007077
time: 2.45 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037629, -0.0000874, -0.0038655, -0.0006973, -0.0024020, 0.0031067
1: -0.0039996, -0.0029633, -0.0040285, -0.0031352, -0.0006772, 0.0008759
2: 0.0090503, 0.0166960, 0.0088368, 0.0154274, -0.0049965, 0.0064626
3: 0.0028250, 0.0038367, 0.0027967, 0.0036689, -0.0006612, 0.0008552
4: -0.0063857, -0.0006717, -0.0054376, -0.0005122, -0.0048298, 0.0037341
5: 0.9937321, 0.9953196, 0.9939956, 0.9953639, -0.0013418, 0.0010374
6: 0.0021943, 0.0036353, 0.0024334, 0.0036755, -0.0012180, 0.0009417
7: -0.0151928, -0.0098153, -0.0143005, -0.0096652, -0.0045453, 0.0035142
8: -0.0015536, 0.0026317, -0.0016704, 0.0019372, -0.0027351, 0.0035376
9: -0.0042368, -0.0038757, -0.0041769, -0.0038656, -0.0003052, 0.0002360

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006939
time: 2.02 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006886
time: 2.02 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037629, -0.0000874, -0.0038679, -0.0004793, -0.0025648, 0.0030751
1: -0.0039996, -0.0029633, -0.0040292, -0.0030738, -0.0007231, 0.0008670
2: 0.0090503, 0.0166960, 0.0088318, 0.0158808, -0.0053352, 0.0063969
3: 0.0028250, 0.0038367, 0.0027960, 0.0037289, -0.0007060, 0.0008465
4: -0.0063857, -0.0006717, -0.0057765, -0.0005085, -0.0047807, 0.0039872
5: 0.9937321, 0.9953196, 0.9939013, 0.9953650, -0.0013282, 0.0011078
6: 0.0021943, 0.0036353, 0.0023479, 0.0036764, -0.0012056, 0.0010055
7: -0.0151928, -0.0098153, -0.0146195, -0.0096617, -0.0044991, 0.0037524
8: -0.0015536, 0.0026317, -0.0016732, 0.0021855, -0.0029205, 0.0035017
9: -0.0042368, -0.0038757, -0.0041983, -0.0038654, -0.0003021, 0.0002520

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006944
time: 2.08 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006883
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038484, -0.0003236, -0.0038925, -0.0006792, -0.0024813, 0.0029072
1: -0.0040237, -0.0030299, -0.0040361, -0.0031302, -0.0006996, 0.0008196
2: 0.0088723, 0.0162047, 0.0087806, 0.0154649, -0.0051616, 0.0060475
3: 0.0028014, 0.0037717, 0.0027893, 0.0036738, -0.0006831, 0.0008003
4: -0.0060185, -0.0005387, -0.0054657, -0.0004702, -0.0045195, 0.0038575
5: 0.9938341, 0.9953565, 0.9939877, 0.9953756, -0.0012557, 0.0010717
6: 0.0022869, 0.0036688, 0.0024263, 0.0036861, -0.0011398, 0.0009728
7: -0.0148473, -0.0096901, -0.0143269, -0.0096256, -0.0042534, 0.0036303
8: -0.0016510, 0.0023628, -0.0017012, 0.0019578, -0.0028255, 0.0033104
9: -0.0042136, -0.0038673, -0.0041786, -0.0038630, -0.0002856, 0.0002438

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007184
time: 2.20 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007143
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038484, -0.0003236, -0.0038884, -0.0004642, -0.0026453, 0.0028812
1: -0.0040237, -0.0030299, -0.0040350, -0.0030695, -0.0007458, 0.0008123
2: 0.0088723, 0.0162047, 0.0087891, 0.0159122, -0.0055027, 0.0059935
3: 0.0028014, 0.0037717, 0.0027904, 0.0037330, -0.0007282, 0.0007931
4: -0.0060185, -0.0005387, -0.0057999, -0.0004766, -0.0044792, 0.0041124
5: 0.9938341, 0.9953565, 0.9938948, 0.9953738, -0.0012444, 0.0011425
6: 0.0022869, 0.0036688, 0.0023420, 0.0036845, -0.0011296, 0.0010371
7: -0.0148473, -0.0096901, -0.0146415, -0.0096316, -0.0042154, 0.0038702
8: -0.0016510, 0.0023628, -0.0016965, 0.0022026, -0.0030122, 0.0032809
9: -0.0042136, -0.0038673, -0.0041998, -0.0038634, -0.0002831, 0.0002599

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007184
time: 2.40 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007151
time: 2.49 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037633, -0.0000837, -0.0038925, -0.0006792, -0.0024209, 0.0031713
1: -0.0039997, -0.0029623, -0.0040361, -0.0031302, -0.0006825, 0.0008941
2: 0.0090494, 0.0167037, 0.0087806, 0.0154649, -0.0050359, 0.0065969
3: 0.0028248, 0.0038378, 0.0027893, 0.0036738, -0.0006664, 0.0008730
4: -0.0063915, -0.0006711, -0.0054657, -0.0004702, -0.0049301, 0.0037635
5: 0.9937305, 0.9953199, 0.9939877, 0.9953756, -0.0013697, 0.0010456
6: 0.0021928, 0.0036354, 0.0024263, 0.0036861, -0.0012433, 0.0009491
7: -0.0151982, -0.0098147, -0.0143269, -0.0096256, -0.0046398, 0.0035419
8: -0.0015541, 0.0026359, -0.0017012, 0.0019578, -0.0027566, 0.0036112
9: -0.0042372, -0.0038757, -0.0041786, -0.0038630, -0.0003116, 0.0002378

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0007010
time: 2.10 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006942
time: 2.02 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037633, -0.0000837, -0.0038884, -0.0004642, -0.0025817, 0.0031407
1: -0.0039997, -0.0029623, -0.0040350, -0.0030695, -0.0007279, 0.0008855
2: 0.0090494, 0.0167037, 0.0087891, 0.0159122, -0.0053704, 0.0065334
3: 0.0028248, 0.0038378, 0.0027904, 0.0037330, -0.0007107, 0.0008646
4: -0.0063915, -0.0006711, -0.0057999, -0.0004766, -0.0048826, 0.0040135
5: 0.9937305, 0.9953199, 0.9938948, 0.9953738, -0.0013565, 0.0011151
6: 0.0021928, 0.0036354, 0.0023420, 0.0036845, -0.0012313, 0.0010121
7: -0.0151982, -0.0098147, -0.0146415, -0.0096316, -0.0045951, 0.0037771
8: -0.0015541, 0.0026359, -0.0016965, 0.0022026, -0.0029398, 0.0035764
9: -0.0042372, -0.0038757, -0.0041998, -0.0038634, -0.0003086, 0.0002536

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0007010
time: 2.33 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006947
time: 2.33 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0038484, -0.0003260, -0.0039129, -0.0006689, -0.0024942, 0.0029404
1: -0.0040237, -0.0030306, -0.0040418, -0.0031272, -0.0007032, 0.0008290
2: 0.0088724, 0.0161998, 0.0087383, 0.0154864, -0.0051885, 0.0061167
3: 0.0028014, 0.0037711, 0.0027837, 0.0036767, -0.0006866, 0.0008094
4: -0.0060149, -0.0005388, -0.0054817, -0.0004386, -0.0045712, 0.0038776
5: 0.9938351, 0.9953565, 0.9939833, 0.9953843, -0.0012700, 0.0010773
6: 0.0022878, 0.0036688, 0.0024223, 0.0036941, -0.0011528, 0.0009779
7: -0.0148438, -0.0096902, -0.0143421, -0.0095959, -0.0043020, 0.0036492
8: -0.0016510, 0.0023601, -0.0017244, 0.0019696, -0.0028402, 0.0033483
9: -0.0042134, -0.0038673, -0.0041797, -0.0038610, -0.0002889, 0.0002450

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007201
time: 2.12 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007169
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0038484, -0.0003260, -0.0039105, -0.0004550, -0.0026587, 0.0029181
1: -0.0040237, -0.0030306, -0.0040412, -0.0030669, -0.0007496, 0.0008227
2: 0.0088724, 0.0161998, 0.0087431, 0.0159314, -0.0055307, 0.0060701
3: 0.0028014, 0.0037711, 0.0027843, 0.0037356, -0.0007319, 0.0008033
4: -0.0060149, -0.0005388, -0.0058143, -0.0004422, -0.0045364, 0.0041333
5: 0.9938351, 0.9953565, 0.9938908, 0.9953834, -0.0012604, 0.0011484
6: 0.0022878, 0.0036688, 0.0023384, 0.0036931, -0.0011440, 0.0010424
7: -0.0148438, -0.0096902, -0.0146550, -0.0095993, -0.0042693, 0.0038899
8: -0.0016510, 0.0023601, -0.0017217, 0.0022132, -0.0030275, 0.0033228
9: -0.0042134, -0.0038673, -0.0042007, -0.0038612, -0.0002867, 0.0002612

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007205
time: 2.40 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007173
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0037632, -0.0000847, -0.0039129, -0.0006689, -0.0024336, 0.0032033
1: -0.0039996, -0.0029625, -0.0040418, -0.0031272, -0.0006861, 0.0009031
2: 0.0090497, 0.0167016, 0.0087383, 0.0154864, -0.0050625, 0.0066635
3: 0.0028249, 0.0038375, 0.0027837, 0.0036767, -0.0006699, 0.0008818
4: -0.0063899, -0.0006713, -0.0054817, -0.0004386, -0.0049799, 0.0037834
5: 0.9937310, 0.9953197, 0.9939833, 0.9953843, -0.0013836, 0.0010511
6: 0.0021932, 0.0036354, 0.0024223, 0.0036941, -0.0012559, 0.0009541
7: -0.0151967, -0.0098149, -0.0143421, -0.0095959, -0.0046866, 0.0035606
8: -0.0015539, 0.0026348, -0.0017244, 0.0019696, -0.0027712, 0.0036476
9: -0.0042371, -0.0038757, -0.0041797, -0.0038610, -0.0003147, 0.0002391

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0007035
time: 2.02 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006978
time: 2.17 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0037632, -0.0000847, -0.0039105, -0.0004550, -0.0025950, 0.0031735
1: -0.0039996, -0.0029625, -0.0040412, -0.0030669, -0.0007316, 0.0008947
2: 0.0090497, 0.0167016, 0.0087431, 0.0159314, -0.0053981, 0.0066016
3: 0.0028249, 0.0038375, 0.0027843, 0.0037356, -0.0007144, 0.0008736
4: -0.0063899, -0.0006713, -0.0058143, -0.0004422, -0.0049336, 0.0040342
5: 0.9937310, 0.9953197, 0.9938908, 0.9953834, -0.0013707, 0.0011208
6: 0.0021932, 0.0036354, 0.0023384, 0.0036931, -0.0012442, 0.0010174
7: -0.0151967, -0.0098149, -0.0146550, -0.0095993, -0.0046431, 0.0037966
8: -0.0015539, 0.0026348, -0.0017217, 0.0022132, -0.0029549, 0.0036137
9: -0.0042371, -0.0038757, -0.0042007, -0.0038612, -0.0003118, 0.0002549

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 88

## Relational analysis of NS_A2_B2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0007031
time: 2.45 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006983
time: 1.66 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.03 seconds
NS_A1_B1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007177, upper bound: 0.0006813
NS_A1_B1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007177, upper bound: 0.0006813
NS_A1_B1_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0006866
NS_A1_B1_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007173, upper bound: 0.0006869
NS_A1_B1_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007011, upper bound: 0.0006619
NS_A1_B1_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006946, upper bound: 0.0006618
NS_A1_B1_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007035, upper bound: 0.0006618
NS_A1_B1_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006618
NS_A1_B2_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007034, upper bound: 0.0007019
NS_A1_B2_B1_A1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006998, upper bound: 0.0007028
NS_A1_B2_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007056, upper bound: 0.0007026
NS_A1_B2_B1_A1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007030, upper bound: 0.0007030
NS_A1_B2_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006909, upper bound: 0.0006794
NS_A1_B2_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006851, upper bound: 0.0006791
NS_A1_B2_B1_A1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006931, upper bound: 0.0006794
NS_A1_B2_B1_A1_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006851, upper bound: 0.0006794
NS_A1_B2_B2_A1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006946, upper bound: 0.0006735
NS_A1_B2_B2_A1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006837, upper bound: 0.0006703
NS_A1_B2_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007010, upper bound: 0.0006704
NS_A1_B2_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006837, upper bound: 0.0006703
NS_A1_B2_B2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006739
NS_A1_B2_B2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006711
NS_A1_B2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0007035, upper bound: 0.0006707
NS_A1_B2_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006982, upper bound: 0.0006707
NS_A2_B1_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007059
NS_A2_B1_B1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007035
NS_A2_B1_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007058
NS_A2_B1_B1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007035
NS_A2_B1_B1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006897
NS_A2_B1_B1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006843
NS_A2_B1_B1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006894
NS_A2_B1_B1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006847
NS_A2_B1_B1_B2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007099
NS_A2_B1_B1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007077
NS_A2_B1_B1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007103
NS_A2_B1_B1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007077
NS_A2_B1_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006939
NS_A2_B1_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006886
NS_A2_B1_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0006944
NS_A2_B1_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006883
NS_A2_B2_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007184
NS_A2_B2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007143
NS_A2_B2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007184
NS_A2_B2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007151
NS_A2_B2_B1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0007010
NS_A2_B2_B1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006942
NS_A2_B2_B1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0007010
NS_A2_B2_B1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006947
NS_A2_B2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007201
NS_A2_B2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007169
NS_A2_B2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007205
NS_A2_B2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006956, upper bound: 0.0007173
NS_A2_B2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006707, upper bound: 0.0007035
NS_A2_B2_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006978
NS_A2_B2_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0007031
NS_A2_B2_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.03
Output dim: 5, lower bound: -0.0006712, upper bound: 0.0006983

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0038477, -0.0005475, -0.0038304, -0.0005722, -0.0025680, 0.0027165
1: -0.0040235, -0.0030930, -0.0040186, -0.0031000, -0.0007240, 0.0007659
2: 0.0088738, 0.0157389, 0.0089099, 0.0156876, -0.0053420, 0.0056509
3: 0.0028016, 0.0037101, 0.0028064, 0.0037033, -0.0007069, 0.0007478
4: -0.0056704, -0.0005398, -0.0056321, -0.0005668, -0.0042231, 0.0039922
5: 0.9939309, 0.9953563, 0.9939414, 0.9953488, -0.0011733, 0.0011092
6: 0.0023747, 0.0036685, 0.0023843, 0.0036617, -0.0010650, 0.0010068
7: -0.0145196, -0.0096912, -0.0144836, -0.0097166, -0.0039744, 0.0037571
8: -0.0016502, 0.0021078, -0.0016304, 0.0020797, -0.0029242, 0.0030933
9: -0.0041916, -0.0038674, -0.0041892, -0.0038691, -0.0002669, 0.0002523

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007106, upper bound: 0.0006817
time: 2.37 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007106, upper bound: 0.0006813
time: 2.48 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038452, -0.0005917, -0.0038718, -0.0006509, -0.0025367, 0.0027597
1: -0.0040228, -0.0031055, -0.0040303, -0.0031222, -0.0007152, 0.0007781
2: 0.0088791, 0.0156469, 0.0088237, 0.0155238, -0.0052768, 0.0057408
3: 0.0028023, 0.0036979, 0.0027950, 0.0036816, -0.0006983, 0.0007597
4: -0.0056017, -0.0005439, -0.0055096, -0.0005024, -0.0042903, 0.0039436
5: 0.9939499, 0.9953551, 0.9939755, 0.9953666, -0.0011920, 0.0010956
6: 0.0023920, 0.0036675, 0.0024152, 0.0036780, -0.0010820, 0.0009945
7: -0.0144549, -0.0096950, -0.0143683, -0.0096559, -0.0040377, 0.0037113
8: -0.0016473, 0.0020574, -0.0016776, 0.0019900, -0.0028885, 0.0031425
9: -0.0041872, -0.0038676, -0.0041814, -0.0038650, -0.0002711, 0.0002492

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006948, upper bound: 0.0006809
time: 2.52 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006948, upper bound: 0.0006809
time: 1.84 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0038478, -0.0005495, -0.0038609, -0.0005576, -0.0025817, 0.0027711
1: -0.0040235, -0.0030936, -0.0040272, -0.0030959, -0.0007279, 0.0007813
2: 0.0088737, 0.0157347, 0.0088463, 0.0157180, -0.0053704, 0.0057644
3: 0.0028016, 0.0037095, 0.0027980, 0.0037073, -0.0007107, 0.0007628
4: -0.0056672, -0.0005398, -0.0056548, -0.0005193, -0.0043080, 0.0040135
5: 0.9939317, 0.9953563, 0.9939351, 0.9953620, -0.0011969, 0.0011151
6: 0.0023755, 0.0036685, 0.0023786, 0.0036737, -0.0010864, 0.0010121
7: -0.0145167, -0.0096911, -0.0145049, -0.0096719, -0.0040543, 0.0037772
8: -0.0016503, 0.0021055, -0.0016652, 0.0020963, -0.0029398, 0.0031555
9: -0.0041914, -0.0038674, -0.0041906, -0.0038661, -0.0002722, 0.0002536

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0006871
time: 2.48 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0006867
time: 2.46 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038452, -0.0005937, -0.0039012, -0.0006361, -0.0025513, 0.0028189
1: -0.0040228, -0.0031061, -0.0040385, -0.0031180, -0.0007193, 0.0007948
2: 0.0088790, 0.0156428, 0.0087626, 0.0155546, -0.0053072, 0.0058639
3: 0.0028023, 0.0036974, 0.0027869, 0.0036857, -0.0007023, 0.0007760
4: -0.0055986, -0.0005438, -0.0055327, -0.0004568, -0.0043823, 0.0039663
5: 0.9939508, 0.9953551, 0.9939691, 0.9953794, -0.0012175, 0.0011020
6: 0.0023928, 0.0036675, 0.0024094, 0.0036895, -0.0011052, 0.0010002
7: -0.0144520, -0.0096949, -0.0143900, -0.0096130, -0.0041243, 0.0037327
8: -0.0016473, 0.0020552, -0.0017111, 0.0020069, -0.0029052, 0.0032099
9: -0.0041870, -0.0038676, -0.0041829, -0.0038621, -0.0002769, 0.0002506

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0006865
time: 2.18 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0006873
time: 2.50 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0038479, -0.0003298, -0.0038856, -0.0007535, -0.0024025, 0.0028928
1: -0.0040235, -0.0030316, -0.0040342, -0.0031511, -0.0006774, 0.0008156
2: 0.0088734, 0.0161918, 0.0087950, 0.0153104, -0.0049978, 0.0060176
3: 0.0028015, 0.0037700, 0.0027912, 0.0036534, -0.0006614, 0.0007963
4: -0.0060088, -0.0005395, -0.0053502, -0.0004810, -0.0044972, 0.0037350
5: 0.9938368, 0.9953563, 0.9940197, 0.9953726, -0.0012495, 0.0010377
6: 0.0022893, 0.0036686, 0.0024554, 0.0036834, -0.0011341, 0.0009419
7: -0.0148381, -0.0096909, -0.0142183, -0.0096358, -0.0042324, 0.0035151
8: -0.0016504, 0.0023557, -0.0016933, 0.0018732, -0.0027358, 0.0032940
9: -0.0042130, -0.0038673, -0.0041713, -0.0038636, -0.0002842, 0.0002360

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007184
time: 2.03 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007059
time: 2.28 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038453, -0.0003760, -0.0039239, -0.0008363, -0.0023798, 0.0029306
1: -0.0040228, -0.0030447, -0.0040449, -0.0031744, -0.0006710, 0.0008263
2: 0.0088787, 0.0160956, 0.0087154, 0.0151382, -0.0049505, 0.0060963
3: 0.0028023, 0.0037573, 0.0027806, 0.0036306, -0.0006551, 0.0008067
4: -0.0059370, -0.0005436, -0.0052215, -0.0004215, -0.0045560, 0.0036997
5: 0.9938568, 0.9953552, 0.9940556, 0.9953892, -0.0012658, 0.0010279
6: 0.0023075, 0.0036676, 0.0024879, 0.0036984, -0.0011490, 0.0009330
7: -0.0147705, -0.0096947, -0.0140972, -0.0095798, -0.0042877, 0.0034819
8: -0.0016475, 0.0023030, -0.0017369, 0.0017790, -0.0027099, 0.0033371
9: -0.0042084, -0.0038676, -0.0041632, -0.0038599, -0.0002879, 0.0002338

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007143
time: 2.04 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007035
time: 2.44 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0038479, -0.0003298, -0.0038815, -0.0005411, -0.0025664, 0.0028665
1: -0.0040235, -0.0030316, -0.0040330, -0.0030912, -0.0007236, 0.0008082
2: 0.0088734, 0.0161918, 0.0088035, 0.0157523, -0.0053387, 0.0059629
3: 0.0028015, 0.0037700, 0.0027923, 0.0037119, -0.0007065, 0.0007891
4: -0.0060088, -0.0005395, -0.0056804, -0.0004873, -0.0044563, 0.0039898
5: 0.9938368, 0.9953563, 0.9939280, 0.9953708, -0.0012381, 0.0011085
6: 0.0022893, 0.0036686, 0.0023721, 0.0036818, -0.0011238, 0.0010062
7: -0.0148381, -0.0096909, -0.0145291, -0.0096417, -0.0041939, 0.0037548
8: -0.0016504, 0.0023557, -0.0016887, 0.0021151, -0.0029224, 0.0032641
9: -0.0042130, -0.0038673, -0.0041922, -0.0038640, -0.0002816, 0.0002521

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007184
time: 2.14 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007059
time: 2.31 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038453, -0.0003760, -0.0039206, -0.0006248, -0.0025415, 0.0029122
1: -0.0040228, -0.0030447, -0.0040440, -0.0031148, -0.0007166, 0.0008211
2: 0.0088787, 0.0160956, 0.0087221, 0.0155781, -0.0052869, 0.0060580
3: 0.0028023, 0.0037573, 0.0027815, 0.0036888, -0.0006996, 0.0008017
4: -0.0059370, -0.0005436, -0.0055503, -0.0004265, -0.0045274, 0.0039511
5: 0.9938568, 0.9953552, 0.9939642, 0.9953878, -0.0012578, 0.0010977
6: 0.0023075, 0.0036676, 0.0024050, 0.0036971, -0.0011417, 0.0009964
7: -0.0147705, -0.0096947, -0.0144066, -0.0095845, -0.0042608, 0.0037184
8: -0.0016475, 0.0023030, -0.0017332, 0.0020198, -0.0028941, 0.0033162
9: -0.0042084, -0.0038676, -0.0041840, -0.0038602, -0.0002861, 0.0002497

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007147
time: 2.12 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007035
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0038479, -0.0003322, -0.0039060, -0.0007453, -0.0024154, 0.0029264
1: -0.0040235, -0.0030323, -0.0040399, -0.0031488, -0.0006810, 0.0008251
2: 0.0088734, 0.0161867, 0.0087526, 0.0153275, -0.0050245, 0.0060875
3: 0.0028016, 0.0037693, 0.0027856, 0.0036556, -0.0006649, 0.0008056
4: -0.0060051, -0.0005396, -0.0053629, -0.0004493, -0.0045494, 0.0037550
5: 0.9938378, 0.9953563, 0.9940162, 0.9953814, -0.0012640, 0.0010432
6: 0.0022903, 0.0036686, 0.0024522, 0.0036914, -0.0011473, 0.0009470
7: -0.0148346, -0.0096910, -0.0142303, -0.0096059, -0.0042815, 0.0035339
8: -0.0016504, 0.0023529, -0.0017165, 0.0018826, -0.0027504, 0.0033323
9: -0.0042127, -0.0038673, -0.0041722, -0.0038616, -0.0002875, 0.0002373

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007201
time: 2.04 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007100
time: 2.23 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0038453, -0.0003782, -0.0039492, -0.0008227, -0.0023919, 0.0029650
1: -0.0040228, -0.0030453, -0.0040521, -0.0031706, -0.0006744, 0.0008359
2: 0.0088788, 0.0160910, 0.0086627, 0.0151666, -0.0049756, 0.0061678
3: 0.0028023, 0.0037567, 0.0027737, 0.0036343, -0.0006584, 0.0008162
4: -0.0059336, -0.0005436, -0.0052427, -0.0003821, -0.0046094, 0.0037185
5: 0.9938577, 0.9953552, 0.9940496, 0.9954001, -0.0012806, 0.0010331
6: 0.0023083, 0.0036676, 0.0024825, 0.0037083, -0.0011624, 0.0009377
7: -0.0147673, -0.0096947, -0.0141171, -0.0095427, -0.0043380, 0.0034995
8: -0.0016474, 0.0023005, -0.0017658, 0.0017945, -0.0027237, 0.0033763
9: -0.0042082, -0.0038676, -0.0041646, -0.0038574, -0.0002913, 0.0002350

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007169
time: 2.14 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007077
time: 2.35 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0038479, -0.0003322, -0.0039038, -0.0005325, -0.0025796, 0.0029038
1: -0.0040235, -0.0030323, -0.0040393, -0.0030888, -0.0007273, 0.0008187
2: 0.0088734, 0.0161867, 0.0087572, 0.0157701, -0.0053661, 0.0060404
3: 0.0028016, 0.0037693, 0.0027862, 0.0037142, -0.0007101, 0.0007994
4: -0.0060051, -0.0005396, -0.0056937, -0.0004527, -0.0045142, 0.0040103
5: 0.9938378, 0.9953563, 0.9939243, 0.9953805, -0.0012542, 0.0011142
6: 0.0022903, 0.0036686, 0.0023688, 0.0036905, -0.0011384, 0.0010113
7: -0.0148346, -0.0096910, -0.0145416, -0.0096092, -0.0042484, 0.0037741
8: -0.0016504, 0.0023529, -0.0017140, 0.0021249, -0.0029374, 0.0033065
9: -0.0042127, -0.0038673, -0.0041931, -0.0038619, -0.0002853, 0.0002534

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007201
time: 2.13 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007099
time: 2.49 seconds

## BFS NS instance: NS_A2_B2_B1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0038453, -0.0003782, -0.0039445, -0.0006141, -0.0025537, 0.0029483
1: -0.0040228, -0.0030453, -0.0040508, -0.0031118, -0.0007200, 0.0008312
2: 0.0088788, 0.0160910, 0.0086724, 0.0156003, -0.0053122, 0.0061330
3: 0.0028023, 0.0037567, 0.0027749, 0.0036917, -0.0007030, 0.0008116
4: -0.0059336, -0.0005436, -0.0055668, -0.0003893, -0.0045834, 0.0039700
5: 0.9938577, 0.9953552, 0.9939597, 0.9953980, -0.0012734, 0.0011030
6: 0.0023083, 0.0036676, 0.0024008, 0.0037065, -0.0011559, 0.0010012
7: -0.0147673, -0.0096947, -0.0144221, -0.0095496, -0.0043135, 0.0037362
8: -0.0016474, 0.0023005, -0.0017604, 0.0020319, -0.0029079, 0.0033572
9: -0.0042082, -0.0038676, -0.0041850, -0.0038579, -0.0002896, 0.0002509

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of NS_A2_B2_B1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007169
time: 2.05 seconds

## Relational analysis of NS_A2_B2_B1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007077
time: 2.44 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 6.32 seconds
NS_A1_B1_B2_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0007106, upper bound: 0.0006817
NS_A1_B1_B2_A1_B1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0007106, upper bound: 0.0006813
NS_A1_B1_B2_A1_B1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006948, upper bound: 0.0006809
NS_A1_B1_B2_A1_B1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006948, upper bound: 0.0006809
NS_A1_B1_B2_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0006871
NS_A1_B1_B2_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006949, upper bound: 0.0006867
NS_A1_B1_B2_A1_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0006865
NS_A1_B1_B2_A1_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0007101, upper bound: 0.0006873
NS_A2_B2_B1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007184
NS_A2_B2_B1_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007059
NS_A2_B2_B1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007143
NS_A2_B2_B1_B1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007035
NS_A2_B2_B1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007184
NS_A2_B2_B1_B1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007059
NS_A2_B2_B1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007147
NS_A2_B2_B1_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007035
NS_A2_B2_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007201
NS_A2_B2_B1_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007100
NS_A2_B2_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007169
NS_A2_B2_B1_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007077
NS_A2_B2_B1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007201
NS_A2_B2_B1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007099
NS_A2_B2_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007169
NS_A2_B2_B1_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.32
Output dim: 5, lower bound: -0.0006873, upper bound: 0.0007077

## BFS NS instance: NS_A2_B2_B1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0038426, -0.0004208, -0.0038856, -0.0007535, -0.0024477, 0.0027859
1: -0.0040220, -0.0030573, -0.0040342, -0.0031511, -0.0006901, 0.0007854
2: 0.0088846, 0.0160024, 0.0087950, 0.0153104, -0.0050918, 0.0057952
3: 0.0028030, 0.0037450, 0.0027912, 0.0036534, -0.0006738, 0.0007669
4: -0.0058673, -0.0005479, -0.0053502, -0.0004810, -0.0043310, 0.0038053
5: 0.9938761, 0.9953540, 0.9940197, 0.9953726, -0.0012033, 0.0010572
6: 0.0023250, 0.0036665, 0.0024554, 0.0036834, -0.0010922, 0.0009596
7: -0.0147050, -0.0096988, -0.0142183, -0.0096358, -0.0040759, 0.0035812
8: -0.0016443, 0.0022520, -0.0016933, 0.0018732, -0.0027872, 0.0031723
9: -0.0042040, -0.0038679, -0.0041713, -0.0038636, -0.0002737, 0.0002405

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006631, upper bound: 0.0006951
time: 2.12 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006633, upper bound: 0.0006943
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_B1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0038399, -0.0004657, -0.0039239, -0.0008363, -0.0024196, 0.0028252
1: -0.0040213, -0.0030700, -0.0040449, -0.0031744, -0.0006822, 0.0007965
2: 0.0088902, 0.0159090, 0.0087154, 0.0151382, -0.0050332, 0.0058769
3: 0.0028038, 0.0037326, 0.0027806, 0.0036306, -0.0006661, 0.0007777
4: -0.0057975, -0.0005521, -0.0052215, -0.0004215, -0.0043921, 0.0037615
5: 0.9938955, 0.9953528, 0.9940556, 0.9953892, -0.0012202, 0.0010451
6: 0.0023426, 0.0036654, 0.0024879, 0.0036984, -0.0011076, 0.0009486
7: -0.0146393, -0.0097027, -0.0140972, -0.0095798, -0.0041334, 0.0035400
8: -0.0016412, 0.0022009, -0.0017369, 0.0017790, -0.0027552, 0.0032170
9: -0.0041996, -0.0038681, -0.0041632, -0.0038599, -0.0002776, 0.0002377

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006631, upper bound: 0.0006899
time: 1.74 seconds

## Relational analysis of NS_A2_B2_B1_B1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_B1_B1_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006629, upper bound: 0.0006892
time: 2.44 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.25 + 597.92 = 602.16 seconds
