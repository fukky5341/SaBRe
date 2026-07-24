## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 43.1275729563


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-24.6925583, 19.7640648, -24.6925583, 19.7640648, -44.4566231, 44.4566231)
1: (-22.1655941, 17.7110901, -22.1655941, 17.7110901, -39.8766861, 39.8766861)
2: (-28.0165939, 17.5855999, -28.0165939, 17.5855999, -45.6021957, 45.6021957)
3: (-30.1115532, 15.0882940, -30.1115532, 15.0882940, -45.1998444, 45.1998482)
4: (-28.4748173, 20.2006111, -28.4748173, 20.2006111, -48.6754303, 48.6754303)
5: (-24.4868717, 19.1075554, -24.4868717, 19.1075554, -43.5944290, 43.5944290)
6: (-22.5470924, 22.3522205, -22.5470924, 22.3522205, -44.8993111, 44.8993149)
7: (-24.8867416, 23.5568867, -24.8867416, 23.5568867, -48.4436264, 48.4436264)
8: (-34.8141861, 16.7162991, -34.8141861, 16.7162991, -51.5304832, 51.5304871)
9: (-21.9554176, 22.3245659, -21.9554176, 22.3245659, -44.2799835, 44.2799835)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 11.94 = 12.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1705002, upper bound: 43.1706408
time: 6.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437
time: 4.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.21 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.21
Output dim: 8, lower bound: -43.1705002, upper bound: 43.1706408
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.21
Output dim: 8, lower bound: -43.1707437, upper bound: 43.1707437

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -24.3706722, 19.5024204, -23.7293777, 18.9999046, -43.3705750, 43.2317963
1: -21.8586521, 17.4819832, -21.3050957, 17.0339031, -38.8925552, 38.7870789
2: -27.6241531, 17.3405914, -26.9258842, 16.9057484, -44.5298996, 44.2664757
3: -29.6621819, 14.9008579, -28.9340725, 14.5140362, -44.1762161, 43.8349304
4: -28.0579205, 19.9350491, -27.3763237, 19.4126072, -47.4705276, 47.3113708
5: -24.1463985, 18.8390179, -23.5370712, 18.3707199, -42.5171204, 42.3760872
6: -22.2339973, 22.0551300, -21.6582737, 21.4876881, -43.7216873, 43.7134018
7: -24.5400143, 23.1987343, -23.9124851, 22.6648006, -47.2048111, 47.1112175
8: -34.2982140, 16.5464706, -33.5120163, 16.0368385, -50.3350487, 50.0584869
9: -21.6667118, 22.0264530, -21.0936375, 21.4534988, -43.1202087, 43.1200905

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1681183, upper bound: 43.1684686
time: 6.40 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1680409, upper bound: 43.1683212
time: 5.84 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -24.2108173, 19.3813133, -24.6299152, 19.7143040, -43.9251137, 44.0112228
1: -21.7347298, 17.3723087, -22.1095390, 17.6670246, -39.4017563, 39.4818420
2: -27.4703083, 17.2466469, -27.9456024, 17.5415535, -45.0118561, 45.1922493
3: -29.5238304, 14.8024702, -30.0350666, 15.0510893, -44.5749130, 44.8375359
4: -27.9244385, 19.8066959, -28.4033089, 20.1493416, -48.0737801, 48.2100067
5: -24.0113335, 18.7387772, -24.4249897, 19.0596142, -43.0709457, 43.1637650
6: -22.1033802, 21.9187889, -22.4893761, 22.2958450, -44.3992233, 44.4081650
7: -24.3987465, 23.1103039, -24.8233261, 23.4988098, -47.8975563, 47.9336281
8: -34.1606636, 16.3779488, -34.7292480, 16.6722507, -50.8329163, 51.1071930
9: -21.5237293, 21.8889427, -21.8992653, 22.2679310, -43.7916603, 43.7882080

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1687986, upper bound: 43.1688728
time: 7.08 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1687308, upper bound: 43.1687308
time: 23.27 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.18 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 8, lower bound: -43.1681183, upper bound: 43.1684686
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 8, lower bound: -43.1680409, upper bound: 43.1683212
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 8, lower bound: -43.1687986, upper bound: 43.1688728
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.18
Output dim: 8, lower bound: -43.1687308, upper bound: 43.1687308

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -24.0274734, 19.2259598, -22.4434185, 17.9652138, -41.9926872, 41.6693802
1: -21.5484715, 17.2337952, -20.1415367, 16.1099892, -37.6584511, 37.3753319
2: -27.2355461, 17.0914726, -25.4634438, 15.9779587, -43.2135010, 42.5549164
3: -29.2451973, 14.6875143, -27.3715172, 13.7165775, -42.9617767, 42.0590248
4: -27.6738758, 19.6465912, -25.9315643, 18.3364410, -46.0103149, 45.5781517
5: -23.8128948, 18.5773220, -22.2795792, 17.3917427, -41.2046356, 40.8569031
6: -21.9131813, 21.7422218, -20.4598961, 20.3185863, -42.2317657, 42.2021179
7: -24.1890182, 22.8952007, -22.6008167, 21.5184364, -45.7074432, 45.4960136
8: -33.8406487, 16.2714901, -31.7780952, 15.0259771, -48.8666267, 48.0495834
9: -21.3593922, 21.7078362, -19.9446354, 20.2580185, -41.6174088, 41.6524734

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1638498, upper bound: 43.1631056
time: 7.55 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1621450, upper bound: 43.1624340
time: 6.98 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -23.4692764, 18.7772446, -26.2538834, 20.9394150, -44.4086838, 45.0311279
1: -21.0457382, 16.8315544, -23.5655441, 18.7577667, -39.8035011, 40.3970909
2: -26.6018257, 16.6888313, -29.8010178, 18.5634918, -45.1653175, 46.4898453
3: -28.5657501, 14.3411970, -32.0147247, 15.8700066, -44.4357567, 46.3559227
4: -27.0464745, 19.1799641, -30.3568516, 21.3696156, -48.4160919, 49.5368118
5: -23.2684937, 18.1534843, -26.0840454, 20.2800159, -43.5485077, 44.2375298
6: -21.3925285, 21.2332096, -23.8919868, 23.7084827, -45.1010132, 45.1251984
7: -23.6194878, 22.4003086, -26.4481239, 25.1325226, -48.7520103, 48.8484344
8: -33.0904808, 15.8325100, -37.0121346, 17.3814564, -50.4719391, 52.8446426
9: -20.8608608, 21.1862316, -23.3076611, 23.5980835, -44.4589424, 44.4938927

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1635388, upper bound: 43.1626125
time: 6.76 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1619750, upper bound: 43.1620769
time: 20.16 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -23.8581047, 19.0980396, -23.2867889, 18.6356487, -42.4937515, 42.3848267
1: -21.4170055, 17.1183109, -20.8980331, 16.7013950, -38.1184006, 38.0163422
2: -27.0702362, 16.9915276, -26.4206753, 16.5733204, -43.6435547, 43.4121971
3: -29.0950127, 14.5852718, -28.4045334, 14.2210846, -43.3160973, 42.9898071
4: -27.5283089, 19.5113564, -26.8937950, 19.0252609, -46.5535660, 46.4051514
5: -23.6685734, 18.4695988, -23.1144886, 18.0344944, -41.7030678, 41.5840874
6: -21.7743282, 21.5979595, -21.2378349, 21.0762005, -42.8505287, 42.8357925
7: -24.0376511, 22.7972603, -23.4515667, 22.3013382, -46.3389893, 46.2488174
8: -33.6883965, 16.0996914, -32.9226608, 15.6192646, -49.3076553, 49.0223503
9: -21.2082195, 21.5628204, -20.6984024, 21.0237617, -42.2319794, 42.2612228

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1641638, upper bound: 43.1641551
time: 6.59 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1640451, upper bound: 43.1642098
time: 6.71 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -23.2807846, 18.6340904, -27.1942177, 21.6846485, -44.9654312, 45.8283081
1: -20.8955154, 16.7026577, -24.4111290, 19.4196281, -40.3151398, 41.1137848
2: -26.4127350, 16.5755234, -30.8662395, 19.2280502, -45.6407776, 47.4417648
3: -28.3894768, 14.2289448, -33.1678391, 16.4308186, -44.8202972, 47.3967819
4: -26.8767815, 19.0287514, -31.4320526, 22.1347961, -49.0115776, 50.4608040
5: -23.1044922, 18.0298748, -27.0135536, 20.9974918, -44.1019821, 45.0434265
6: -21.2365646, 21.0722923, -24.7594185, 24.5535793, -45.7901459, 45.8317108
7: -23.4471989, 22.2817726, -27.3982124, 26.0073681, -49.4545631, 49.6799850
8: -32.9089661, 15.6511354, -38.2802734, 18.0389023, -50.9478645, 53.9314079
9: -20.6923428, 21.0251579, -24.1517849, 24.4449902, -45.1373329, 45.1769371

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 50

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1637315, upper bound: 43.1627296
time: 7.38 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1621556, upper bound: 43.1621556
time: 8.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 16.73 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1638498, upper bound: 43.1631056
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1621450, upper bound: 43.1624340
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1635388, upper bound: 43.1626125
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1619750, upper bound: 43.1620769
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1641638, upper bound: 43.1641551
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1640451, upper bound: 43.1642098
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1637315, upper bound: 43.1627296
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.73
Output dim: 8, lower bound: -43.1621556, upper bound: 43.1621556

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -23.0166969, 18.4229088, -22.3311176, 17.8756771, -40.8923721, 40.7540283
1: -20.6393051, 16.5213699, -20.0402870, 16.0308933, -36.6701889, 36.5616570
2: -26.0778675, 16.3866272, -25.3347206, 15.8994627, -41.9773293, 41.7213364
3: -27.9982262, 14.0914955, -27.2325745, 13.6502914, -41.6485176, 41.3240700
4: -26.5054054, 18.8207893, -25.8016224, 18.2445984, -44.7500038, 44.6224136
5: -22.8055935, 17.8061886, -22.1673889, 17.3057728, -40.1113586, 39.9735794
6: -20.9821720, 20.8259583, -20.3563099, 20.2168274, -41.1989975, 41.1822624
7: -23.1558418, 21.9507389, -22.4859543, 21.4131775, -44.5690193, 44.4366913
8: -32.4507065, 15.5854120, -31.6229496, 14.9501095, -47.4008179, 47.2083626
9: -20.4549904, 20.7888737, -19.8441353, 20.1557331, -40.6107140, 40.6330109

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1587060, upper bound: 43.1581355
time: 8.50 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1585692, upper bound: 43.1578447
time: 6.22 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -29.7193317, 23.7543755, -21.6301098, 17.3174362, -47.0367661, 45.3844833
1: -26.7321701, 21.2260361, -19.4073029, 15.5375290, -42.2696991, 40.6333237
2: -33.7635612, 21.0684204, -24.5320721, 15.4089994, -49.1725616, 45.6004944
3: -36.1951561, 18.0335808, -26.3678398, 13.2361298, -49.4312859, 44.4014091
4: -34.2366905, 24.2265396, -24.9904232, 17.6698704, -51.9065628, 49.2169571
5: -29.4471245, 22.9202232, -21.4668045, 16.7697449, -46.2168617, 44.3870163
6: -27.1004105, 26.8186226, -19.7100830, 19.5798626, -46.6802750, 46.5286980
7: -29.9742260, 28.2555981, -21.7682114, 20.7562103, -50.7304382, 50.0238113
8: -41.6308746, 20.0867081, -30.6566315, 14.4753723, -56.1062431, 50.7433395
9: -26.4267998, 26.8178082, -19.2152061, 19.5178223, -45.9446220, 46.0330124

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1568953, upper bound: 43.1573334
time: 8.24 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1567754, upper bound: 43.1571593
time: 7.21 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -22.4738045, 17.9855995, -26.1413803, 20.8500710, -43.3238678, 44.1269798
1: -20.1489410, 16.1305790, -23.4642181, 18.6785488, -38.8274841, 39.5947952
2: -25.4607315, 15.9941998, -29.6723118, 18.4850159, -43.9457436, 45.6665115
3: -27.3360424, 13.7547235, -31.8757477, 15.8035908, -43.1396332, 45.6304703
4: -25.8956299, 18.3678799, -30.2271194, 21.2777100, -47.1733398, 48.5950012
5: -22.2747192, 17.3931942, -25.9717522, 20.1941795, -42.4688988, 43.3649445
6: -20.4757290, 20.3309135, -23.7883873, 23.6066399, -44.0823631, 44.1193008
7: -22.6025696, 21.4675064, -26.3332062, 25.0272579, -47.6298294, 47.8007126
8: -31.7193565, 15.1603193, -36.8576431, 17.3051434, -49.0244980, 52.0179596
9: -19.9706726, 20.2803593, -23.2070560, 23.4960251, -43.4666977, 43.4874153

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1583765, upper bound: 43.1576914
time: 7.61 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1581200, upper bound: 43.1573022
time: 6.85 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -29.0856514, 23.2412491, -25.4289837, 20.2833443, -49.3689957, 48.6702347
1: -26.1672516, 20.7624626, -22.8232231, 18.1771851, -44.3444366, 43.5856857
2: -33.0477180, 20.6027565, -28.8580952, 17.9862194, -51.0339279, 49.4608536
3: -35.4258995, 17.6480961, -30.9976692, 15.3823566, -50.8082581, 48.6457672
4: -33.5265732, 23.6888657, -29.4064198, 20.6946602, -54.2212334, 53.0952835
5: -28.8367386, 22.4262333, -25.2613945, 19.6503983, -48.4871368, 47.6876259
6: -26.5031662, 26.2482586, -23.1313934, 22.9610634, -49.4642258, 49.3796539
7: -29.3153381, 27.6850815, -25.6049995, 24.3621254, -53.6774635, 53.2900810
8: -40.7987404, 19.5758820, -35.8796997, 16.8193722, -57.6181068, 55.4555817
9: -25.8558006, 26.2390919, -22.5691872, 22.8491077, -48.7048988, 48.8082809

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1567514, upper bound: 43.1570541
time: 6.17 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1565995, upper bound: 43.1567698
time: 6.10 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -21.8107224, 17.4580841, -22.6156349, 18.0982246, -39.9089470, 40.0737190
1: -19.6237011, 15.6917381, -20.3104324, 16.2345753, -35.8582726, 36.0021706
2: -24.7673454, 15.5326052, -25.6675911, 16.0956974, -40.8630371, 41.2001953
3: -26.6479301, 13.3193970, -27.6043167, 13.8069324, -40.4548607, 40.9237137
4: -25.2511616, 17.8157063, -26.1498985, 18.4678497, -43.7190094, 43.9656067
5: -21.6812630, 16.9218407, -22.4616966, 17.5285378, -39.2098007, 39.3835335
6: -19.8263588, 19.7629681, -20.5993176, 20.4757118, -40.3020630, 40.3622818
7: -21.9688492, 20.9880142, -22.7770596, 21.7092152, -43.6780624, 43.7650757
8: -31.0022793, 14.4879150, -32.0442657, 15.0915318, -46.0938072, 46.5321693
9: -19.3596573, 19.6809883, -20.0937881, 20.4067116, -39.7663689, 39.7747765

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577456, upper bound: 43.1589220
time: 5.93 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
time: 5.82 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.0404472, 20.0120049, -22.1884670, 17.7573490, -42.7977982, 42.2004700
1: -22.5797195, 17.9686184, -19.9391575, 15.9402485, -38.5199661, 37.9077759
2: -28.4670925, 17.7540874, -25.1885605, 15.7885637, -44.2556572, 42.9426498
3: -30.6687870, 15.1888447, -27.0988503, 13.5413494, -44.2101364, 42.2876930
4: -29.0285683, 20.4057255, -25.6771030, 18.1155663, -47.1441345, 46.0828247
5: -24.9263039, 19.3847256, -22.0480652, 17.2077980, -42.1340942, 41.4327927
6: -22.7554379, 22.6729450, -20.1921520, 20.0951920, -42.8506317, 42.8650970
7: -25.2554893, 24.0742607, -22.3433514, 21.3348999, -46.5903778, 46.4176025
8: -35.4793243, 16.5323257, -31.4918404, 14.7521801, -50.2315063, 48.0241661
9: -22.2196407, 22.5628891, -19.7063713, 20.0137653, -42.2333984, 42.2692604

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 92

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
time: 7.90 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
time: 6.39 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -22.3649044, 17.9046574, -27.0820293, 21.5957279, -43.9606323, 44.9866867
1: -20.0698032, 16.0568008, -24.3104362, 19.3404942, -39.4102974, 40.3672371
2: -25.3626633, 15.9357929, -30.7382355, 19.1495304, -44.5121880, 46.6740265
3: -27.2571144, 13.6871891, -33.0297165, 16.3644104, -43.6215248, 46.7169037
4: -25.8176422, 18.2792492, -31.3030739, 22.0433445, -47.8609848, 49.5823212
5: -22.1898022, 17.3288345, -26.9020863, 20.9115620, -43.1013641, 44.2309189
6: -20.3925343, 20.2419491, -24.6558762, 24.4521084, -44.8446426, 44.8978271
7: -22.5107479, 21.4234581, -27.2835636, 25.9029083, -48.4136581, 48.7070236
8: -31.6445427, 15.0310926, -38.1271133, 17.9621181, -49.6066589, 53.1582069
9: -19.8725357, 20.1904659, -24.0511398, 24.3434982, -44.2160301, 44.2416000

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1587542, upper bound: 43.1579115
time: 5.78 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1586416, upper bound: 43.1576015
time: 7.06 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -27.9443855, 22.3447609, -26.3535748, 21.0176640, -48.9620476, 48.6983337
1: -25.1368427, 19.9594421, -23.6564178, 18.8270035, -43.9638443, 43.6158562
2: -31.7523346, 19.8244228, -29.9073887, 18.6384430, -50.3907776, 49.7318115
3: -34.0747910, 16.9819317, -32.1344528, 15.9329300, -50.0077209, 49.1163864
4: -32.2333450, 22.7744884, -30.4659004, 21.4487400, -53.6820831, 53.2403870
5: -27.7168789, 21.5684204, -26.1780128, 20.3541985, -48.0710678, 47.7464256
6: -25.4803562, 25.2302933, -23.9835815, 23.7920818, -49.2724380, 49.2138748
7: -28.1641350, 26.6416416, -26.5392857, 25.2251186, -53.3892517, 53.1809273
8: -39.2703972, 18.8056641, -37.1325912, 17.4603386, -56.7307358, 55.9382553
9: -24.8422623, 25.2171440, -23.3975105, 23.6833572, -48.5256195, 48.6146507

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 216

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1571258, upper bound: 43.1572659
time: 6.00 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1570256, upper bound: 43.1570256
time: 6.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 12.97 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1587060, upper bound: 43.1581355
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1585692, upper bound: 43.1578447
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1568953, upper bound: 43.1573334
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1567754, upper bound: 43.1571593
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1583765, upper bound: 43.1576914
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1581200, upper bound: 43.1573022
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1567514, upper bound: 43.1570541
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1565995, upper bound: 43.1567698
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1577456, upper bound: 43.1589220
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1587542, upper bound: 43.1579115
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1586416, upper bound: 43.1576015
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1571258, upper bound: 43.1572659
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.97
Output dim: 8, lower bound: -43.1570256, upper bound: 43.1570256

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -22.3169975, 17.8605328, -20.3841782, 16.3228855, -38.6398849, 38.2447128
1: -20.0269089, 16.0306282, -18.3283730, 14.6772566, -34.7041588, 34.3589973
2: -25.2892475, 15.8863840, -23.1450958, 14.5153866, -39.8046341, 39.0314789
3: -27.1594620, 13.6564550, -24.9023781, 12.4489145, -39.6083755, 38.5588303
4: -25.7276840, 18.2401600, -23.6283379, 16.6355133, -42.3631973, 41.8684998
5: -22.1280937, 17.2756977, -20.2714119, 15.8366966, -37.9647903, 37.5471115
6: -20.3164902, 20.1969223, -18.5031204, 18.4741344, -38.7906113, 38.7000427
7: -22.4498482, 21.3343124, -20.5076866, 19.6823425, -42.1321793, 41.8419991
8: -31.5306454, 15.0288601, -29.0695992, 13.4439240, -44.9745636, 44.0984573
9: -19.8234673, 20.1410465, -18.0882988, 18.3783207, -38.2017746, 38.2293396

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 216

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
time: 6.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1378680, upper bound: 43.1381622
time: 7.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -21.8758907, 17.5039005, -23.4327221, 18.7319031, -40.6077881, 40.9366226
1: -19.6435127, 15.7230968, -21.1363316, 16.8326645, -36.4761772, 36.8594284
2: -24.7908268, 15.5678511, -26.6434669, 16.6088963, -41.3997231, 42.2113190
3: -26.6363258, 13.3798409, -28.7116852, 14.2134190, -40.8497467, 42.0915260
4: -25.2399521, 17.8734360, -27.2091789, 19.0822544, -44.3222046, 45.0826149
5: -21.7022305, 16.9421654, -23.3472023, 18.1645470, -39.8667679, 40.2893677
6: -19.8945198, 19.8017559, -21.2644615, 21.2288094, -41.1233292, 41.0662155
7: -22.0012169, 20.9485722, -23.6152458, 22.6135998, -44.6148148, 44.5638199
8: -30.9528065, 14.6686840, -33.3240204, 15.3498402, -46.3026428, 47.9927063
9: -19.4216499, 19.7281837, -20.7852230, 21.0994339, -40.5210800, 40.5134048

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1434155, upper bound: 43.1420335
time: 8.26 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1376026, upper bound: 43.1378050
time: 6.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -28.9991226, 23.1766033, -19.7339573, 15.8070164, -44.8061371, 42.9105568
1: -26.1037750, 20.7163696, -17.7355175, 14.2161045, -40.3198738, 38.4518890
2: -32.9571571, 20.5505810, -22.3953686, 14.0610046, -47.0181618, 42.9459496
3: -35.3330536, 17.5885735, -24.0975838, 12.0625973, -47.3956528, 41.6861572
4: -33.4344025, 23.6209812, -22.8682518, 16.1034164, -49.5378189, 46.4892349
5: -28.7532806, 22.3700504, -19.6212692, 15.3362827, -44.0895615, 41.9913177
6: -26.4134827, 26.1744766, -17.9026260, 17.8808994, -44.2943726, 44.0771027
7: -29.2463264, 27.6234932, -19.8336678, 19.0664406, -48.3127670, 47.4571609
8: -40.6896744, 19.5064812, -28.1663380, 13.0126324, -53.7023010, 47.6728210
9: -25.7749023, 26.1528854, -17.5053406, 17.7861862, -43.5610847, 43.6582184

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1425270, upper bound: 43.1418076
time: 8.96 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1343339, upper bound: 43.1359994
time: 21.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -28.5200386, 22.7891617, -22.7457905, 18.1865540, -46.7065926, 45.5349503
1: -25.6881828, 20.3784237, -20.5171432, 16.3503380, -42.0385208, 40.8955650
2: -32.4198990, 20.2016048, -25.8569126, 16.1291561, -48.5490570, 46.0585175
3: -34.7637291, 17.2879791, -27.8628922, 13.8107882, -48.5745163, 45.1508713
4: -32.9036064, 23.2180977, -26.4138432, 18.5216484, -51.4252548, 49.6319427
5: -28.2930984, 22.0044937, -22.6610374, 17.6394863, -45.9325867, 44.6655312
6: -25.9522171, 25.7461300, -20.6313820, 20.6074867, -46.5597000, 46.3774986
7: -28.7591000, 27.2065048, -22.9091454, 21.9694061, -50.7285080, 50.1156502
8: -40.0657578, 19.1100826, -32.3758049, 14.8907099, -54.9564667, 51.4858856
9: -25.3376102, 25.7051449, -20.1742649, 20.4777622, -45.8153648, 45.8794098

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1421543, upper bound: 43.1413049
time: 8.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1341723, upper bound: 43.1357489
time: 7.22 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -21.7958412, 17.4394684, -24.1253586, 19.2420921, -41.0379333, 41.5648270
1: -19.5551434, 15.6554518, -21.7005310, 17.2795792, -36.8347168, 37.3559837
2: -24.6955929, 15.5096703, -27.4160900, 17.0530434, -41.7486267, 42.9257584
3: -26.5236702, 13.3317862, -29.4712086, 14.5597439, -41.0834122, 42.8029938
4: -25.1417885, 17.8061275, -27.9915352, 19.6146431, -44.7564316, 45.7976608
5: -21.6169014, 16.8782578, -24.0095882, 18.6792908, -40.2961922, 40.8878479
6: -19.8299046, 19.7216530, -21.8713455, 21.8066940, -41.6365967, 41.5929985
7: -21.9173546, 20.8683548, -24.3057594, 23.2413406, -45.1586914, 45.1741142
8: -30.8247433, 14.6236200, -34.2338943, 15.7437305, -46.5684700, 48.8575134
9: -19.3577938, 19.6525574, -21.3932610, 21.6604080, -41.0182037, 41.0458183

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1427077, upper bound: 43.1413273
time: 7.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1372735, upper bound: 43.1372832
time: 7.15 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -21.3600750, 17.0900135, -27.0307732, 21.5381966, -42.8982697, 44.1207886
1: -19.1763878, 15.3527975, -24.3743401, 19.3348789, -38.5112686, 39.7271385
2: -24.2049255, 15.1963387, -30.7471828, 19.0490608, -43.2539749, 45.9435196
3: -26.0071373, 13.0599041, -33.1030426, 16.2420673, -42.2492065, 46.1629486
4: -24.6591816, 17.4437008, -31.4012032, 21.9439926, -46.6031647, 48.8449020
5: -21.1954422, 16.5507832, -26.9374962, 20.8993435, -42.0947800, 43.4882812
6: -19.4135933, 19.3326721, -24.5042477, 24.4293079, -43.8429031, 43.8369179
7: -21.4726944, 20.4869003, -27.2594223, 26.0355949, -47.5082893, 47.7463226
8: -30.2601585, 14.2740879, -38.2786560, 17.5610981, -47.8212433, 52.5527420
9: -18.9601250, 19.2496719, -23.9632912, 24.2564220, -43.2165451, 43.2129593

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 210

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1420324, upper bound: 43.1404044
time: 13.99 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1368476, upper bound: 43.1367260
time: 7.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -28.4017792, 22.6917801, -23.4391308, 18.6964054, -47.0981827, 46.1309128
1: -25.5681000, 20.2822857, -21.0818481, 16.7960377, -42.3641357, 41.3641357
2: -32.2794495, 20.1122131, -26.6298447, 16.5730305, -48.8524780, 46.7420578
3: -34.6057930, 17.2232819, -28.6226120, 14.1559124, -48.7617035, 45.8458939
4: -32.7666817, 23.1198578, -27.1958847, 19.0534210, -51.8201027, 50.3157349
5: -28.1739635, 21.9091892, -23.3252678, 18.1538982, -46.3278580, 45.2344551
6: -25.8496265, 25.6351662, -21.2387161, 21.1845589, -47.0341873, 46.8738823
7: -28.6274643, 27.0838280, -23.6002445, 22.5979500, -51.2254066, 50.6840706
8: -39.8980560, 19.0331230, -33.2873535, 15.2814550, -55.1795120, 52.3204765
9: -25.2398548, 25.6046677, -20.7794189, 21.0372200, -46.2770653, 46.3840790

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 52

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1416049, upper bound: 43.1407731
time: 8.49 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1340917, upper bound: 43.1354146
time: 20.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -27.9290390, 22.3094101, -26.3333244, 20.9832726, -48.9123039, 48.6427345
1: -25.1565628, 19.9507065, -23.7458420, 18.8447609, -44.0013237, 43.6965446
2: -31.7471256, 19.7691383, -29.9491272, 18.5607662, -50.3078880, 49.7182579
3: -34.0428238, 16.9251957, -32.2431068, 15.8305244, -49.8733482, 49.1682968
4: -32.2432442, 22.7248764, -30.5955238, 21.3737946, -53.6170349, 53.3204002
5: -27.7181053, 21.5514145, -26.2416916, 20.3659039, -48.0840034, 47.7931061
6: -25.3948174, 25.2115002, -23.8606510, 23.7977352, -49.1925507, 49.0721474
7: -28.1485901, 26.6721764, -26.5444374, 25.3830585, -53.5316429, 53.2166138
8: -39.2772369, 18.6460743, -37.3183327, 17.0864391, -56.3636780, 55.9644089
9: -24.8099785, 25.1608829, -23.3388252, 23.6231518, -48.4331284, 48.4996948

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 161

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1409992, upper bound: 43.1398193
time: 8.47 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1338473, upper bound: 43.1350177
time: 6.95 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -21.7038612, 17.3731079, -21.7182274, 17.3838654, -39.0877190, 39.0913353
1: -19.5273743, 15.6165009, -19.4996376, 15.6013842, -35.1287575, 35.1161346
2: -24.6448936, 15.4580469, -24.6385574, 15.4691200, -40.1140137, 40.0966034
3: -26.5152397, 13.2564468, -26.4934959, 13.2756948, -39.7909355, 39.7499428
4: -25.1272545, 17.7283669, -25.1090870, 17.7332897, -42.8605423, 42.8374557
5: -21.5744591, 16.8401279, -21.5647202, 16.8413258, -38.4157829, 38.4048462
6: -19.7277431, 19.6662560, -19.7728977, 19.6606407, -39.3883820, 39.4391479
7: -21.8589344, 20.8877678, -21.8563232, 20.8656693, -42.7246017, 42.7440872
8: -30.8547745, 14.4161091, -30.8046761, 14.4887667, -45.3435402, 45.2207870
9: -19.2638569, 19.5840149, -19.2886238, 19.5903473, -38.8542023, 38.8726349

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
time: 5.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
time: 5.97 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -21.0068817, 16.8179626, -27.1860657, 21.7152252, -42.7221031, 44.0040283
1: -18.8960724, 15.1230173, -24.4577141, 19.4102249, -38.3062973, 39.5807304
2: -23.8438721, 14.9697189, -30.8860817, 19.2588673, -43.1027374, 45.8558006
3: -25.6524258, 12.8441973, -33.1571007, 16.4992638, -42.1516838, 46.0012970
4: -24.3171959, 17.1558609, -31.3926735, 22.1173325, -46.4345284, 48.5485344
5: -20.8778877, 16.3058319, -26.9891968, 20.9838791, -41.8617668, 43.2950287
6: -19.0856133, 19.0315094, -24.7466927, 24.5343742, -43.6199875, 43.7782021
7: -21.1400852, 20.2319489, -27.3989487, 25.9918861, -47.1319656, 47.6308899
8: -29.8894424, 13.9451599, -38.2470932, 18.1375275, -48.0269661, 52.1922379
9: -18.6371956, 18.9486618, -24.1517353, 24.4822845, -43.1194801, 43.1003914

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 210

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1449620, upper bound: 43.1436103
time: 8.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1413022, upper bound: 43.1415268
time: 7.37 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -24.9341640, 19.9274826, -21.2979813, 17.0487576, -41.9829216, 41.2254524
1: -22.4838905, 17.8939018, -19.1336842, 15.3107500, -37.7946396, 37.0275841
2: -28.3454323, 17.6798630, -24.1668587, 15.1669312, -43.5123558, 41.8467102
3: -30.5372505, 15.1262417, -25.9951591, 13.0145378, -43.5517883, 41.1213951
4: -28.9056511, 20.3189201, -24.6424789, 17.3853550, -46.2910042, 44.9613914
5: -24.8202057, 19.3034248, -21.1586151, 16.5252228, -41.3454285, 40.4620361
6: -22.6574230, 22.5767212, -19.3718967, 19.2862377, -41.9436607, 41.9486160
7: -25.1466045, 23.9747276, -21.4271183, 20.4969845, -45.6435890, 45.4018478
8: -35.3328400, 16.4606743, -30.2603302, 14.1538296, -49.4866714, 46.7210045
9: -22.1244774, 22.4664288, -18.9067459, 19.2028351, -41.3273125, 41.3731766

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
time: 6.91 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
time: 5.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -24.2138195, 19.3546333, -26.6997089, 21.3254318, -45.5392532, 46.0543442
1: -21.8347931, 17.3879509, -24.0449448, 19.0699654, -40.9047585, 41.4328957
2: -27.5215034, 17.1759491, -30.3420677, 18.9088898, -46.4303932, 47.5180130
3: -29.6486187, 14.7020111, -32.5881577, 16.1946564, -45.8432732, 47.2901688
4: -28.0732269, 19.7303581, -30.8643646, 21.7137661, -49.7869911, 50.5947189
5: -24.1014748, 18.7528439, -26.5209045, 20.6223755, -44.7238464, 45.2737503
6: -21.9934044, 21.9241676, -24.2843971, 24.1059074, -46.0993118, 46.2085609
7: -24.4084358, 23.3007526, -26.9155331, 25.5737267, -49.9821548, 50.2162743
8: -34.3411064, 15.9729719, -37.6258392, 17.7476177, -52.0887222, 53.5988045
9: -21.4796066, 21.8124199, -23.7152672, 24.0362663, -45.5158730, 45.5276871

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
time: 7.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574636
time: 6.20 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -21.7195969, 17.3864822, -25.0382080, 19.9666920, -41.6862869, 42.4246902
1: -19.5034771, 15.6070557, -22.5253639, 17.9201698, -37.4236450, 38.1324158
2: -24.6378365, 15.4754276, -28.4537754, 17.6953850, -42.3332176, 43.9291992
3: -26.4865685, 13.2872143, -30.5958500, 15.1019583, -41.5885277, 43.8830643
4: -25.1001129, 17.7426262, -29.0424080, 20.3569641, -45.4570770, 46.7850304
5: -21.5622501, 16.8408623, -24.9149742, 19.3758545, -40.9381027, 41.7558250
6: -19.7775402, 19.6628361, -22.7123909, 22.6275234, -42.4050598, 42.3752289
7: -21.8593941, 20.8528175, -25.2303772, 24.0958557, -45.9552498, 46.0831947
8: -30.7973328, 14.5224876, -35.4749031, 16.3717785, -47.1691132, 49.9973907
9: -19.2886696, 19.5960045, -22.2112560, 22.4839439, -41.7726135, 41.8072548

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 210

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1457409, upper bound: 43.1436490
time: 7.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1427877, upper bound: 43.1417305
time: 8.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -21.2774258, 17.0338707, -28.1179867, 22.4046669, -43.6820908, 45.1518555
1: -19.1183128, 15.3009081, -25.3549309, 20.0962009, -39.2145157, 40.6558380
2: -24.1410046, 15.1577320, -31.9842739, 19.8172855, -43.9582863, 47.1419945
3: -25.9615479, 13.0123472, -34.4375916, 16.8887711, -42.8503189, 47.4499359
4: -24.6095200, 17.3766994, -32.6495323, 22.8266106, -47.4361305, 50.0262222
5: -21.1348934, 16.5083351, -28.0123405, 21.7274818, -42.8623695, 44.5206757
6: -19.3566265, 19.2684631, -25.5089626, 25.4051495, -44.7617760, 44.7774200
7: -21.4080811, 20.4647503, -28.3622723, 27.0502148, -48.4582901, 48.8270226
8: -30.2238121, 14.1715326, -39.7445335, 18.3234577, -48.5472603, 53.9160614
9: -18.8868256, 19.1889114, -24.9398117, 25.2347279, -44.1215515, 44.1287155

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1450311, upper bound: 43.1423399
time: 7.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1422602, upper bound: 43.1407977
time: 6.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -27.2799320, 21.8099499, -24.3366165, 19.4080429, -46.6879730, 46.1465683
1: -24.5545216, 19.4934082, -21.8922882, 17.4264603, -41.9809723, 41.3856964
2: -31.0054207, 19.3472977, -27.6507912, 17.2043571, -48.2097778, 46.9980888
3: -33.2779045, 16.5700607, -29.7290955, 14.6876383, -47.9655418, 46.2991562
4: -31.4943810, 22.2204762, -28.2314034, 19.7834511, -51.2778320, 50.4518738
5: -27.0735741, 21.0658665, -24.2151489, 18.8393478, -45.9129219, 45.2810020
6: -24.8458023, 24.6351814, -22.0648117, 21.9912262, -46.8370285, 46.6999931
7: -27.4951591, 26.0578537, -24.5118542, 23.4391556, -50.9343147, 50.5697060
8: -38.3936996, 18.2769184, -34.5103645, 15.8949738, -54.2886581, 52.7872734
9: -24.2430763, 24.6009483, -21.5826473, 21.8466644, -46.0897408, 46.1835938

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1444121, upper bound: 43.1429397
time: 6.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1403407, upper bound: 43.1407971
time: 6.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -26.8044033, 21.4258156, -27.4044991, 21.8371391, -48.6415405, 48.8303146
1: -24.1394386, 19.1599255, -24.7123489, 19.5944328, -43.7338638, 43.8722725
2: -30.4694462, 19.0020618, -31.1687622, 19.3173122, -49.7867584, 50.1708183
3: -32.7104225, 16.2710152, -33.5580597, 16.4670315, -49.1774521, 49.8290749
4: -30.9669762, 21.8229828, -31.8266106, 22.2438049, -53.2107811, 53.6495934
5: -26.6149635, 20.7054348, -27.3014183, 21.1820145, -47.7969780, 48.0068512
6: -24.3904247, 24.2090874, -24.8505726, 24.7587662, -49.1491852, 49.0596504
7: -27.0131378, 25.6428547, -27.6320801, 26.3836746, -53.3968124, 53.2749329
8: -37.7680969, 17.8897305, -38.7647247, 17.8362865, -55.6043854, 56.6544495
9: -23.8110466, 24.1565228, -24.3009472, 24.5880337, -48.3990746, 48.4574699

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1437519, upper bound: 43.1417729
time: 7.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1400550, upper bound: 43.1400550
time: 5.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 12.99 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1378680, upper bound: 43.1381622
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1434155, upper bound: 43.1420335
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1376026, upper bound: 43.1378050
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1425270, upper bound: 43.1418076
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1343339, upper bound: 43.1359994
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1421543, upper bound: 43.1413049
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1341723, upper bound: 43.1357489
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1427077, upper bound: 43.1413273
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1372735, upper bound: 43.1372832
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1420324, upper bound: 43.1404044
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1368476, upper bound: 43.1367260
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1416049, upper bound: 43.1407731
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1340917, upper bound: 43.1354146
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1409992, upper bound: 43.1398193
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1338473, upper bound: 43.1350177
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1574921, upper bound: 43.1575417
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1449620, upper bound: 43.1436103
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1413022, upper bound: 43.1415268
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1577081, upper bound: 43.1588368
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574639
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1572100, upper bound: 43.1574636
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1457409, upper bound: 43.1436490
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1427877, upper bound: 43.1417305
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1450311, upper bound: 43.1423399
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1422602, upper bound: 43.1407977
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1444121, upper bound: 43.1429397
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1403407, upper bound: 43.1407971
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1437519, upper bound: 43.1417729
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.99
Output dim: 8, lower bound: -43.1400550, upper bound: 43.1400550

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -21.6825409, 17.3576088, -20.3826065, 16.3216476, -38.0041885, 37.7402153
1: -19.4594078, 15.5867901, -18.3269596, 14.6761532, -34.1355591, 33.9137421
2: -24.5701447, 15.4438124, -23.1433144, 14.5142899, -39.0844345, 38.5871201
3: -26.3900719, 13.2765951, -24.9004669, 12.4479694, -38.8380356, 38.1770554
4: -25.0076675, 17.7255077, -23.6265450, 16.6342411, -41.6419067, 41.3520432
5: -21.5060825, 16.7976513, -20.2698669, 15.8355122, -37.3415947, 37.0675125
6: -19.7360935, 19.6281929, -18.5016804, 18.4727192, -38.2088089, 38.1298676
7: -21.8061352, 20.7579479, -20.5060749, 19.6809006, -41.4870338, 41.2640228
8: -30.6779327, 14.5811291, -29.0674839, 13.4428358, -44.1207657, 43.6486130
9: -19.2596035, 19.5652466, -18.0869102, 18.3768997, -37.6364975, 37.6521568

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 92

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
time: 8.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1438005, upper bound: 43.1426490
time: 7.59 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -27.0998726, 21.6743279, -19.7793999, 15.8471279, -42.9469986, 41.4537277
1: -24.4106026, 19.3896446, -17.7844124, 14.2541809, -38.6647835, 37.1740532
2: -30.8508549, 19.2533131, -22.4641666, 14.0962296, -44.9470825, 41.7174797
3: -33.0800056, 16.3766289, -24.1717720, 12.0803623, -45.1603699, 40.5484009
4: -31.3325195, 22.1236420, -22.9433365, 16.1460857, -47.4786034, 45.0669785
5: -26.9409790, 20.9609966, -19.6832161, 15.3848476, -42.3258286, 40.6442108
6: -24.6803856, 24.5318871, -17.9507370, 17.9319839, -42.6123543, 42.4826202
7: -27.3474941, 25.9527702, -19.8933125, 19.1402740, -46.4877701, 45.8460770
8: -38.2813950, 18.0903797, -28.2664909, 13.0147495, -51.2961426, 46.3568726
9: -24.0825577, 24.4897652, -17.5563927, 17.8330212, -41.9155807, 42.0461578

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 92

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1378680, upper bound: 43.1381622
time: 7.23 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1378680, upper bound: 43.1381622
time: 7.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -21.2508430, 17.0090275, -23.4310989, 18.7306175, -39.9814606, 40.4401245
1: -19.0841599, 15.2855196, -21.1348801, 16.8315334, -35.9156952, 36.4203987
2: -24.0822086, 15.1320543, -26.6416378, 16.6077614, -40.6899719, 41.7736893
3: -25.8781204, 13.0060940, -28.7097130, 14.2124481, -40.0905685, 41.7158051
4: -24.5291386, 17.3658009, -27.2073326, 19.0809383, -43.6100769, 44.5731354
5: -21.0896606, 16.4712334, -23.3456135, 18.1633282, -39.2529907, 39.8168488
6: -19.3226566, 19.2417107, -21.2629757, 21.2273579, -40.5500107, 40.5046844
7: -21.3653717, 20.3800621, -23.6135941, 22.6121254, -43.9774971, 43.9936562
8: -30.1133308, 14.2286968, -33.3218498, 15.3486996, -45.4620285, 47.5505447
9: -18.8655033, 19.1612873, -20.7837887, 21.0979710, -39.9634705, 39.9450760

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 161

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1434155, upper bound: 43.1420335
time: 6.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1434155, upper bound: 43.1420335
time: 7.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26.5893898, 21.2647877, -22.8025627, 18.2355423, -44.8249321, 44.0673523
1: -23.9749107, 19.0374546, -20.5740166, 16.3928661, -40.3677750, 39.6114693
2: -30.2876034, 18.8867950, -25.9360237, 16.1715164, -46.4591141, 44.8228149
3: -32.4783287, 16.0548096, -27.9514389, 13.8323250, -46.3106537, 44.0062408
4: -30.7822666, 21.7098808, -26.4999161, 18.5709076, -49.3531723, 48.2097969
5: -26.4452286, 20.5827942, -22.7339478, 17.6948204, -44.1400452, 43.3167381
6: -24.1950417, 24.0732288, -20.6898899, 20.6654663, -44.8605080, 44.7631187
7: -26.8524628, 25.5108147, -22.9779053, 22.0509357, -48.9033966, 48.4887161
8: -37.6254196, 17.6784782, -32.4880905, 14.9004383, -52.5258560, 50.1665688
9: -23.6259270, 24.0173035, -20.2331886, 20.5327568, -44.1586761, 44.2504845

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 161

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1376026, upper bound: 43.1378050
time: 7.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -43.1376026, upper bound: 43.1378050
time: 6.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -28.3339691, 22.6485634, -19.7323551, 15.8057566, -44.1397247, 42.3809166
1: -25.5105534, 20.2498550, -17.7340775, 14.2149906, -39.7255440, 37.9839287
2: -32.2058411, 20.0842323, -22.3935528, 14.0598907, -46.2657318, 42.4777832
3: -34.5266380, 17.1906452, -24.0956440, 12.0616379, -46.5882759, 41.2862892
4: -32.6802139, 23.0794678, -22.8664246, 16.1021252, -48.7823372, 45.9458885
5: -28.1018181, 21.8674145, -19.6197033, 15.3350811, -43.4368973, 41.4871178
6: -25.8014259, 25.5789146, -17.9011612, 17.8794689, -43.6808891, 43.4800644
7: -28.5716820, 27.0196342, -19.8320370, 19.0649834, -47.6366653, 46.8516693
8: -39.8003731, 19.0315323, -28.1641960, 13.0115261, -52.8118973, 47.1957245
9: -25.1833687, 25.5482254, -17.5039310, 17.7847481, -42.9681168, 43.0521545

Time for backsubstitution: 0.81 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 12.83 + 587.47 = 600.30 seconds
