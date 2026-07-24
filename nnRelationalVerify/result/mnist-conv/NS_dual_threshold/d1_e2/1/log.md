## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06342095499999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.4947338, -12.0108538, -12.4947338, -12.0108538, -0.2105641, 0.2105641)
1: (-12.4234695, -12.0680580, -12.4234695, -12.0680580, -0.1409148, 0.1409148)
2: (-8.9157972, -8.6541748, -8.9157972, -8.6541748, -0.1139829, 0.1139829)
3: (-8.4745026, -8.1830769, -8.4745026, -8.1830769, -0.1432159, 0.1432159)
4: (-3.6523490, -3.3683569, -3.6523490, -3.3683569, -0.1160847, 0.1160847)
5: (-5.3451724, -5.0357218, -5.3451724, -5.0357218, -0.1068949, 0.1068948)
6: (-13.7031565, -13.2556944, -13.7031565, -13.2556944, -0.1659200, 0.1659200)
7: (-3.5963995, -3.3030066, -3.5963995, -3.3030066, -0.1338000, 0.1338000)
8: (-1.8303380, -1.5334034, -1.8303380, -1.5334034, -0.1171519, 0.1171519)
9: (2.8932495, 3.0946174, 2.8932495, 3.0946174, -0.0977726, 0.0977726)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.87 + 33.27 = 55.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0667589, upper bound: 0.0667588

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0558195, upper bound: 0.0664321
time: 2.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664317, upper bound: 0.0664318
time: 2.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.44 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.44
Output dim: 9, lower bound: -0.0558195, upper bound: 0.0664321
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.44
Output dim: 9, lower bound: -0.0664317, upper bound: 0.0664318

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.4985085, -12.0152187, -12.4947338, -12.0114107, -0.2103176, 0.2033267
1: -12.4288855, -12.0743332, -12.4234695, -12.0688610, -0.1396670, 0.1298993
2: -8.9030781, -8.6535711, -8.9140863, -8.6541748, -0.1004341, 0.1117654
3: -8.4718475, -8.1805115, -8.4741650, -8.1830769, -0.1381090, 0.1424544
4: -3.6523294, -3.3683152, -3.6523457, -3.3683569, -0.1160318, 0.1160839
5: -5.3451982, -5.0369172, -5.3451724, -5.0358758, -0.1066321, 0.1055963
6: -13.7034760, -13.2564182, -13.7031565, -13.2557964, -0.1658880, 0.1649687
7: -3.5989194, -3.3283336, -3.5963995, -3.3068631, -0.1288443, 0.1049256
8: -1.8258963, -1.5308955, -1.8297729, -1.5334034, -0.1099774, 0.1157850
9: 2.9236603, 3.0950036, 2.8975067, 3.0946174, -0.0644492, 0.0909585

Time for backsubstitution: 7.73 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1241

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0538631, upper bound: 0.0645580
time: 2.61 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0541818, upper bound: 0.0645583
time: 2.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.4947338, -12.0111685, -12.4947338, -12.0108538, -0.2105641, 0.2040396
1: -12.4234695, -12.0685129, -12.4234695, -12.0680580, -0.1409148, 0.1301540
2: -8.9157248, -8.6541748, -8.9157972, -8.6541748, -0.1039537, 0.1139829
3: -8.4742899, -8.1830769, -8.4745026, -8.1830769, -0.1380887, 0.1432159
4: -3.6523454, -3.3683569, -3.6523490, -3.3683569, -0.1160314, 0.1160847
5: -5.3451724, -5.0357313, -5.3451724, -5.0357218, -0.1068949, 0.1058676
6: -13.7031565, -13.2557182, -13.7031565, -13.2556944, -0.1659200, 0.1651974
7: -3.5963995, -3.3030052, -3.5963995, -3.3030066, -0.1338000, 0.1104110
8: -1.8301201, -1.5334034, -1.8303380, -1.5334034, -0.1103985, 0.1171519
9: 2.8932500, 3.0946174, 2.8932495, 3.0946174, -0.0711721, 0.0977726

Time for backsubstitution: 8.38 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664313, upper bound: 0.0558190
time: 2.76 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664322, upper bound: 0.0664320
time: 2.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.92 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 9, lower bound: -0.0538631, upper bound: 0.0645580
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 9, lower bound: -0.0541818, upper bound: 0.0645583
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 9, lower bound: -0.0664313, upper bound: 0.0558190
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.92
Output dim: 9, lower bound: -0.0664322, upper bound: 0.0664320

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -12.4980373, -12.0152187, -12.4947338, -12.0114107, -0.2097037, 0.2033267
1: -12.4272118, -12.0743332, -12.4234695, -12.0688610, -0.1366094, 0.1298993
2: -8.9023037, -8.6535711, -8.9140863, -8.6541748, -0.0978667, 0.1117654
3: -8.4710579, -8.1805115, -8.4741650, -8.1830769, -0.1372129, 0.1424544
4: -3.6521955, -3.3683152, -3.6523457, -3.3683569, -0.1152838, 0.1160839
5: -5.3451982, -5.0374336, -5.3451724, -5.0358758, -0.1066321, 0.1043023
6: -13.7019148, -13.2564182, -13.7031565, -13.2557964, -0.1616918, 0.1649687
7: -3.5989194, -3.3303103, -3.5963995, -3.3068631, -0.1288443, 0.1009871
8: -1.8258963, -1.5309024, -1.8297729, -1.5334034, -0.1099774, 0.1157601
9: 2.9248981, 3.0950036, 2.8975067, 3.0946174, -0.0622082, 0.0909585

Time for backsubstitution: 7.83 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0538634, upper bound: 0.0541819
time: 2.57 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0538634, upper bound: 0.0645586
time: 2.56 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -12.4975815, -12.0131502, -12.4945192, -12.0114107, -0.2101057, 0.2064981
1: -12.4222364, -12.0676775, -12.4219351, -12.0688610, -0.1369209, 0.1441948
2: -8.8960943, -8.6500864, -8.9120741, -8.6541748, -0.0968765, 0.1241487
3: -8.4713011, -8.1768341, -8.4740391, -8.1830769, -0.1384579, 0.1466727
4: -3.6496994, -3.3679569, -3.6516643, -3.3683569, -0.1146625, 0.1185596
5: -5.3468809, -5.0406618, -5.3451724, -5.0367393, -0.1128356, 0.1039075
6: -13.6916504, -13.2507401, -13.7002344, -13.2557964, -0.1603953, 0.1843574
7: -3.6064024, -3.3379049, -3.5963995, -3.3091345, -0.1477759, 0.1009717
8: -1.8259037, -1.5310037, -1.8297729, -1.5334280, -0.1100488, 0.1157309
9: 2.9286337, 3.1000061, 2.8987155, 3.0946174, -0.0625050, 0.1019380

Time for backsubstitution: 8.51 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0541820, upper bound: 0.0541820
time: 3.02 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0541820, upper bound: 0.0645587
time: 2.99 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.4947338, -12.0111685, -12.4985085, -12.0152187, -0.2033265, 0.2111478
1: -12.4234695, -12.0685129, -12.4288855, -12.0743332, -0.1298993, 0.1410770
2: -8.9157248, -8.6541748, -8.9030781, -8.6535711, -0.1135969, 0.1004341
3: -8.4742899, -8.1830769, -8.4718475, -8.1805115, -0.1430975, 0.1381090
4: -3.6523454, -3.3683569, -3.6523294, -3.3683152, -0.1160898, 0.1160318
5: -5.3451724, -5.0357313, -5.3451982, -5.0369172, -0.1055963, 0.1067975
6: -13.7031565, -13.2557182, -13.7034760, -13.2564182, -0.1649688, 0.1659971
7: -3.5963995, -3.3030052, -3.5989194, -3.3283336, -0.1049256, 0.1327024
8: -1.8301201, -1.5334034, -1.8258963, -1.5308955, -0.1167149, 0.1099774
9: 2.8932500, 3.0946174, 2.9236603, 3.0950036, -0.0952155, 0.0644492

Time for backsubstitution: 8.59 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1241

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645234, upper bound: 0.0541816
time: 2.75 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645574, upper bound: 0.0541819
time: 2.69 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.4947338, -12.0111685, -12.4947338, -12.0111685, -0.2040396, 0.2040396
1: -12.4234695, -12.0685129, -12.4234695, -12.0685129, -0.1301540, 0.1301540
2: -8.9157248, -8.6541748, -8.9157248, -8.6541748, -0.1039537, 0.1039537
3: -8.4742899, -8.1830769, -8.4742899, -8.1830769, -0.1380887, 0.1380887
4: -3.6523454, -3.3683569, -3.6523454, -3.3683569, -0.1160314, 0.1160314
5: -5.3451724, -5.0357313, -5.3451724, -5.0357313, -0.1058677, 0.1058676
6: -13.7031565, -13.2557182, -13.7031565, -13.2557182, -0.1651973, 0.1651974
7: -3.5963995, -3.3030052, -3.5963995, -3.3030052, -0.1104111, 0.1104110
8: -1.8301201, -1.5334034, -1.8301201, -1.5334034, -0.1103985, 0.1103985
9: 2.8932500, 3.0946174, 2.8932500, 3.0946174, -0.0711721, 0.0711721

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1241
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1241

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645247, upper bound: 0.0550611
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645587, upper bound: 0.0550615
time: 2.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.38 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0538634, upper bound: 0.0541819
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0538634, upper bound: 0.0645586
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0541820, upper bound: 0.0541820
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0541820, upper bound: 0.0645587
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0645234, upper bound: 0.0541816
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0645574, upper bound: 0.0541819
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0645247, upper bound: 0.0550611
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 14.38
Output dim: 9, lower bound: -0.0645587, upper bound: 0.0550615

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -12.4980373, -12.0152187, -12.4947338, -12.0111685, -0.2105341, 0.2033267
1: -12.4272118, -12.0743332, -12.4234695, -12.0685129, -0.1380196, 0.1298993
2: -8.9023037, -8.6535711, -8.9157248, -8.6541748, -0.0978667, 0.1135969
3: -8.4710579, -8.1805115, -8.4742899, -8.1830769, -0.1372129, 0.1430975
4: -3.6521955, -3.3683152, -3.6523454, -3.3683569, -0.1152838, 0.1160898
5: -5.3451982, -5.0374336, -5.3451724, -5.0357313, -0.1067975, 0.1043023
6: -13.7019148, -13.2564182, -13.7031565, -13.2557182, -0.1618011, 0.1649687
7: -3.5989194, -3.3303103, -3.5963995, -3.3030052, -0.1327024, 0.1009871
8: -1.8258963, -1.5309024, -1.8301201, -1.5334034, -0.1099774, 0.1166900
9: 2.9248981, 3.0950036, 2.8932500, 3.0946174, -0.0622082, 0.0952156

Time for backsubstitution: 8.56 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0538635, upper bound: 0.0645263
time: 2.63 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0538635, upper bound: 0.0645587
time: 2.64 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -12.4975815, -12.0131502, -12.4945192, -12.0111685, -0.2109361, 0.2064981
1: -12.4222364, -12.0676775, -12.4219351, -12.0685129, -0.1383309, 0.1441948
2: -8.8960943, -8.6500864, -8.9137115, -8.6541748, -0.0968765, 0.1259588
3: -8.4713011, -8.1768341, -8.4741640, -8.1830769, -0.1384579, 0.1473076
4: -3.6496994, -3.3679569, -3.6516633, -3.3683569, -0.1146625, 0.1185658
5: -5.3468809, -5.0406618, -5.3451724, -5.0365963, -0.1130010, 0.1039075
6: -13.6916504, -13.2507401, -13.7002344, -13.2557182, -0.1605046, 0.1843574
7: -3.6064024, -3.3379049, -3.5963995, -3.3053184, -0.1516339, 0.1009717
8: -1.8259037, -1.5310037, -1.8301201, -1.5334280, -0.1100488, 0.1166607
9: 2.9286337, 3.1000061, 2.8944578, 3.0946174, -0.0625050, 0.1061951

Time for backsubstitution: 8.47 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0541821, upper bound: 0.0645247
time: 2.82 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0541822, upper bound: 0.0645247
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.4942646, -12.0111685, -12.4985085, -12.0152187, -0.2027125, 0.2111478
1: -12.4217949, -12.0685129, -12.4288855, -12.0743332, -0.1268412, 0.1410770
2: -8.9149494, -8.6541748, -8.9030781, -8.6535711, -0.1110895, 0.1004341
3: -8.4735003, -8.1830769, -8.4718475, -8.1805115, -0.1422220, 0.1381090
4: -3.6522112, -3.3683569, -3.6523294, -3.3683152, -0.1153430, 0.1160318
5: -5.3451724, -5.0362453, -5.3451982, -5.0369172, -0.1055963, 0.1055143
6: -13.7015915, -13.2557182, -13.7034760, -13.2564182, -0.1607724, 0.1659971
7: -3.5963995, -3.3048902, -3.5989194, -3.3283336, -0.1049256, 0.1288559
8: -1.8301201, -1.5334108, -1.8258963, -1.5308955, -0.1167149, 0.1099525
9: 2.8944612, 3.0946174, 2.9236603, 3.0950036, -0.0930281, 0.0644492

Time for backsubstitution: 8.55 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645235, upper bound: 0.0538627
time: 2.60 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645235, upper bound: 0.0541813
time: 2.61 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.4938087, -12.0091009, -12.4982929, -12.0152187, -0.2031145, 0.2143202
1: -12.4168205, -12.0618582, -12.4273472, -12.0743332, -0.1271527, 0.1553727
2: -8.9087400, -8.6506910, -8.9010668, -8.6535711, -0.1100993, 0.1127331
3: -8.4737434, -8.1793985, -8.4717216, -8.1805115, -0.1434672, 0.1422981
4: -3.6497169, -3.3679972, -3.6516459, -3.3683152, -0.1147217, 0.1185063
5: -5.3468571, -5.0394764, -5.3451982, -5.0377836, -0.1117876, 0.1051195
6: -13.6913290, -13.2500401, -13.7005539, -13.2564182, -0.1594759, 0.1853862
7: -3.6036847, -3.3126845, -3.5989194, -3.3305440, -0.1237134, 0.1288406
8: -1.8301275, -1.5335114, -1.8258963, -1.5309219, -0.1167861, 0.1099233
9: 2.8984590, 3.0993590, 2.9248095, 3.0950036, -0.0933250, 0.0752198

Time for backsubstitution: 8.51 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645573, upper bound: 0.0538630
time: 2.69 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645573, upper bound: 0.0541816
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.4942646, -12.0111685, -12.4947338, -12.0111685, -0.2034259, 0.2040396
1: -12.4217949, -12.0685129, -12.4234695, -12.0685129, -0.1270965, 0.1301540
2: -8.9149494, -8.6541748, -8.9157248, -8.6541748, -0.1013861, 0.1039537
3: -8.4735003, -8.1830769, -8.4742899, -8.1830769, -0.1371925, 0.1380887
4: -3.6522112, -3.3683569, -3.6523454, -3.3683569, -0.1152836, 0.1160314
5: -5.3451724, -5.0362453, -5.3451724, -5.0357313, -0.1058677, 0.1045737
6: -13.7015915, -13.2557182, -13.7031565, -13.2557182, -0.1610013, 0.1651974
7: -3.5963995, -3.3048902, -3.5963995, -3.3030052, -0.1104111, 0.1064725
8: -1.8301201, -1.5334108, -1.8301201, -1.5334034, -0.1103985, 0.1103736
9: 2.8944612, 3.0946174, 2.8932500, 3.0946174, -0.0689310, 0.0711721

Time for backsubstitution: 8.57 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645247, upper bound: 0.0547155
time: 2.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645247, upper bound: 0.0550611
time: 2.73 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.4938087, -12.0091009, -12.4945192, -12.0111685, -0.2038279, 0.2072110
1: -12.4168205, -12.0618582, -12.4219351, -12.0685129, -0.1274080, 0.1444495
2: -8.9087400, -8.6506910, -8.9137115, -8.6541748, -0.1003960, 0.1163704
3: -8.4737434, -8.1793985, -8.4741640, -8.1830769, -0.1384376, 0.1423213
4: -3.6497169, -3.3679972, -3.6516633, -3.3683569, -0.1146623, 0.1185074
5: -5.3468571, -5.0394764, -5.3451724, -5.0365963, -0.1120715, 0.1041788
6: -13.6913290, -13.2500401, -13.7002344, -13.2557182, -0.1597048, 0.1845859
7: -3.6036847, -3.3126845, -3.5963995, -3.3053184, -0.1294447, 0.1062595
8: -1.8301275, -1.5335114, -1.8301201, -1.5334280, -0.1104698, 0.1103444
9: 2.8984590, 3.0993590, 2.8944578, 3.0946174, -0.0689641, 0.0822106

Time for backsubstitution: 8.53 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1241
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1241

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645588, upper bound: 0.0547161
time: 2.63 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645588, upper bound: 0.0550615
time: 2.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 13.99 seconds
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0538635, upper bound: 0.0645263
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0538635, upper bound: 0.0645587
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0541821, upper bound: 0.0645247
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0541822, upper bound: 0.0645247
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645235, upper bound: 0.0538627
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645235, upper bound: 0.0541813
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645573, upper bound: 0.0538630
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645573, upper bound: 0.0541816
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645247, upper bound: 0.0547155
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645247, upper bound: 0.0550611
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645588, upper bound: 0.0547161
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.99
Output dim: 9, lower bound: -0.0645588, upper bound: 0.0550615

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -12.4980373, -12.0152187, -12.4942646, -12.0111685, -0.2105341, 0.2027125
1: -12.4272118, -12.0743332, -12.4217949, -12.0685129, -0.1380196, 0.1268412
2: -8.9023037, -8.6535711, -8.9149494, -8.6541748, -0.0978667, 0.1110895
3: -8.4710579, -8.1805115, -8.4735003, -8.1830769, -0.1372129, 0.1422220
4: -3.6521955, -3.3683152, -3.6522112, -3.3683569, -0.1152838, 0.1153430
5: -5.3451982, -5.0374336, -5.3451724, -5.0362453, -0.1055143, 0.1043023
6: -13.7019148, -13.2564182, -13.7015915, -13.2557182, -0.1618011, 0.1607724
7: -3.5989194, -3.3303103, -3.5963995, -3.3048902, -0.1288559, 0.1009871
8: -1.8258963, -1.5309024, -1.8301201, -1.5334108, -0.1099526, 0.1166900
9: 2.9248981, 3.0950036, 2.8944612, 3.0946174, -0.0622082, 0.0930282

Time for backsubstitution: 8.57 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0522434, upper bound: 0.0643196
time: 2.64 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0538204, upper bound: 0.0644896
time: 2.70 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -12.4980373, -12.0152187, -12.4938087, -12.0091009, -0.2137094, 0.2033229
1: -12.4272118, -12.0743332, -12.4168205, -12.0618582, -0.1523155, 0.1298987
2: -8.9023037, -8.6535711, -8.9087400, -8.6506910, -0.1101656, 0.1135392
3: -8.4710579, -8.1805115, -8.4737434, -8.1793985, -0.1414037, 0.1430726
4: -3.6521955, -3.3683152, -3.6497169, -3.3679972, -0.1177585, 0.1160890
5: -5.3451982, -5.0374336, -5.3468571, -5.0394764, -0.1067958, 0.1104937
6: -13.7019148, -13.2564182, -13.6913290, -13.2500401, -0.1811901, 0.1649685
7: -3.5989194, -3.3303103, -3.6036847, -3.3126845, -0.1325486, 0.1197748
8: -1.8258963, -1.5309024, -1.8301275, -1.5335114, -0.1099698, 0.1167631
9: 2.9248981, 3.0950036, 2.8984590, 3.0993590, -0.0729787, 0.0950010

Time for backsubstitution: 8.55 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0522434, upper bound: 0.0643206
time: 2.67 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0538204, upper bound: 0.0645215
time: 2.74 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -12.4975815, -12.0131502, -12.4942646, -12.0111685, -0.2111440, 0.2058873
1: -12.4222364, -12.0676775, -12.4217949, -12.0685129, -0.1410767, 0.1411369
2: -8.8960943, -8.6500864, -8.9149494, -8.6541748, -0.1004341, 0.1235063
3: -8.4713011, -8.1768341, -8.4735003, -8.1830769, -0.1381062, 0.1464565
4: -3.6496994, -3.3679569, -3.6522112, -3.3683569, -0.1160318, 0.1178190
5: -5.3468809, -5.0406618, -5.3451724, -5.0362453, -0.1117181, 0.1055963
6: -13.6916504, -13.2507401, -13.7015915, -13.2557182, -0.1659969, 0.1801611
7: -3.6064024, -3.3379049, -3.5963995, -3.3048902, -0.1478895, 0.1049256
8: -1.8259037, -1.5310037, -1.8301201, -1.5334108, -0.1100258, 0.1167071
9: 2.9286337, 3.1000061, 2.8944612, 3.0946174, -0.0644490, 0.1040667

Time for backsubstitution: 8.85 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0500316, upper bound: 0.0642831
time: 2.77 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0541444, upper bound: 0.0644878
time: 2.77 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -12.4975815, -12.0131502, -12.4938087, -12.0091009, -0.2109361, 0.2031145
1: -12.4222364, -12.0676775, -12.4168205, -12.0618582, -0.1383309, 0.1271527
2: -8.8960943, -8.6500864, -8.9087400, -8.6506910, -0.0968765, 0.1100993
3: -8.4713011, -8.1768341, -8.4737434, -8.1793985, -0.1384579, 0.1434672
4: -3.6496994, -3.3679569, -3.6497169, -3.3679972, -0.1146625, 0.1147217
5: -5.3468809, -5.0406618, -5.3468571, -5.0394764, -0.1051195, 0.1039075
6: -13.6916504, -13.2507401, -13.6913290, -13.2500401, -0.1605046, 0.1594759
7: -3.6064024, -3.3379049, -3.6036847, -3.3126845, -0.1288406, 0.1009717
8: -1.8259037, -1.5310037, -1.8301275, -1.5335114, -0.1099233, 0.1166607
9: 2.9286337, 3.1000061, 2.8984590, 3.0993590, -0.0625050, 0.0933250

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0500316, upper bound: 0.0642817
time: 2.78 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0541445, upper bound: 0.0644878
time: 2.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.4942646, -12.0111685, -12.4980373, -12.0152187, -0.2027125, 0.2105343
1: -12.4217949, -12.0685129, -12.4272118, -12.0743332, -0.1268412, 0.1380195
2: -8.9149494, -8.6541748, -8.9023037, -8.6535711, -0.1110895, 0.0978666
3: -8.4735003, -8.1830769, -8.4710579, -8.1805115, -0.1422220, 0.1372129
4: -3.6522112, -3.3683569, -3.6521955, -3.3683152, -0.1153430, 0.1152840
5: -5.3451724, -5.0362453, -5.3451982, -5.0374336, -0.1043023, 0.1055143
6: -13.7015915, -13.2557182, -13.7019148, -13.2564182, -0.1607724, 0.1618011
7: -3.5963995, -3.3048902, -3.5989194, -3.3303103, -0.1009871, 0.1288559
8: -1.8301201, -1.5334108, -1.8258963, -1.5309024, -0.1166899, 0.1099525
9: 2.8944612, 3.0946174, 2.9248981, 3.0950036, -0.0930281, 0.0622081

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0628878, upper bound: 0.0538174
time: 2.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0644864, upper bound: 0.0538438
time: 2.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.4942646, -12.0111685, -12.4975815, -12.0131502, -0.2058873, 0.2111442
1: -12.4217949, -12.0685129, -12.4222364, -12.0676775, -0.1411369, 0.1410767
2: -8.9149494, -8.6541748, -8.8960943, -8.6500864, -0.1235063, 0.1004341
3: -8.4735003, -8.1830769, -8.4713011, -8.1768341, -0.1464565, 0.1381061
4: -3.6522112, -3.3683569, -3.6496994, -3.3679569, -0.1178191, 0.1160316
5: -5.3451724, -5.0362453, -5.3468809, -5.0406618, -0.1055963, 0.1117181
6: -13.7015915, -13.2557182, -13.6916504, -13.2507401, -0.1801611, 0.1659970
7: -3.5963995, -3.3048902, -3.6064024, -3.3379049, -0.1049255, 0.1478895
8: -1.8301201, -1.5334108, -1.8259037, -1.5310037, -0.1167071, 0.1100257
9: 2.8944612, 3.0946174, 2.9286337, 3.1000061, -0.1040667, 0.0644491

Time for backsubstitution: 8.34 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0628878, upper bound: 0.0538184
time: 2.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0644864, upper bound: 0.0541448
time: 2.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.4938087, -12.0091009, -12.4980373, -12.0152187, -0.2033229, 0.2137094
1: -12.4168205, -12.0618582, -12.4272118, -12.0743332, -0.1298988, 0.1523154
2: -8.9087400, -8.6506910, -8.9023037, -8.6535711, -0.1135392, 0.1101657
3: -8.4737434, -8.1793985, -8.4710579, -8.1805115, -0.1430726, 0.1414037
4: -3.6497169, -3.3679972, -3.6521955, -3.3683152, -0.1160890, 0.1177585
5: -5.3468571, -5.0394764, -5.3451982, -5.0374336, -0.1104937, 0.1067958
6: -13.6913290, -13.2500401, -13.7019148, -13.2564182, -0.1649686, 0.1811901
7: -3.6036847, -3.3126845, -3.5989194, -3.3303103, -0.1197748, 0.1325487
8: -1.8301275, -1.5335114, -1.8258963, -1.5309024, -0.1167631, 0.1099697
9: 2.8984590, 3.0993590, 2.9248981, 3.0950036, -0.0950009, 0.0729788

Time for backsubstitution: 7.85 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0604913, upper bound: 0.0537809
time: 2.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645192, upper bound: 0.0538199
time: 2.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.4938087, -12.0091009, -12.4975815, -12.0131502, -0.2031145, 0.2109363
1: -12.4168205, -12.0618582, -12.4222364, -12.0676775, -0.1271527, 0.1383310
2: -8.9087400, -8.6506910, -8.8960943, -8.6500864, -0.1100993, 0.0968764
3: -8.4737434, -8.1793985, -8.4713011, -8.1768341, -0.1434672, 0.1384579
4: -3.6497169, -3.3679972, -3.6496994, -3.3679569, -0.1147217, 0.1146626
5: -5.3468571, -5.0394764, -5.3468809, -5.0406618, -0.1039075, 0.1051195
6: -13.6913290, -13.2500401, -13.6916504, -13.2507401, -0.1594759, 0.1605046
7: -3.6036847, -3.3126845, -3.6064024, -3.3379049, -0.1009717, 0.1288406
8: -1.8301275, -1.5335114, -1.8259037, -1.5310037, -0.1166607, 0.1099233
9: 2.8984590, 3.0993590, 2.9286337, 3.1000061, -0.0933250, 0.0625051

Time for backsubstitution: 8.46 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0604913, upper bound: 0.0537809
time: 2.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645195, upper bound: 0.0538202
time: 2.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.4942646, -12.0111685, -12.4942646, -12.0111685, -0.2034259, 0.2034256
1: -12.4217949, -12.0685129, -12.4217949, -12.0685129, -0.1270965, 0.1270965
2: -8.9149494, -8.6541748, -8.9149494, -8.6541748, -0.1013861, 0.1013861
3: -8.4735003, -8.1830769, -8.4735003, -8.1830769, -0.1371925, 0.1371926
4: -3.6522112, -3.3683569, -3.6522112, -3.3683569, -0.1152836, 0.1152835
5: -5.3451724, -5.0362453, -5.3451724, -5.0362453, -0.1045737, 0.1045737
6: -13.7015915, -13.2557182, -13.7015915, -13.2557182, -0.1610013, 0.1610013
7: -3.5963995, -3.3048902, -3.5963995, -3.3048902, -0.1064725, 0.1064725
8: -1.8301201, -1.5334108, -1.8301201, -1.5334108, -0.1103736, 0.1103736
9: 2.8944612, 3.0946174, 2.8944612, 3.0946174, -0.0689310, 0.0689310

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0628888, upper bound: 0.0546774
time: 2.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0644870, upper bound: 0.0547034
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.4942646, -12.0111685, -12.4938087, -12.0091009, -0.2066004, 0.2040355
1: -12.4217949, -12.0685129, -12.4168205, -12.0618582, -0.1413922, 0.1301537
2: -8.9149494, -8.6541748, -8.9087400, -8.6506910, -0.1138030, 0.1039536
3: -8.4735003, -8.1830769, -8.4737434, -8.1793985, -0.1414269, 0.1380858
4: -3.6522112, -3.3683569, -3.6497169, -3.3679972, -0.1177596, 0.1160314
5: -5.3451724, -5.0362453, -5.3468571, -5.0394764, -0.1058676, 0.1107775
6: -13.7015915, -13.2557182, -13.6913290, -13.2500401, -0.1803898, 0.1651971
7: -3.5963995, -3.3048902, -3.6036847, -3.3126845, -0.1104109, 0.1255060
8: -1.8301201, -1.5334108, -1.8301275, -1.5335114, -0.1103908, 0.1104468
9: 2.8944612, 3.0946174, 2.8984590, 3.0993590, -0.0799696, 0.0711719

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0628888, upper bound: 0.0546784
time: 2.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0644870, upper bound: 0.0550240
time: 2.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.4938087, -12.0091009, -12.4942646, -12.0111685, -0.2040355, 0.2066004
1: -12.4168205, -12.0618582, -12.4217949, -12.0685129, -0.1301537, 0.1413922
2: -8.9087400, -8.6506910, -8.9149494, -8.6541748, -0.1039536, 0.1138030
3: -8.4737434, -8.1793985, -8.4735003, -8.1830769, -0.1380858, 0.1414269
4: -3.6497169, -3.3679972, -3.6522112, -3.3683569, -0.1160313, 0.1177596
5: -5.3468571, -5.0394764, -5.3451724, -5.0362453, -0.1107775, 0.1058676
6: -13.6913290, -13.2500401, -13.7015915, -13.2557182, -0.1651971, 0.1803899
7: -3.6036847, -3.3126845, -3.5963995, -3.3048902, -0.1255060, 0.1104110
8: -1.8301275, -1.5335114, -1.8301201, -1.5334108, -0.1104468, 0.1103908
9: 2.8984590, 3.0993590, 2.8944612, 3.0946174, -0.0711719, 0.0799696

Time for backsubstitution: 8.49 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0604926, upper bound: 0.0546299
time: 2.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645200, upper bound: 0.0546787
time: 2.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.4938087, -12.0091009, -12.4938087, -12.0091009, -0.2038279, 0.2038276
1: -12.4168205, -12.0618582, -12.4168205, -12.0618582, -0.1274080, 0.1274080
2: -8.9087400, -8.6506910, -8.9087400, -8.6506910, -0.1003960, 0.1003960
3: -8.4737434, -8.1793985, -8.4737434, -8.1793985, -0.1384376, 0.1384376
4: -3.6497169, -3.3679972, -3.6497169, -3.3679972, -0.1146623, 0.1146622
5: -5.3468571, -5.0394764, -5.3468571, -5.0394764, -0.1041788, 0.1041788
6: -13.6913290, -13.2500401, -13.6913290, -13.2500401, -0.1597048, 0.1597047
7: -3.6036847, -3.3126845, -3.6036847, -3.3126845, -0.1062595, 0.1062595
8: -1.8301275, -1.5335114, -1.8301275, -1.5335114, -0.1103444, 0.1103444
9: 2.8984590, 3.0993590, 2.8984590, 3.0993590, -0.0689641, 0.0689641

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2459
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 2459

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0604919, upper bound: 0.0546296
time: 2.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0645210, upper bound: 0.0546792
time: 2.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 13.86 seconds
NS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0522434, upper bound: 0.0643196
NS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0538204, upper bound: 0.0644896
NS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0522434, upper bound: 0.0643206
NS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0538204, upper bound: 0.0645215
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0500316, upper bound: 0.0642831
NS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0541444, upper bound: 0.0644878
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0500316, upper bound: 0.0642817
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0541445, upper bound: 0.0644878
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0628878, upper bound: 0.0538174
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0644864, upper bound: 0.0538438
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0628878, upper bound: 0.0538184
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0644864, upper bound: 0.0541448
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0604913, upper bound: 0.0537809
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0645192, upper bound: 0.0538199
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0604913, upper bound: 0.0537809
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0645195, upper bound: 0.0538202
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0628888, upper bound: 0.0546774
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0644870, upper bound: 0.0547034
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0628888, upper bound: 0.0546784
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0644870, upper bound: 0.0550240
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0604926, upper bound: 0.0546299
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0645200, upper bound: 0.0546787
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0604919, upper bound: 0.0546296
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.86
Output dim: 9, lower bound: -0.0645210, upper bound: 0.0546792

## BFS NS instance: NS_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -12.4973469, -12.0152187, -12.4940786, -12.0111685, -0.2097771, 0.2025106
1: -12.4272118, -12.0745983, -12.4217949, -12.0685844, -0.1379089, 0.1264271
2: -8.9022236, -8.6535711, -8.9149294, -8.6541748, -0.0976428, 0.1110296
3: -8.4710579, -8.1805172, -8.4735003, -8.1830788, -0.1371862, 0.1421239
4: -3.6521955, -3.3691342, -3.6522112, -3.3685760, -0.1149422, 0.1140631
5: -5.3451948, -5.0374336, -5.3451724, -5.0362453, -0.1055096, 0.1043011
6: -13.7019148, -13.2569294, -13.7015915, -13.2558584, -0.1615883, 0.1599754
7: -3.5989194, -3.3305776, -3.5963995, -3.3049626, -0.1287842, 0.1007185
8: -1.8256156, -1.5309024, -1.8300464, -1.5334108, -0.1096545, 0.1166103
9: 2.9258804, 3.0950036, 2.8947239, 3.0946174, -0.0611651, 0.0927496

Time for backsubstitution: 8.41 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0518192, upper bound: 0.0639796
time: 2.69 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0518196, upper bound: 0.0640660
time: 2.68 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -12.4979401, -12.0143433, -12.4942303, -12.0111685, -0.2101302, 0.2037492
1: -12.4274216, -12.0745392, -12.4217949, -12.0685825, -0.1385872, 0.1265029
2: -8.9021072, -8.6536016, -8.9148827, -8.6541748, -0.0976018, 0.1113962
3: -8.4709721, -8.1806393, -8.4735003, -8.1831198, -0.1373471, 0.1420702
4: -3.6528554, -3.3689485, -3.6522112, -3.3685734, -0.1170386, 0.1142985
5: -5.3451953, -5.0374312, -5.3451719, -5.0362453, -0.1055102, 0.1043087
6: -13.7023239, -13.2568140, -13.7015915, -13.2558537, -0.1628939, 0.1601212
7: -3.5992863, -3.3303092, -3.5963995, -3.3048902, -0.1292245, 0.1008823
8: -1.8258722, -1.5305343, -1.8301125, -1.5334108, -0.1098162, 0.1170979
9: 2.9249821, 3.0962882, 2.8944898, 3.0946174, -0.0617316, 0.0944582

Time for backsubstitution: 8.38 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0534193, upper bound: 0.0639783
time: 2.64 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0534196, upper bound: 0.0640646
time: 2.68 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -12.4973469, -12.0152187, -12.4936247, -12.0091009, -0.2129521, 0.2031322
1: -12.4272118, -12.0745983, -12.4168205, -12.0619268, -0.1523092, 0.1294845
2: -8.9022236, -8.6535711, -8.9086809, -8.6506910, -0.1099418, 0.1134826
3: -8.4710579, -8.1805172, -8.4737434, -8.1794004, -0.1414020, 0.1429744
4: -3.6521955, -3.3691342, -3.6497169, -3.3682175, -0.1176577, 0.1148092
5: -5.3451948, -5.0374336, -5.3468561, -5.0394764, -0.1067911, 0.1104937
6: -13.7019148, -13.2569294, -13.6913290, -13.2501745, -0.1811776, 0.1641715
7: -3.5989194, -3.3305776, -3.6036847, -3.3127565, -0.1324809, 0.1195061
8: -1.8256156, -1.5309024, -1.8300529, -1.5335114, -0.1096718, 0.1166861
9: 2.9258804, 3.0950036, 2.8987217, 3.0993590, -0.0719358, 0.0947385

Time for backsubstitution: 8.35 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0518193, upper bound: 0.0638092
time: 2.74 seconds

## Relational analysis of NS_A1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0518192, upper bound: 0.0638964
time: 2.68 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -12.4979401, -12.0143433, -12.4937763, -12.0091009, -0.2133055, 0.2043395
1: -12.4274216, -12.0745392, -12.4168205, -12.0619287, -0.1526998, 0.1295605
2: -8.9021072, -8.6536016, -8.9087410, -8.6506910, -0.1099008, 0.1138400
3: -8.4709721, -8.1806393, -8.4737434, -8.1794434, -0.1414948, 0.1429206
4: -3.6528554, -3.3689485, -3.6497169, -3.3682165, -0.1193985, 0.1150445
5: -5.3451953, -5.0374312, -5.3468556, -5.0394764, -0.1067917, 0.1104980
6: -13.7023239, -13.2568140, -13.6913290, -13.2501736, -0.1819301, 0.1643173
7: -3.5992863, -3.3303092, -3.6036847, -3.3126845, -0.1329103, 0.1196702
8: -1.8258722, -1.5305343, -1.8301191, -1.5335114, -0.1098334, 0.1171674
9: 2.9249821, 3.0962882, 2.8984885, 3.0993590, -0.0725021, 0.0964028

Time for backsubstitution: 8.49 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0533963, upper bound: 0.0640099
time: 2.66 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0533963, upper bound: 0.0640961
time: 2.71 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -12.4968939, -12.0131502, -12.4940786, -12.0111685, -0.2104294, 0.2056851
1: -12.4222364, -12.0679455, -12.4217949, -12.0685844, -0.1409661, 0.1411130
2: -8.8958702, -8.6500864, -8.9149294, -8.6541748, -0.1002229, 0.1234465
3: -8.4713011, -8.1768389, -8.4735003, -8.1830788, -0.1380795, 0.1464508
4: -3.6496994, -3.3687763, -3.6522112, -3.3685760, -0.1156901, 0.1176635
5: -5.3468795, -5.0406618, -5.3451724, -5.0362453, -0.1117178, 0.1055950
6: -13.6916504, -13.2512474, -13.7015915, -13.2558584, -0.1657841, 0.1801150
7: -3.6064024, -3.3381736, -3.5963995, -3.3049626, -0.1478177, 0.1046720
8: -1.8256226, -1.5310037, -1.8300464, -1.5334108, -0.1097829, 0.1166276
9: 2.9296145, 3.1000061, 2.8947239, 3.0946174, -0.0634660, 0.1037883

Time for backsubstitution: 8.82 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0496077, upper bound: 0.0639484
time: 2.75 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0496078, upper bound: 0.0640344
time: 2.68 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -12.4974861, -12.0122747, -12.4942303, -12.0111685, -0.2107825, 0.2068224
1: -12.4224491, -12.0678864, -12.4217949, -12.0685825, -0.1407191, 0.1411889
2: -8.8960962, -8.6497803, -8.9148827, -8.6541748, -0.1001819, 0.1237828
3: -8.4712162, -8.1769619, -8.4735003, -8.1831198, -0.1380206, 0.1463971
4: -3.6503611, -3.3685896, -3.6522112, -3.3685734, -0.1165079, 0.1178988
5: -5.3468790, -5.0406604, -5.3451719, -5.0362453, -0.1117183, 0.1055922
6: -13.6920624, -13.2511330, -13.7015915, -13.2558537, -0.1653090, 0.1802608
7: -3.6067691, -3.3379056, -3.5963995, -3.3048902, -0.1482217, 0.1048357
8: -1.8258791, -1.5306368, -1.8301125, -1.5334108, -0.1099445, 0.1170563
9: 2.9287176, 3.1012912, 2.8944898, 3.0946174, -0.0640324, 0.1053548

Time for backsubstitution: 8.76 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0537204, upper bound: 0.0639767
time: 2.88 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0537202, upper bound: 0.0640637
time: 2.74 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -12.4968939, -12.0131502, -12.4936247, -12.0091009, -0.2101791, 0.2029123
1: -12.4222364, -12.0679455, -12.4168205, -12.0619268, -0.1382203, 0.1267385
2: -8.8958702, -8.6500864, -8.9086809, -8.6506910, -0.0966526, 0.1100395
3: -8.4713011, -8.1768389, -8.4737434, -8.1794004, -0.1384312, 0.1433690
4: -3.6496994, -3.3687763, -3.6497169, -3.3682175, -0.1143209, 0.1134418
5: -5.3468795, -5.0406618, -5.3468561, -5.0394764, -0.1051148, 0.1039063
6: -13.6916504, -13.2512474, -13.6913290, -13.2501745, -0.1602918, 0.1586789
7: -3.6064024, -3.3381736, -3.6036847, -3.3127565, -0.1287688, 0.1007031
8: -1.8256226, -1.5310037, -1.8300529, -1.5335114, -0.1096253, 0.1165811
9: 2.9296145, 3.1000061, 2.8987217, 3.0993590, -0.0614620, 0.0930465

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0496079, upper bound: 0.0637733
time: 2.69 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0496076, upper bound: 0.0638588
time: 2.64 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -12.4974861, -12.0122747, -12.4937763, -12.0091009, -0.2105322, 0.2041512
1: -12.4224491, -12.0678864, -12.4168205, -12.0619287, -0.1388988, 0.1268144
2: -8.8960962, -8.6497803, -8.9087410, -8.6506910, -0.0967897, 0.1104062
3: -8.4712162, -8.1769619, -8.4737434, -8.1794434, -0.1385921, 0.1433152
4: -3.6503611, -3.3685896, -3.6497169, -3.3682165, -0.1164173, 0.1136773
5: -5.3468790, -5.0406604, -5.3468556, -5.0394764, -0.1051153, 0.1039139
6: -13.6920624, -13.2511330, -13.6913290, -13.2501736, -0.1615974, 0.1588246
7: -3.6067691, -3.3379056, -3.6036847, -3.3126845, -0.1292092, 0.1008670
8: -1.8258791, -1.5306368, -1.8301191, -1.5335114, -0.1097870, 0.1170686
9: 2.9287176, 3.1012912, 2.8984885, 3.0993590, -0.0620271, 0.0947551

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0537201, upper bound: 0.0639779
time: 2.90 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0537203, upper bound: 0.0640624
time: 2.88 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.4941673, -12.0102921, -12.4980011, -12.0152187, -0.2023084, 0.2115710
1: -12.4220076, -12.0687199, -12.4272118, -12.0744038, -0.1274091, 0.1376814
2: -8.9147539, -8.6542053, -8.9022379, -8.6535711, -0.1108245, 0.0981733
3: -8.4734144, -8.1832047, -8.4710579, -8.1805553, -0.1423564, 0.1370609
4: -3.6528733, -3.3689904, -3.6521955, -3.3685312, -0.1170975, 0.1142392
5: -5.3451691, -5.0362430, -5.3451967, -5.0374336, -0.1042981, 0.1055207
6: -13.7020035, -13.2561131, -13.7019148, -13.2565517, -0.1618652, 0.1611497
7: -3.5967681, -3.3048918, -3.5989194, -3.3303089, -0.1013556, 0.1287512
8: -1.8300962, -1.5330420, -1.8258882, -1.5309024, -0.1165537, 0.1103604
9: 2.8945441, 3.0959053, 2.9249268, 3.0950036, -0.0925514, 0.0636382

Time for backsubstitution: 8.70 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640642, upper bound: 0.0533334
time: 2.64 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640637, upper bound: 0.0534191
time: 2.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.4941673, -12.0102921, -12.4975500, -12.0131502, -0.2054830, 0.2121606
1: -12.4220076, -12.0687199, -12.4222364, -12.0677490, -0.1415215, 0.1407385
2: -8.9147539, -8.6542053, -8.8960943, -8.6500864, -0.1232413, 0.1007346
3: -8.4734144, -8.1832047, -8.4713011, -8.1768780, -0.1465472, 0.1379542
4: -3.6528733, -3.3689904, -3.6496994, -3.3681736, -0.1194588, 0.1149871
5: -5.3451691, -5.0362430, -5.3468814, -5.0406618, -0.1055921, 0.1117225
6: -13.7020035, -13.2561131, -13.6916504, -13.2508736, -0.1809008, 0.1653457
7: -3.5967681, -3.3048918, -3.6064024, -3.3379054, -0.1052866, 0.1477847
8: -1.8300962, -1.5330420, -1.8258958, -1.5310037, -0.1165709, 0.1104299
9: 2.8945441, 3.0959053, 2.9286613, 3.1000061, -0.1035901, 0.0658511

Time for backsubstitution: 8.89 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A2_B1_A1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640618, upper bound: 0.0536339
time: 2.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640618, upper bound: 0.0537201
time: 2.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.4937134, -12.0082226, -12.4980011, -12.0152187, -0.2029614, 0.2146451
1: -12.4170341, -12.0620623, -12.4272118, -12.0744038, -0.1295410, 0.1523673
2: -8.9087391, -8.6503830, -8.9022379, -8.6535711, -0.1132870, 0.1104425
3: -8.4736595, -8.1795273, -8.4710579, -8.1805553, -0.1429869, 0.1413441
4: -3.6503768, -3.3686323, -3.6521955, -3.3685312, -0.1165651, 0.1178384
5: -5.3468556, -5.0394735, -5.3451967, -5.0374336, -0.1104940, 0.1067917
6: -13.6917400, -13.2504330, -13.7019148, -13.2565517, -0.1642806, 0.1812899
7: -3.6040521, -3.3126850, -3.5989194, -3.3303089, -0.1201072, 0.1324592
8: -1.8301048, -1.5331447, -1.8258882, -1.5309024, -0.1166819, 0.1103188
9: 2.8985429, 3.1006432, 2.9249268, 3.0950036, -0.0945844, 0.0742669

Time for backsubstitution: 8.76 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640954, upper bound: 0.0533102
time: 2.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640956, upper bound: 0.0533963
time: 2.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.4937134, -12.0082226, -12.4975500, -12.0131502, -0.2027104, 0.2119730
1: -12.4170341, -12.0620623, -12.4222364, -12.0677490, -0.1277205, 0.1379927
2: -8.9087391, -8.6503830, -8.8960943, -8.6500864, -0.1100125, 0.0971832
3: -8.4736595, -8.1795273, -8.4713011, -8.1768780, -0.1436015, 0.1383059
4: -3.6503768, -3.3686323, -3.6496994, -3.3681736, -0.1164762, 0.1136181
5: -5.3468556, -5.0394735, -5.3468814, -5.0406618, -0.1039033, 0.1051259
6: -13.6917400, -13.2504330, -13.6916504, -13.2508736, -0.1605687, 0.1598531
7: -3.6040521, -3.3126850, -3.6064024, -3.3379054, -0.1013402, 0.1287358
8: -1.8301048, -1.5331447, -1.8258958, -1.5310037, -0.1165244, 0.1103312
9: 2.8985429, 3.1006432, 2.9286613, 3.1000061, -0.0928469, 0.0639351

Time for backsubstitution: 8.80 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640955, upper bound: 0.0533101
time: 2.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640957, upper bound: 0.0533966
time: 2.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.4941673, -12.0102921, -12.4942303, -12.0111685, -0.2030218, 0.2044625
1: -12.4220076, -12.0687199, -12.4217949, -12.0685825, -0.1276642, 0.1267583
2: -8.9147539, -8.6542053, -8.9148827, -8.6541748, -0.1011213, 0.1016929
3: -8.4734144, -8.1832047, -8.4735003, -8.1831198, -0.1373271, 0.1370407
4: -3.6528733, -3.3689904, -3.6522112, -3.3685734, -0.1170381, 0.1142390
5: -5.3451691, -5.0362430, -5.3451719, -5.0362453, -0.1045695, 0.1045801
6: -13.7020035, -13.2561131, -13.7015915, -13.2558537, -0.1620942, 0.1603501
7: -3.5967681, -3.3048918, -3.5963995, -3.3048902, -0.1068411, 0.1063677
8: -1.8300962, -1.5330420, -1.8301125, -1.5334108, -0.1102372, 0.1107814
9: 2.8945441, 3.0959053, 2.8944898, 3.0946174, -0.0684544, 0.0703611

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A2_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640655, upper bound: 0.0541937
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640656, upper bound: 0.0542798
time: 2.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.4941673, -12.0102921, -12.4937763, -12.0091009, -0.2061965, 0.2050521
1: -12.4220076, -12.0687199, -12.4168205, -12.0619287, -0.1417766, 0.1298153
2: -8.9147539, -8.6542053, -8.9087410, -8.6506910, -0.1135381, 0.1042542
3: -8.4734144, -8.1832047, -8.4737434, -8.1794434, -0.1415179, 0.1379341
4: -3.6528733, -3.3689904, -3.6497169, -3.3682165, -0.1193994, 0.1149869
5: -5.3451691, -5.0362430, -5.3468556, -5.0394764, -0.1058635, 0.1107818
6: -13.7020035, -13.2561131, -13.6913290, -13.2501736, -0.1811299, 0.1645460
7: -3.5967681, -3.3048918, -3.6036847, -3.3126845, -0.1107723, 0.1254013
8: -1.8300962, -1.5330420, -1.8301191, -1.5335114, -0.1102544, 0.1108510
9: 2.8945441, 3.0959053, 2.8984885, 3.0993590, -0.0794930, 0.0725739

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 2459
type: A, layer: 3, pos: 2251
type: B, layer: 3, pos: 2251
type: A, layer: 3, pos: 612
type: B, layer: 3, pos: 612

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 676

## Relational analysis of NS_A2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640636, upper bound: 0.0545139
time: 2.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0640638, upper bound: 0.0546003
time: 2.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.4937134, -12.0082226, -12.4942303, -12.0111685, -0.2036741, 0.2075360
1: -12.4170341, -12.0620623, -12.4217949, -12.0685825, -0.1297959, 0.1414441
2: -8.9087391, -8.6503830, -8.9148827, -8.6541748, -0.1037015, 0.1140795
3: -8.4736595, -8.1795273, -8.4735003, -8.1831198, -0.1380007, 0.1413676
4: -3.6503768, -3.3686323, -3.6522112, -3.3685734, -0.1165074, 0.1178396
5: -5.3468556, -5.0394735, -5.3451719, -5.0362453, -0.1107777, 0.1058636
6: -13.6917400, -13.2504330, -13.7015915, -13.2558537, -0.1645094, 0.1804897
7: -3.6040521, -3.3126850, -3.5963995, -3.3048902, -0.1258383, 0.1103212
8: -1.8301048, -1.5331447, -1.8301125, -1.5334108, -0.1103656, 0.1107399
9: 2.8985429, 3.1006432, 2.8944898, 3.0946174, -0.0707553, 0.0812577

Time for backsubstitution: 8.76 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.14 + 551.79 = 606.94 seconds
