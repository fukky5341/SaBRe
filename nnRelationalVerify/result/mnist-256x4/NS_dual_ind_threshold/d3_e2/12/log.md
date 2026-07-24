## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.38366993


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374)
1: (-0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201)
2: (-0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346)
3: (-0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890)
4: (-0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673)
5: (-0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257)
6: (-0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823)
7: (-0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595)
8: (0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006)
9: (-0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.13 + 2.27 = 4.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4026718, upper bound: 0.4007881
time: 1.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4005962, upper bound: 0.4005961
time: 1.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.17 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.17
Output dim: 8, lower bound: -0.4026718, upper bound: 0.4007881
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.17
Output dim: 8, lower bound: -0.4005962, upper bound: 0.4005961

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0772326, 0.1030378, -0.1352843, 0.1904531, -0.2676857, 0.2383221
1: -0.0670717, 0.0770266, -0.1201238, 0.1458962, -0.2129679, 0.1971504
2: -0.0707251, 0.1053402, -0.1303060, 0.1809286, -0.2516538, 0.2356461
3: -0.0516543, 0.0973753, -0.0989191, 0.1781699, -0.2298242, 0.1962943
4: -0.0912210, 0.0746620, -0.1623547, 0.1264126, -0.2176336, 0.2370166
5: -0.0746264, 0.0998376, -0.1397409, 0.1723848, -0.2470112, 0.2395786
6: -0.0624522, 0.1021059, -0.1049636, 0.2513187, -0.3137709, 0.2070695
7: -0.1080345, 0.0710284, -0.1651493, 0.1279102, -0.2359447, 0.2361777
8: 0.7889467, 1.0057085, 0.5992520, 1.0262526, -0.2373059, 0.4064565
9: -0.0688367, 0.1581012, -0.1538356, 0.2383174, -0.3071542, 0.3119367

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4005729, upper bound: 0.4005729
time: 1.21 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4005729, upper bound: 0.4005962
time: 1.17 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0498724, 0.0644551, -0.1157721, 0.1633425, -0.2132150, 0.1802272
1: -0.0392483, 0.0470732, -0.1026184, 0.1241348, -0.1633831, 0.1496916
2: -0.0437750, 0.0727217, -0.1121685, 0.1565909, -0.2003659, 0.1848902
3: -0.0295466, 0.0550446, -0.0843106, 0.1525706, -0.1821172, 0.1393552
4: -0.0651141, 0.0489479, -0.1400953, 0.1098118, -0.1749259, 0.1890431
5: -0.0483807, 0.0640360, -0.1198129, 0.1492427, -0.1976233, 0.1838488
6: -0.0417086, 0.0649768, -0.0902544, 0.2119775, -0.2536861, 0.1552312
7: -0.0791256, 0.0429326, -0.1459142, 0.1087905, -0.1879161, 0.1888468
8: 0.8429493, 0.9983708, 0.6503146, 1.0186924, -0.1757431, 0.3480561
9: -0.0368403, 0.1237958, -0.1277052, 0.2143277, -0.2511680, 0.2515010

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4005962, upper bound: 0.4005729
time: 1.13 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4005962, upper bound: 0.4005962
time: 1.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.39 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 8, lower bound: -0.4005729, upper bound: 0.4005729
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 8, lower bound: -0.4005729, upper bound: 0.4005962
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 8, lower bound: -0.4005962, upper bound: 0.4005729
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.39
Output dim: 8, lower bound: -0.4005962, upper bound: 0.4005962

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0772326, 0.1030378, -0.0772326, 0.1030378, -0.1802704, 0.1802704
1: -0.0670717, 0.0770266, -0.0670717, 0.0770266, -0.1440983, 0.1440983
2: -0.0707251, 0.1053402, -0.0707251, 0.1053402, -0.1760653, 0.1760653
3: -0.0516543, 0.0973753, -0.0516543, 0.0973753, -0.1490295, 0.1490295
4: -0.0912210, 0.0746620, -0.0912210, 0.0746620, -0.1658829, 0.1658829
5: -0.0746264, 0.0998376, -0.0746264, 0.0998376, -0.1744640, 0.1744640
6: -0.0624522, 0.1021059, -0.0624522, 0.1021059, -0.1645581, 0.1645581
7: -0.1080345, 0.0710284, -0.1080345, 0.0710284, -0.1790629, 0.1790629
8: 0.7889467, 1.0057085, 0.7889467, 1.0057085, -0.2167617, 0.2167617
9: -0.0688367, 0.1581012, -0.0688367, 0.1581012, -0.2269379, 0.2269379

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3935155, upper bound: 0.3955085
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3996811, upper bound: 0.3970281
time: 1.48 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0772326, 0.1030378, -0.0498724, 0.0644551, -0.1416877, 0.1529102
1: -0.0670717, 0.0770266, -0.0392483, 0.0470732, -0.1141449, 0.1162749
2: -0.0707251, 0.1053402, -0.0437750, 0.0727217, -0.1434468, 0.1491151
3: -0.0516543, 0.0973753, -0.0295466, 0.0550446, -0.1066989, 0.1269219
4: -0.0912210, 0.0746620, -0.0651141, 0.0489479, -0.1401688, 0.1397761
5: -0.0746264, 0.0998376, -0.0483807, 0.0640360, -0.1386624, 0.1482183
6: -0.0624522, 0.1021059, -0.0417086, 0.0649768, -0.1274290, 0.1438145
7: -0.1080345, 0.0710284, -0.0791256, 0.0429326, -0.1509671, 0.1501540
8: 0.7889467, 1.0057085, 0.8429493, 0.9983708, -0.2094240, 0.1627592
9: -0.0688367, 0.1581012, -0.0368403, 0.1237958, -0.1926326, 0.1949415

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3935155, upper bound: 0.3955184
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3996811, upper bound: 0.3970309
time: 1.42 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0498724, 0.0644551, -0.0772326, 0.1030378, -0.1529102, 0.1416877
1: -0.0392483, 0.0470732, -0.0670717, 0.0770266, -0.1162749, 0.1141449
2: -0.0437750, 0.0727217, -0.0707251, 0.1053402, -0.1491151, 0.1434468
3: -0.0295466, 0.0550446, -0.0516543, 0.0973753, -0.1269219, 0.1066989
4: -0.0651141, 0.0489479, -0.0912210, 0.0746620, -0.1397761, 0.1401688
5: -0.0483807, 0.0640360, -0.0746264, 0.0998376, -0.1482183, 0.1386624
6: -0.0417086, 0.0649768, -0.0624522, 0.1021059, -0.1438145, 0.1274290
7: -0.0791256, 0.0429326, -0.1080345, 0.0710284, -0.1501540, 0.1509671
8: 0.8429493, 0.9983708, 0.7889467, 1.0057085, -0.1627592, 0.2094240
9: -0.0368403, 0.1237958, -0.0688367, 0.1581012, -0.1949415, 0.1926326

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3898260, upper bound: 0.3952460
time: 1.85 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3968568, upper bound: 0.3968132
time: 1.23 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0498724, 0.0644551, -0.0498724, 0.0644551, -0.1143275, 0.1143275
1: -0.0392483, 0.0470732, -0.0392483, 0.0470732, -0.0863215, 0.0863215
2: -0.0437750, 0.0727217, -0.0437750, 0.0727217, -0.1164967, 0.1164967
3: -0.0295466, 0.0550446, -0.0295466, 0.0550446, -0.0845912, 0.0845912
4: -0.0651141, 0.0489479, -0.0651141, 0.0489479, -0.1140619, 0.1140619
5: -0.0483807, 0.0640360, -0.0483807, 0.0640360, -0.1124167, 0.1124167
6: -0.0417086, 0.0649768, -0.0417086, 0.0649768, -0.1066854, 0.1066854
7: -0.0791256, 0.0429326, -0.0791256, 0.0429326, -0.1220582, 0.1220582
8: 0.8429493, 0.9983708, 0.8429493, 0.9983708, -0.1554215, 0.1554215
9: -0.0368403, 0.1237958, -0.0368403, 0.1237958, -0.1606361, 0.1606361

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3898260, upper bound: 0.3952916
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3968568, upper bound: 0.3968340
time: 1.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.42 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3935155, upper bound: 0.3955085
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3996811, upper bound: 0.3970281
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3935155, upper bound: 0.3955184
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3996811, upper bound: 0.3970309
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3898260, upper bound: 0.3952460
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3968568, upper bound: 0.3968132
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3898260, upper bound: 0.3952916
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 8, lower bound: -0.3968568, upper bound: 0.3968340

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0267657, 0.0295957, -0.0640839, 0.0834012, -0.1101669, 0.0936795
1: -0.0183217, 0.0138815, -0.0541517, 0.0627987, -0.0811204, 0.0680332
2: -0.0145404, 0.0377148, -0.0576013, 0.0886853, -0.1032256, 0.0953161
3: -0.0097822, 0.0188234, -0.0403526, 0.0795347, -0.0893168, 0.0591760
4: -0.0268445, 0.0177344, -0.0777975, 0.0618019, -0.0886464, 0.0955319
5: -0.0157390, 0.0350826, -0.0604838, 0.0831121, -0.0988510, 0.0955663
6: -0.0202868, 0.0221805, -0.0522418, 0.0786235, -0.0989102, 0.0744223
7: -0.0454834, 0.0118032, -0.0948396, 0.0571339, -0.1026173, 0.1066428
8: 0.9245001, 0.9927343, 0.8207357, 1.0021104, -0.0776103, 0.1719986
9: -0.0076698, 0.0597213, -0.0523945, 0.1402816, -0.1479514, 0.1121158

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3946066, upper bound: 0.3946065
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3946066, upper bound: 0.3996378
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0446593, 0.0508000, -0.0771546, 0.1029208, -0.1475801, 0.1279547
1: -0.0350039, 0.0393280, -0.0669963, 0.0769414, -0.1119452, 0.1063243
2: -0.0350894, 0.0619035, -0.0706475, 0.1052411, -0.1403306, 0.1325510
3: -0.0230137, 0.0475617, -0.0515872, 0.0972708, -0.1202845, 0.0991489
4: -0.0541894, 0.0419273, -0.0911402, 0.0745860, -0.1287754, 0.1330675
5: -0.0359531, 0.0554233, -0.0745415, 0.0997392, -0.1356923, 0.1299648
6: -0.0356649, 0.0381099, -0.0623923, 0.1019619, -0.1376268, 0.1005023
7: -0.0723669, 0.0366911, -0.1079573, 0.0709467, -0.1433137, 0.1446484
8: 0.8788870, 0.9968827, 0.7891396, 1.0056851, -0.1267981, 0.2077430
9: -0.0241333, 0.1080855, -0.0687385, 0.1579956, -0.1821289, 0.1768239

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3996378, upper bound: 0.3948105
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3996378, upper bound: 0.3948105
time: 1.70 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0267657, 0.0295957, -0.0424050, 0.0490221, -0.0757878, 0.0720006
1: -0.0183217, 0.0138815, -0.0310263, 0.0366849, -0.0550066, 0.0449078
2: -0.0145404, 0.0377148, -0.0334184, 0.0621435, -0.0766839, 0.0711332
3: -0.0097822, 0.0188234, -0.0226233, 0.0406119, -0.0503941, 0.0414467
4: -0.0268445, 0.0177344, -0.0540246, 0.0402133, -0.0670578, 0.0717590
5: -0.0157390, 0.0350826, -0.0373862, 0.0516148, -0.0673538, 0.0724688
6: -0.0202868, 0.0221805, -0.0341766, 0.0481703, -0.0684571, 0.0563572
7: -0.0454834, 0.0118032, -0.0679580, 0.0342842, -0.0797676, 0.0797612
8: 0.9245001, 0.9927343, 0.8686637, 0.9967324, -0.0722322, 0.1240705
9: -0.0076698, 0.0597213, -0.0244982, 0.1075927, -0.1152625, 0.0842194

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3933110, upper bound: 0.3898060
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3933110, upper bound: 0.3955184
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0446593, 0.0508000, -0.0498272, 0.0643608, -0.1090200, 0.1006272
1: -0.0350039, 0.0393280, -0.0391986, 0.0470103, -0.0820142, 0.0785266
2: -0.0350894, 0.0619035, -0.0437111, 0.0726599, -0.1077493, 0.1056145
3: -0.0230137, 0.0475617, -0.0295050, 0.0549540, -0.0779676, 0.0770667
4: -0.0541894, 0.0419273, -0.0650458, 0.0488958, -0.1030852, 0.1069731
5: -0.0359531, 0.0554233, -0.0483128, 0.0639598, -0.0999128, 0.1037361
6: -0.0356649, 0.0381099, -0.0416630, 0.0648674, -0.1005323, 0.0797729
7: -0.0723669, 0.0366911, -0.0790598, 0.0428826, -0.1152495, 0.1157509
8: 0.8788870, 0.9968827, 0.8431107, 0.9983573, -0.1194703, 0.1537719
9: -0.0241333, 0.1080855, -0.0367633, 0.1237006, -0.1478339, 0.1448488

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3980816, upper bound: 0.3899877
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3980816, upper bound: 0.3970309
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0172581, 0.0298437, -0.0640839, 0.0834012, -0.1006593, 0.0939276
1: -0.0151399, 0.0083981, -0.0541517, 0.0627987, -0.0779385, 0.0625498
2: -0.0109017, 0.0294701, -0.0576013, 0.0886853, -0.0995869, 0.0870715
3: -0.0072293, 0.0169131, -0.0403526, 0.0795347, -0.0867639, 0.0572657
4: -0.0164731, 0.0124494, -0.0777975, 0.0618019, -0.0782749, 0.0902469
5: -0.0136928, 0.0260826, -0.0604838, 0.0831121, -0.0968049, 0.0865664
6: -0.0188366, 0.0187052, -0.0522418, 0.0786235, -0.0974600, 0.0709470
7: -0.0414973, 0.0064738, -0.0948396, 0.0571339, -0.0986312, 0.1013134
8: 0.9403118, 0.9927152, 0.8207357, 1.0021104, -0.0617986, 0.1719795
9: -0.0072902, 0.0393496, -0.0523945, 0.1402816, -0.1475718, 0.0917442

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3898060, upper bound: 0.3933110
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3898060, upper bound: 0.3980816
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0326248, 0.0337374, -0.0771546, 0.1029208, -0.1355456, 0.1108920
1: -0.0228376, 0.0182326, -0.0669963, 0.0769414, -0.0997790, 0.0852289
2: -0.0199897, 0.0442654, -0.0706475, 0.1052411, -0.1252308, 0.1149129
3: -0.0138262, 0.0234056, -0.0515872, 0.0972708, -0.1110970, 0.0749928
4: -0.0362488, 0.0248295, -0.0911402, 0.0745860, -0.1108347, 0.1159696
5: -0.0216912, 0.0403950, -0.0745415, 0.0997392, -0.1214304, 0.1149365
6: -0.0245827, 0.0256339, -0.0623923, 0.1019619, -0.1265446, 0.0880262
7: -0.0506919, 0.0207384, -0.1079573, 0.0709467, -0.1216386, 0.1286956
8: 0.9117983, 0.9946759, 0.7891396, 1.0056851, -0.0938868, 0.2055362
9: -0.0089768, 0.0776040, -0.0687385, 0.1579956, -0.1669724, 0.1463425

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3955184, upper bound: 0.3935154
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3955184, upper bound: 0.3935154
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0172581, 0.0298437, -0.0424050, 0.0490221, -0.0662802, 0.0722487
1: -0.0151399, 0.0083981, -0.0310263, 0.0366849, -0.0518248, 0.0394244
2: -0.0109017, 0.0294701, -0.0334184, 0.0621435, -0.0730452, 0.0628885
3: -0.0072293, 0.0169131, -0.0226233, 0.0406119, -0.0478412, 0.0395364
4: -0.0164731, 0.0124494, -0.0540246, 0.0402133, -0.0566864, 0.0664740
5: -0.0136928, 0.0260826, -0.0373862, 0.0516148, -0.0653077, 0.0634688
6: -0.0188366, 0.0187052, -0.0341766, 0.0481703, -0.0670069, 0.0528819
7: -0.0414973, 0.0064738, -0.0679580, 0.0342842, -0.0757814, 0.0744318
8: 0.9403118, 0.9927152, 0.8686637, 0.9967324, -0.0564206, 0.1240515
9: -0.0072902, 0.0393496, -0.0244982, 0.1075927, -0.1148829, 0.0638478

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3896137, upper bound: 0.3895962
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3896137, upper bound: 0.3952916
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0326248, 0.0337374, -0.0498272, 0.0643608, -0.0969856, 0.0835646
1: -0.0228376, 0.0182326, -0.0391986, 0.0470103, -0.0698479, 0.0574311
2: -0.0199897, 0.0442654, -0.0437111, 0.0726599, -0.0926495, 0.0879764
3: -0.0138262, 0.0234056, -0.0295050, 0.0549540, -0.0687801, 0.0529106
4: -0.0362488, 0.0248295, -0.0650458, 0.0488958, -0.0851446, 0.0898752
5: -0.0216912, 0.0403950, -0.0483128, 0.0639598, -0.0856510, 0.0887078
6: -0.0245827, 0.0256339, -0.0416630, 0.0648674, -0.0894501, 0.0672969
7: -0.0506919, 0.0207384, -0.0790598, 0.0428826, -0.0935745, 0.0997982
8: 0.9117983, 0.9946759, 0.8431107, 0.9983573, -0.0865590, 0.1515651
9: -0.0089768, 0.0776040, -0.0367633, 0.1237006, -0.1326774, 0.1143674

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3953301, upper bound: 0.3898161
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3953301, upper bound: 0.3968340
time: 1.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.64 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3946066, upper bound: 0.3946065
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3946066, upper bound: 0.3996378
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3996378, upper bound: 0.3948105
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3996378, upper bound: 0.3948105
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3933110, upper bound: 0.3898060
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3933110, upper bound: 0.3955184
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3980816, upper bound: 0.3899877
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3980816, upper bound: 0.3970309
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3898060, upper bound: 0.3933110
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3898060, upper bound: 0.3980816
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3955184, upper bound: 0.3935154
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3955184, upper bound: 0.3935154
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3896137, upper bound: 0.3895962
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3896137, upper bound: 0.3952916
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3953301, upper bound: 0.3898161
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.64
Output dim: 8, lower bound: -0.3953301, upper bound: 0.3968340

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0267657, 0.0295957, -0.0267657, 0.0295957, -0.0563613, 0.0563613
1: -0.0183217, 0.0138815, -0.0183217, 0.0138815, -0.0322032, 0.0322032
2: -0.0145404, 0.0377148, -0.0145404, 0.0377148, -0.0522552, 0.0522552
3: -0.0097822, 0.0188234, -0.0097822, 0.0188234, -0.0286056, 0.0286056
4: -0.0268445, 0.0177344, -0.0268445, 0.0177344, -0.0445789, 0.0445789
5: -0.0157390, 0.0350826, -0.0157390, 0.0350826, -0.0508216, 0.0508216
6: -0.0202868, 0.0221805, -0.0202868, 0.0221805, -0.0424673, 0.0424673
7: -0.0454834, 0.0118032, -0.0454834, 0.0118032, -0.0572865, 0.0572865
8: 0.9245001, 0.9927343, 0.9245001, 0.9927343, -0.0682341, 0.0682341
9: -0.0076698, 0.0597213, -0.0076698, 0.0597213, -0.0673911, 0.0673911

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3864332, upper bound: 0.3812138
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3809564
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0267657, 0.0295957, -0.0446593, 0.0508000, -0.0775657, 0.0742549
1: -0.0183217, 0.0138815, -0.0350039, 0.0393280, -0.0576497, 0.0488854
2: -0.0145404, 0.0377148, -0.0350894, 0.0619035, -0.0764438, 0.0728042
3: -0.0097822, 0.0188234, -0.0230137, 0.0475617, -0.0573439, 0.0418371
4: -0.0268445, 0.0177344, -0.0541894, 0.0419273, -0.0687718, 0.0719238
5: -0.0157390, 0.0350826, -0.0359531, 0.0554233, -0.0711622, 0.0710357
6: -0.0202868, 0.0221805, -0.0356649, 0.0381099, -0.0583967, 0.0578455
7: -0.0454834, 0.0118032, -0.0723669, 0.0366911, -0.0821745, 0.0841701
8: 0.9245001, 0.9927343, 0.8788870, 0.9968827, -0.0723826, 0.1138473
9: -0.0076698, 0.0597213, -0.0241333, 0.1080855, -0.1157553, 0.0838546

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3864332, upper bound: 0.3812138
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3861938
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0446593, 0.0508000, -0.0267657, 0.0295957, -0.0742549, 0.0775657
1: -0.0350039, 0.0393280, -0.0183217, 0.0138815, -0.0488854, 0.0576497
2: -0.0350894, 0.0619035, -0.0145404, 0.0377148, -0.0728042, 0.0764438
3: -0.0230137, 0.0475617, -0.0097822, 0.0188234, -0.0418371, 0.0573439
4: -0.0541894, 0.0419273, -0.0268445, 0.0177344, -0.0719238, 0.0687718
5: -0.0359531, 0.0554233, -0.0157390, 0.0350826, -0.0710357, 0.0711622
6: -0.0356649, 0.0381099, -0.0202868, 0.0221805, -0.0578455, 0.0583967
7: -0.0723669, 0.0366911, -0.0454834, 0.0118032, -0.0841701, 0.0821745
8: 0.8788870, 0.9968827, 0.9245001, 0.9927343, -0.1138473, 0.0723826
9: -0.0241333, 0.1080855, -0.0076698, 0.0597213, -0.0838546, 0.1157553

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3931998, upper bound: 0.3865507
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3914334, upper bound: 0.3864523
time: 2.43 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0446593, 0.0508000, -0.0446593, 0.0508000, -0.0954593, 0.0954593
1: -0.0350039, 0.0393280, -0.0350039, 0.0393280, -0.0743319, 0.0743319
2: -0.0350894, 0.0619035, -0.0350894, 0.0619035, -0.0969929, 0.0969929
3: -0.0230137, 0.0475617, -0.0230137, 0.0475617, -0.0705754, 0.0705754
4: -0.0541894, 0.0419273, -0.0541894, 0.0419273, -0.0961167, 0.0961167
5: -0.0359531, 0.0554233, -0.0359531, 0.0554233, -0.0913763, 0.0913763
6: -0.0356649, 0.0381099, -0.0356649, 0.0381099, -0.0737749, 0.0737749
7: -0.0723669, 0.0366911, -0.0723669, 0.0366911, -0.1090581, 0.1090581
8: 0.8788870, 0.9968827, 0.8788870, 0.9968827, -0.1179957, 0.1179957
9: -0.0241333, 0.1080855, -0.0241333, 0.1080855, -0.1322187, 0.1322187

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3931998, upper bound: 0.3929456
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3914334, upper bound: 0.3864523
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0267657, 0.0295957, -0.0172581, 0.0298437, -0.0566094, 0.0468538
1: -0.0183217, 0.0138815, -0.0151399, 0.0083981, -0.0267198, 0.0290214
2: -0.0145404, 0.0377148, -0.0109017, 0.0294701, -0.0440105, 0.0486165
3: -0.0097822, 0.0188234, -0.0072293, 0.0169131, -0.0266953, 0.0260527
4: -0.0268445, 0.0177344, -0.0164731, 0.0124494, -0.0392939, 0.0342075
5: -0.0157390, 0.0350826, -0.0136928, 0.0260826, -0.0418216, 0.0487754
6: -0.0202868, 0.0221805, -0.0188366, 0.0187052, -0.0389920, 0.0410171
7: -0.0454834, 0.0118032, -0.0414973, 0.0064738, -0.0519572, 0.0533004
8: 0.9245001, 0.9927343, 0.9403118, 0.9927152, -0.0682151, 0.0524225
9: -0.0076698, 0.0597213, -0.0072902, 0.0393496, -0.0470194, 0.0670115

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3850013, upper bound: 0.3768528
time: 1.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3798226, upper bound: 0.3765986
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0267657, 0.0295957, -0.0326248, 0.0337374, -0.0605031, 0.0622205
1: -0.0183217, 0.0138815, -0.0228376, 0.0182326, -0.0365543, 0.0367192
2: -0.0145404, 0.0377148, -0.0199897, 0.0442654, -0.0588057, 0.0577045
3: -0.0097822, 0.0188234, -0.0138262, 0.0234056, -0.0331877, 0.0326496
4: -0.0268445, 0.0177344, -0.0362488, 0.0248295, -0.0516740, 0.0539831
5: -0.0157390, 0.0350826, -0.0216912, 0.0403950, -0.0561340, 0.0567738
6: -0.0202868, 0.0221805, -0.0245827, 0.0256339, -0.0459206, 0.0467633
7: -0.0454834, 0.0118032, -0.0506919, 0.0207384, -0.0662217, 0.0624951
8: 0.9245001, 0.9927343, 0.9117983, 0.9946759, -0.0701758, 0.0809360
9: -0.0076698, 0.0597213, -0.0089768, 0.0776040, -0.0852739, 0.0686981

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3850013, upper bound: 0.3827300
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3798226, upper bound: 0.3824547
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0446593, 0.0508000, -0.0172581, 0.0298437, -0.0745030, 0.0680581
1: -0.0350039, 0.0393280, -0.0151399, 0.0083981, -0.0434020, 0.0544678
2: -0.0350894, 0.0619035, -0.0109017, 0.0294701, -0.0645596, 0.0728052
3: -0.0230137, 0.0475617, -0.0072293, 0.0169131, -0.0399268, 0.0547910
4: -0.0541894, 0.0419273, -0.0164731, 0.0124494, -0.0666388, 0.0584004
5: -0.0359531, 0.0554233, -0.0136928, 0.0260826, -0.0620357, 0.0691161
6: -0.0356649, 0.0381099, -0.0188366, 0.0187052, -0.0543701, 0.0569465
7: -0.0723669, 0.0366911, -0.0414973, 0.0064738, -0.0788408, 0.0781884
8: 0.8788870, 0.9968827, 0.9403118, 0.9927152, -0.1138282, 0.0565709
9: -0.0241333, 0.1080855, -0.0072902, 0.0393496, -0.0634829, 0.1153757

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3912773, upper bound: 0.3817161
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3900790, upper bound: 0.3816544
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0446593, 0.0508000, -0.0326248, 0.0337374, -0.0783967, 0.0834249
1: -0.0350039, 0.0393280, -0.0228376, 0.0182326, -0.0532364, 0.0621656
2: -0.0350894, 0.0619035, -0.0199897, 0.0442654, -0.0793548, 0.0818931
3: -0.0230137, 0.0475617, -0.0138262, 0.0234056, -0.0464192, 0.0613879
4: -0.0541894, 0.0419273, -0.0362488, 0.0248295, -0.0790188, 0.0781761
5: -0.0359531, 0.0554233, -0.0216912, 0.0403950, -0.0763481, 0.0771145
6: -0.0356649, 0.0381099, -0.0245827, 0.0256339, -0.0612988, 0.0626926
7: -0.0723669, 0.0366911, -0.0506919, 0.0207384, -0.0931053, 0.0873830
8: 0.8788870, 0.9968827, 0.9117983, 0.9946759, -0.1157889, 0.0850844
9: -0.0241333, 0.1080855, -0.0089768, 0.0776040, -0.1017373, 0.1170622

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3912773, upper bound: 0.3891668
time: 1.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3900790, upper bound: 0.3890839
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0172581, 0.0298437, -0.0267657, 0.0295957, -0.0468538, 0.0566094
1: -0.0151399, 0.0083981, -0.0183217, 0.0138815, -0.0290214, 0.0267198
2: -0.0109017, 0.0294701, -0.0145404, 0.0377148, -0.0486165, 0.0440105
3: -0.0072293, 0.0169131, -0.0097822, 0.0188234, -0.0260527, 0.0266953
4: -0.0164731, 0.0124494, -0.0268445, 0.0177344, -0.0342075, 0.0392939
5: -0.0136928, 0.0260826, -0.0157390, 0.0350826, -0.0487754, 0.0418216
6: -0.0188366, 0.0187052, -0.0202868, 0.0221805, -0.0410171, 0.0389920
7: -0.0414973, 0.0064738, -0.0454834, 0.0118032, -0.0533004, 0.0519572
8: 0.9403118, 0.9927152, 0.9245001, 0.9927343, -0.0524225, 0.0682151
9: -0.0072902, 0.0393496, -0.0076698, 0.0597213, -0.0670115, 0.0470194

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3813432, upper bound: 0.3799972
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3765986, upper bound: 0.3798226
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0172581, 0.0298437, -0.0446593, 0.0508000, -0.0680581, 0.0745030
1: -0.0151399, 0.0083981, -0.0350039, 0.0393280, -0.0544678, 0.0434020
2: -0.0109017, 0.0294701, -0.0350894, 0.0619035, -0.0728052, 0.0645596
3: -0.0072293, 0.0169131, -0.0230137, 0.0475617, -0.0547910, 0.0399268
4: -0.0164731, 0.0124494, -0.0541894, 0.0419273, -0.0584004, 0.0666388
5: -0.0136928, 0.0260826, -0.0359531, 0.0554233, -0.0691161, 0.0620357
6: -0.0188366, 0.0187052, -0.0356649, 0.0381099, -0.0569465, 0.0543701
7: -0.0414973, 0.0064738, -0.0723669, 0.0366911, -0.0781884, 0.0788408
8: 0.9403118, 0.9927152, 0.8788870, 0.9968827, -0.0565709, 0.1138282
9: -0.0072902, 0.0393496, -0.0241333, 0.1080855, -0.1153757, 0.0634829

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3813432, upper bound: 0.3849989
time: 1.38 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3765986, upper bound: 0.3847507
time: 1.26 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0326248, 0.0337374, -0.0267657, 0.0295957, -0.0622205, 0.0605031
1: -0.0228376, 0.0182326, -0.0183217, 0.0138815, -0.0367192, 0.0365543
2: -0.0199897, 0.0442654, -0.0145404, 0.0377148, -0.0577045, 0.0588057
3: -0.0138262, 0.0234056, -0.0097822, 0.0188234, -0.0326496, 0.0331877
4: -0.0362488, 0.0248295, -0.0268445, 0.0177344, -0.0539831, 0.0516740
5: -0.0216912, 0.0403950, -0.0157390, 0.0350826, -0.0567738, 0.0561340
6: -0.0245827, 0.0256339, -0.0202868, 0.0221805, -0.0467633, 0.0459206
7: -0.0506919, 0.0207384, -0.0454834, 0.0118032, -0.0624951, 0.0662217
8: 0.9117983, 0.9946759, 0.9245001, 0.9927343, -0.0809360, 0.0701758
9: -0.0089768, 0.0776040, -0.0076698, 0.0597213, -0.0686981, 0.0852739

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3866924, upper bound: 0.3801647
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3824546, upper bound: 0.3800133
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0326248, 0.0337374, -0.0446593, 0.0508000, -0.0834249, 0.0783967
1: -0.0228376, 0.0182326, -0.0350039, 0.0393280, -0.0621656, 0.0532364
2: -0.0199897, 0.0442654, -0.0350894, 0.0619035, -0.0818931, 0.0793548
3: -0.0138262, 0.0234056, -0.0230137, 0.0475617, -0.0613879, 0.0464192
4: -0.0362488, 0.0248295, -0.0541894, 0.0419273, -0.0781761, 0.0790188
5: -0.0216912, 0.0403950, -0.0359531, 0.0554233, -0.0771145, 0.0763481
6: -0.0245827, 0.0256339, -0.0356649, 0.0381099, -0.0626926, 0.0612988
7: -0.0506919, 0.0207384, -0.0723669, 0.0366911, -0.0873830, 0.0931053
8: 0.9117983, 0.9946759, 0.8788870, 0.9968827, -0.0850844, 0.1157889
9: -0.0089768, 0.0776040, -0.0241333, 0.1080855, -0.1170622, 0.1017373

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3866924, upper bound: 0.3865905
time: 1.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3824547, upper bound: 0.3864161
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0172581, 0.0298437, -0.0172581, 0.0298437, -0.0471018, 0.0471018
1: -0.0151399, 0.0083981, -0.0151399, 0.0083981, -0.0235380, 0.0235380
2: -0.0109017, 0.0294701, -0.0109017, 0.0294701, -0.0403718, 0.0403718
3: -0.0072293, 0.0169131, -0.0072293, 0.0169131, -0.0241424, 0.0241424
4: -0.0164731, 0.0124494, -0.0164731, 0.0124494, -0.0289225, 0.0289225
5: -0.0136928, 0.0260826, -0.0136928, 0.0260826, -0.0397754, 0.0397754
6: -0.0188366, 0.0187052, -0.0188366, 0.0187052, -0.0375418, 0.0375418
7: -0.0414973, 0.0064738, -0.0414973, 0.0064738, -0.0479711, 0.0479711
8: 0.9403118, 0.9927152, 0.9403118, 0.9927152, -0.0524035, 0.0524035
9: -0.0072902, 0.0393496, -0.0072902, 0.0393496, -0.0466398, 0.0466398

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3812058, upper bound: 0.3766269
time: 1.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3764211, upper bound: 0.3764188
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0172581, 0.0298437, -0.0326248, 0.0337374, -0.0509955, 0.0624686
1: -0.0151399, 0.0083981, -0.0228376, 0.0182326, -0.0333724, 0.0312357
2: -0.0109017, 0.0294701, -0.0199897, 0.0442654, -0.0551670, 0.0494598
3: -0.0072293, 0.0169131, -0.0138262, 0.0234056, -0.0306348, 0.0307393
4: -0.0164731, 0.0124494, -0.0362488, 0.0248295, -0.0413025, 0.0486982
5: -0.0136928, 0.0260826, -0.0216912, 0.0403950, -0.0540878, 0.0477739
6: -0.0188366, 0.0187052, -0.0245827, 0.0256339, -0.0444704, 0.0432879
7: -0.0414973, 0.0064738, -0.0506919, 0.0207384, -0.0622356, 0.0571657
8: 0.9403118, 0.9927152, 0.9117983, 0.9946759, -0.0543641, 0.0809169
9: -0.0072902, 0.0393496, -0.0089768, 0.0776040, -0.0848943, 0.0483264

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3812058, upper bound: 0.3824672
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3764211, upper bound: 0.3822139
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0326248, 0.0337374, -0.0172581, 0.0298437, -0.0624686, 0.0509955
1: -0.0228376, 0.0182326, -0.0151399, 0.0083981, -0.0312357, 0.0333724
2: -0.0199897, 0.0442654, -0.0109017, 0.0294701, -0.0494598, 0.0551670
3: -0.0138262, 0.0234056, -0.0072293, 0.0169131, -0.0307393, 0.0306348
4: -0.0362488, 0.0248295, -0.0164731, 0.0124494, -0.0486982, 0.0413025
5: -0.0216912, 0.0403950, -0.0136928, 0.0260826, -0.0477739, 0.0540878
6: -0.0245827, 0.0256339, -0.0188366, 0.0187052, -0.0432879, 0.0444704
7: -0.0506919, 0.0207384, -0.0414973, 0.0064738, -0.0571657, 0.0622356
8: 0.9117983, 0.9946759, 0.9403118, 0.9927152, -0.0809169, 0.0543641
9: -0.0089768, 0.0776040, -0.0072902, 0.0393496, -0.0483264, 0.0848943

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3865818, upper bound: 0.3768233
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3766304
time: 2.03 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0326248, 0.0337374, -0.0326248, 0.0337374, -0.0663622, 0.0663622
1: -0.0228376, 0.0182326, -0.0228376, 0.0182326, -0.0410702, 0.0410702
2: -0.0199897, 0.0442654, -0.0199897, 0.0442654, -0.0642550, 0.0642550
3: -0.0138262, 0.0234056, -0.0138262, 0.0234056, -0.0372317, 0.0372317
4: -0.0362488, 0.0248295, -0.0362488, 0.0248295, -0.0610782, 0.0610782
5: -0.0216912, 0.0403950, -0.0216912, 0.0403950, -0.0620862, 0.0620862
6: -0.0245827, 0.0256339, -0.0245827, 0.0256339, -0.0502166, 0.0502166
7: -0.0506919, 0.0207384, -0.0506919, 0.0207384, -0.0714303, 0.0714303
8: 0.9117983, 0.9946759, 0.9117983, 0.9946759, -0.0828776, 0.0828776
9: -0.0089768, 0.0776040, -0.0089768, 0.0776040, -0.0865808, 0.0865808

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3865818, upper bound: 0.3838869
time: 2.04 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3836867
time: 1.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.84 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3864332, upper bound: 0.3812138
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3809564
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3864332, upper bound: 0.3812138
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3861938
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3931998, upper bound: 0.3865507
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3914334, upper bound: 0.3864523
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3931998, upper bound: 0.3929456
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3914334, upper bound: 0.3864523
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3850013, upper bound: 0.3768528
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3798226, upper bound: 0.3765986
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3850013, upper bound: 0.3827300
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3798226, upper bound: 0.3824547
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3912773, upper bound: 0.3817161
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3900790, upper bound: 0.3816544
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3912773, upper bound: 0.3891668
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3900790, upper bound: 0.3890839
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3813432, upper bound: 0.3799972
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3765986, upper bound: 0.3798226
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3813432, upper bound: 0.3849989
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3765986, upper bound: 0.3847507
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3866924, upper bound: 0.3801647
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3824546, upper bound: 0.3800133
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3866924, upper bound: 0.3865905
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3824547, upper bound: 0.3864161
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3812058, upper bound: 0.3766269
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3764211, upper bound: 0.3764188
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3812058, upper bound: 0.3824672
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3764211, upper bound: 0.3822139
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3865818, upper bound: 0.3768233
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3766304
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3865818, upper bound: 0.3838869
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.84
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3836867

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0095893, 0.0232121, -0.0267657, 0.0295957, -0.0391850, 0.0499778
1: -0.0114111, 0.0043119, -0.0183217, 0.0138815, -0.0252927, 0.0226336
2: -0.0040541, 0.0239550, -0.0145404, 0.0377148, -0.0417689, 0.0384954
3: -0.0047653, 0.0129477, -0.0097822, 0.0188234, -0.0235887, 0.0227298
4: -0.0114221, 0.0055739, -0.0268445, 0.0177344, -0.0291565, 0.0324184
5: -0.0061224, 0.0193764, -0.0157390, 0.0350826, -0.0412050, 0.0351154
6: -0.0147106, 0.0174241, -0.0202868, 0.0221805, -0.0368911, 0.0377109
7: -0.0365771, 0.0032983, -0.0454834, 0.0118032, -0.0483803, 0.0487816
8: 0.9454601, 0.9908453, 0.9245001, 0.9927343, -0.0472741, 0.0663452
9: -0.0061706, 0.0287391, -0.0076698, 0.0597213, -0.0658918, 0.0364089

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3809564
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3809564
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0095893, 0.0232121, -0.0446593, 0.0508000, -0.0603893, 0.0678714
1: -0.0114111, 0.0043119, -0.0350039, 0.0393280, -0.0507391, 0.0393158
2: -0.0040541, 0.0239550, -0.0350894, 0.0619035, -0.0659575, 0.0590445
3: -0.0047653, 0.0129477, -0.0230137, 0.0475617, -0.0523270, 0.0359613
4: -0.0114221, 0.0055739, -0.0541894, 0.0419273, -0.0533495, 0.0597633
5: -0.0061224, 0.0193764, -0.0359531, 0.0554233, -0.0615457, 0.0553295
6: -0.0147106, 0.0174241, -0.0356649, 0.0381099, -0.0528205, 0.0530890
7: -0.0365771, 0.0032983, -0.0723669, 0.0366911, -0.0732682, 0.0756652
8: 0.9454601, 0.9908453, 0.8788870, 0.9968827, -0.0514225, 0.1119583
9: -0.0061706, 0.0287391, -0.0241333, 0.1080855, -0.1142560, 0.0528724

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3778105, upper bound: 0.3791256
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3777142, upper bound: 0.3778668
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0077039, 0.0230440, -0.0373747, 0.0360188, -0.0437227, 0.0604187
1: -0.0108323, 0.0042899, -0.0260123, 0.0273732, -0.0382055, 0.0303022
2: -0.0032717, 0.0228667, -0.0248163, 0.0514424, -0.0547141, 0.0476830
3: -0.0044244, 0.0125478, -0.0156397, 0.0310336, -0.0354579, 0.0281875
4: -0.0104294, 0.0034819, -0.0436816, 0.0315868, -0.0420161, 0.0471635
5: -0.0040877, 0.0176446, -0.0240181, 0.0446941, -0.0487818, 0.0416626
6: -0.0153521, 0.0170517, -0.0269493, 0.0282676, -0.0436197, 0.0440010
7: -0.0357896, 0.0039835, -0.0591240, 0.0271309, -0.0629205, 0.0631075
8: 0.9467671, 0.9911283, 0.8986889, 0.9942101, -0.0474430, 0.0924394
9: -0.0066507, 0.0261685, -0.0131521, 0.0916385, -0.0982891, 0.0393207

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3725845, upper bound: 0.3789321
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3724938, upper bound: 0.3775928
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0265467, 0.0303759, -0.0267657, 0.0295957, -0.0561424, 0.0571415
1: -0.0182748, 0.0140857, -0.0183217, 0.0138815, -0.0321563, 0.0324074
2: -0.0148019, 0.0378118, -0.0145404, 0.0377148, -0.0525167, 0.0523522
3: -0.0109130, 0.0189780, -0.0097822, 0.0188234, -0.0297364, 0.0287602
4: -0.0267555, 0.0189954, -0.0268445, 0.0177344, -0.0444899, 0.0458399
5: -0.0179390, 0.0348805, -0.0157390, 0.0350826, -0.0530216, 0.0506195
6: -0.0211310, 0.0221200, -0.0202868, 0.0221805, -0.0433116, 0.0424068
7: -0.0453460, 0.0120630, -0.0454834, 0.0118032, -0.0571492, 0.0575464
8: 0.9249541, 0.9936405, 0.9245001, 0.9927343, -0.0677802, 0.0691404
9: -0.0079011, 0.0596590, -0.0076698, 0.0597213, -0.0676224, 0.0673288

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3791256, upper bound: 0.3778105
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3789322, upper bound: 0.3725845
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0267401, 0.0329629, -0.0227014, 0.0287954, -0.0555355, 0.0556643
1: -0.0189058, 0.0146470, -0.0164858, 0.0113226, -0.0302284, 0.0311328
2: -0.0161642, 0.0386078, -0.0123476, 0.0334565, -0.0496207, 0.0509554
3: -0.0111078, 0.0198881, -0.0078992, 0.0177349, -0.0288427, 0.0277872
4: -0.0275032, 0.0190706, -0.0210384, 0.0150468, -0.0425501, 0.0401091
5: -0.0185541, 0.0350990, -0.0142180, 0.0311621, -0.0497162, 0.0493170
6: -0.0234157, 0.0222461, -0.0189098, 0.0200782, -0.0434939, 0.0411559
7: -0.0461166, 0.0125291, -0.0430554, 0.0074679, -0.0535845, 0.0555846
8: 0.9244128, 0.9948584, 0.9333054, 0.9923183, -0.0679055, 0.0615531
9: -0.0086460, 0.0601269, -0.0073098, 0.0478696, -0.0565157, 0.0674367

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3778668, upper bound: 0.3777141
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3775929, upper bound: 0.3724937
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0265467, 0.0303759, -0.0446593, 0.0508000, -0.0773468, 0.0750351
1: -0.0182748, 0.0140857, -0.0350039, 0.0393280, -0.0576028, 0.0490896
2: -0.0148019, 0.0378118, -0.0350894, 0.0619035, -0.0767054, 0.0729012
3: -0.0109130, 0.0189780, -0.0230137, 0.0475617, -0.0584747, 0.0419917
4: -0.0267555, 0.0189954, -0.0541894, 0.0419273, -0.0686828, 0.0731848
5: -0.0179390, 0.0348805, -0.0359531, 0.0554233, -0.0733623, 0.0708336
6: -0.0211310, 0.0221200, -0.0356649, 0.0381099, -0.0592410, 0.0577849
7: -0.0453460, 0.0120630, -0.0723669, 0.0366911, -0.0820371, 0.0844300
8: 0.9249541, 0.9936405, 0.8788870, 0.9968827, -0.0719286, 0.1147535
9: -0.0079011, 0.0596590, -0.0241333, 0.1080855, -0.1159866, 0.0837923

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3928508, upper bound: 0.3928391
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3928508, upper bound: 0.3928392
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0267401, 0.0329629, -0.0384018, 0.0389229, -0.0656630, 0.0713646
1: -0.0189058, 0.0146470, -0.0276264, 0.0300386, -0.0489444, 0.0422734
2: -0.0161642, 0.0386078, -0.0268419, 0.0528977, -0.0690619, 0.0654498
3: -0.0111078, 0.0198881, -0.0173109, 0.0349044, -0.0460122, 0.0371989
4: -0.0275032, 0.0190706, -0.0454982, 0.0341106, -0.0616138, 0.0645689
5: -0.0185541, 0.0350990, -0.0273566, 0.0457029, -0.0642570, 0.0624556
6: -0.0234157, 0.0222461, -0.0288896, 0.0291202, -0.0525359, 0.0511357
7: -0.0461166, 0.0125291, -0.0618365, 0.0286836, -0.0748002, 0.0743656
8: 0.9244128, 0.9948584, 0.8959473, 0.9955601, -0.0711473, 0.0989111
9: -0.0086460, 0.0601269, -0.0148798, 0.0941900, -0.1028360, 0.0750067

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3791538, upper bound: 0.3844507
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3789153, upper bound: 0.3789139
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0095893, 0.0232121, -0.0172581, 0.0298437, -0.0394330, 0.0404702
1: -0.0114111, 0.0043119, -0.0151399, 0.0083981, -0.0198093, 0.0194517
2: -0.0040541, 0.0239550, -0.0109017, 0.0294701, -0.0335242, 0.0348567
3: -0.0047653, 0.0129477, -0.0072293, 0.0169131, -0.0216784, 0.0201769
4: -0.0114221, 0.0055739, -0.0164731, 0.0124494, -0.0238715, 0.0220470
5: -0.0061224, 0.0193764, -0.0136928, 0.0260826, -0.0322050, 0.0330693
6: -0.0147106, 0.0174241, -0.0188366, 0.0187052, -0.0334158, 0.0362607
7: -0.0365771, 0.0032983, -0.0414973, 0.0064738, -0.0430509, 0.0447955
8: 0.9454601, 0.9908453, 0.9403118, 0.9927152, -0.0472551, 0.0505335
9: -0.0061706, 0.0287391, -0.0072902, 0.0393496, -0.0455202, 0.0360293

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3798196, upper bound: 0.3765986
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3798196, upper bound: 0.3765986
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0095893, 0.0232121, -0.0326248, 0.0337374, -0.0433267, 0.0558370
1: -0.0114111, 0.0043119, -0.0228376, 0.0182326, -0.0296437, 0.0271495
2: -0.0040541, 0.0239550, -0.0199897, 0.0442654, -0.0483194, 0.0439447
3: -0.0047653, 0.0129477, -0.0138262, 0.0234056, -0.0281709, 0.0267738
4: -0.0114221, 0.0055739, -0.0362488, 0.0248295, -0.0362516, 0.0418227
5: -0.0061224, 0.0193764, -0.0216912, 0.0403950, -0.0465174, 0.0410677
6: -0.0147106, 0.0174241, -0.0245827, 0.0256339, -0.0403444, 0.0420068
7: -0.0365771, 0.0032983, -0.0506919, 0.0207384, -0.0573155, 0.0539902
8: 0.9454601, 0.9908453, 0.9117983, 0.9946759, -0.0492157, 0.0790470
9: -0.0061706, 0.0287391, -0.0089768, 0.0776040, -0.0837746, 0.0377159

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3800111, upper bound: 0.3824546
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3800111, upper bound: 0.3824546
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0265467, 0.0303759, -0.0172581, 0.0298437, -0.0563905, 0.0476340
1: -0.0182748, 0.0140857, -0.0151399, 0.0083981, -0.0266729, 0.0292256
2: -0.0148019, 0.0378118, -0.0109017, 0.0294701, -0.0442721, 0.0487135
3: -0.0109130, 0.0189780, -0.0072293, 0.0169131, -0.0278261, 0.0262073
4: -0.0267555, 0.0189954, -0.0164731, 0.0124494, -0.0392049, 0.0354684
5: -0.0179390, 0.0348805, -0.0136928, 0.0260826, -0.0440217, 0.0485733
6: -0.0211310, 0.0221200, -0.0188366, 0.0187052, -0.0398363, 0.0409566
7: -0.0453460, 0.0120630, -0.0414973, 0.0064738, -0.0518198, 0.0535603
8: 0.9249541, 0.9936405, 0.9403118, 0.9927152, -0.0677612, 0.0533287
9: -0.0079011, 0.0596590, -0.0072902, 0.0393496, -0.0472508, 0.0669492

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3773518, upper bound: 0.3731076
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3772248, upper bound: 0.3684283
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0267401, 0.0329629, -0.0133934, 0.0295061, -0.0562462, 0.0463563
1: -0.0189058, 0.0146470, -0.0140920, 0.0065785, -0.0254844, 0.0287390
2: -0.0161642, 0.0386078, -0.0093739, 0.0270934, -0.0432576, 0.0479817
3: -0.0111078, 0.0198881, -0.0067165, 0.0162268, -0.0273346, 0.0266046
4: -0.0275032, 0.0190706, -0.0140709, 0.0103118, -0.0378150, 0.0331415
5: -0.0185541, 0.0350990, -0.0126810, 0.0226709, -0.0412250, 0.0477800
6: -0.0234157, 0.0222461, -0.0180810, 0.0181109, -0.0415266, 0.0403271
7: -0.0461166, 0.0125291, -0.0403330, 0.0055188, -0.0516354, 0.0528621
8: 0.9244128, 0.9948584, 0.9429308, 0.9924831, -0.0680704, 0.0519276
9: -0.0086460, 0.0601269, -0.0069430, 0.0351751, -0.0438212, 0.0670699

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3764406, upper bound: 0.3730360
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3762802, upper bound: 0.3683722
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0265467, 0.0303759, -0.0326248, 0.0337374, -0.0602841, 0.0630007
1: -0.0182748, 0.0140857, -0.0228376, 0.0182326, -0.0365074, 0.0369233
2: -0.0148019, 0.0378118, -0.0199897, 0.0442654, -0.0590673, 0.0578015
3: -0.0109130, 0.0189780, -0.0138262, 0.0234056, -0.0343186, 0.0328042
4: -0.0267555, 0.0189954, -0.0362488, 0.0248295, -0.0515849, 0.0552441
5: -0.0179390, 0.0348805, -0.0216912, 0.0403950, -0.0583340, 0.0565718
6: -0.0211310, 0.0221200, -0.0245827, 0.0256339, -0.0467649, 0.0467027
7: -0.0453460, 0.0120630, -0.0506919, 0.0207384, -0.0660844, 0.0627549
8: 0.9249541, 0.9936405, 0.9117983, 0.9946759, -0.0697218, 0.0818422
9: -0.0079011, 0.0596590, -0.0089768, 0.0776040, -0.0855052, 0.0686358

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3792230, upper bound: 0.3800657
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3791095, upper bound: 0.3756382
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0267401, 0.0329629, -0.0283806, 0.0322592, -0.0589993, 0.0613434
1: -0.0189058, 0.0146470, -0.0197324, 0.0153447, -0.0342506, 0.0343795
2: -0.0161642, 0.0386078, -0.0166086, 0.0399560, -0.0561202, 0.0552165
3: -0.0111078, 0.0198881, -0.0115989, 0.0203204, -0.0314283, 0.0314870
4: -0.0275032, 0.0190706, -0.0293785, 0.0207082, -0.0482114, 0.0484491
5: -0.0185541, 0.0350990, -0.0192536, 0.0365042, -0.0550584, 0.0543526
6: -0.0234157, 0.0222461, -0.0228168, 0.0231564, -0.0465721, 0.0450629
7: -0.0461166, 0.0125291, -0.0470750, 0.0146142, -0.0607308, 0.0596042
8: 0.9244128, 0.9948584, 0.9211489, 0.9942269, -0.0698141, 0.0737095
9: -0.0086460, 0.0601269, -0.0085576, 0.0650722, -0.0737183, 0.0686845

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3780401, upper bound: 0.3799910
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3779219, upper bound: 0.3755863
time: 2.10 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0032995, 0.0241294, -0.0446593, 0.0508000, -0.0540995, 0.0687887
1: -0.0097478, 0.0042597, -0.0350039, 0.0393280, -0.0490758, 0.0392636
2: -0.0019894, 0.0226821, -0.0350894, 0.0619035, -0.0638929, 0.0577715
3: -0.0045132, 0.0119982, -0.0230137, 0.0475617, -0.0520749, 0.0350119
4: -0.0092663, 0.0023866, -0.0541894, 0.0419273, -0.0511937, 0.0565760
5: -0.0050022, 0.0126059, -0.0359531, 0.0554233, -0.0604254, 0.0485590
6: -0.0149018, 0.0159661, -0.0356649, 0.0381099, -0.0530117, 0.0516310
7: -0.0341738, 0.0026725, -0.0723669, 0.0366911, -0.0708649, 0.0750395
8: 0.9499696, 0.9910315, 0.8788870, 0.9968827, -0.0469131, 0.1121445
9: -0.0063073, 0.0204114, -0.0241333, 0.1080855, -0.1143928, 0.0445447

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3731077, upper bound: 0.3773518
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3730360, upper bound: 0.3764406
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0032759, 0.0241034, -0.0373747, 0.0360188, -0.0392946, 0.0614781
1: -0.0096792, 0.0044603, -0.0260123, 0.0273732, -0.0370523, 0.0304726
2: -0.0019557, 0.0226626, -0.0248163, 0.0514424, -0.0533981, 0.0474789
3: -0.0045859, 0.0120954, -0.0156397, 0.0310336, -0.0356195, 0.0277351
4: -0.0092889, 0.0024075, -0.0436816, 0.0315868, -0.0408757, 0.0460891
5: -0.0043410, 0.0125739, -0.0240181, 0.0446941, -0.0490351, 0.0365920
6: -0.0159808, 0.0158973, -0.0269493, 0.0282676, -0.0442484, 0.0428466
7: -0.0339206, 0.0039808, -0.0591240, 0.0271309, -0.0610515, 0.0631048
8: 0.9501686, 0.9916172, 0.8986889, 0.9942101, -0.0440416, 0.0929283
9: -0.0069882, 0.0201466, -0.0131521, 0.0916385, -0.0986267, 0.0332987

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3684283, upper bound: 0.3772247
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3683723, upper bound: 0.3762802
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0151050, 0.0257729, -0.0267657, 0.0295957, -0.0447006, 0.0525386
1: -0.0133149, 0.0067611, -0.0183217, 0.0138815, -0.0271965, 0.0250828
2: -0.0073152, 0.0277913, -0.0145404, 0.0377148, -0.0450300, 0.0423317
3: -0.0061131, 0.0147476, -0.0097822, 0.0188234, -0.0249365, 0.0245298
4: -0.0150394, 0.0099349, -0.0268445, 0.0177344, -0.0327738, 0.0367794
5: -0.0098811, 0.0242113, -0.0157390, 0.0350826, -0.0449637, 0.0399503
6: -0.0175739, 0.0183272, -0.0202868, 0.0221805, -0.0397544, 0.0386139
7: -0.0386058, 0.0059683, -0.0454834, 0.0118032, -0.0504089, 0.0514517
8: 0.9417484, 0.9923577, 0.9245001, 0.9927343, -0.0509859, 0.0678576
9: -0.0072547, 0.0359048, -0.0076698, 0.0597213, -0.0669760, 0.0435746

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3824546, upper bound: 0.3800111
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3824546, upper bound: 0.3800111
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0151050, 0.0257729, -0.0446593, 0.0508000, -0.0659050, 0.0704321
1: -0.0133149, 0.0067611, -0.0350039, 0.0393280, -0.0526429, 0.0417650
2: -0.0073152, 0.0277913, -0.0350894, 0.0619035, -0.0692187, 0.0628808
3: -0.0061131, 0.0147476, -0.0230137, 0.0475617, -0.0536748, 0.0377613
4: -0.0150394, 0.0099349, -0.0541894, 0.0419273, -0.0569668, 0.0641243
5: -0.0098811, 0.0242113, -0.0359531, 0.0554233, -0.0653044, 0.0601644
6: -0.0175739, 0.0183272, -0.0356649, 0.0381099, -0.0556838, 0.0539921
7: -0.0386058, 0.0059683, -0.0723669, 0.0366911, -0.0752969, 0.0783353
8: 0.9417484, 0.9923577, 0.8788870, 0.9968827, -0.0551343, 0.1134707
9: -0.0072547, 0.0359048, -0.0241333, 0.1080855, -0.1153402, 0.0600381

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3800802, upper bound: 0.3791763
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3800087, upper bound: 0.3780150
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135347, 0.0248789, -0.0373747, 0.0360188, -0.0495535, 0.0622536
1: -0.0128549, 0.0054948, -0.0260123, 0.0273732, -0.0402281, 0.0315071
2: -0.0065000, 0.0263553, -0.0248163, 0.0514424, -0.0579425, 0.0511716
3: -0.0049636, 0.0142288, -0.0156397, 0.0310336, -0.0359972, 0.0298685
4: -0.0141117, 0.0065100, -0.0436816, 0.0315868, -0.0456985, 0.0501916
5: -0.0051146, 0.0229052, -0.0240181, 0.0446941, -0.0498087, 0.0469232
6: -0.0175344, 0.0179578, -0.0269493, 0.0282676, -0.0458019, 0.0449072
7: -0.0380482, 0.0062766, -0.0591240, 0.0271309, -0.0651791, 0.0654005
8: 0.9427510, 0.9921382, 0.8986889, 0.9942101, -0.0514591, 0.0934493
9: -0.0077489, 0.0330292, -0.0131521, 0.0916385, -0.0993874, 0.0461814

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3756434, upper bound: 0.3790648
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3755903, upper bound: 0.3779077
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0151050, 0.0257729, -0.0172581, 0.0298437, -0.0449487, 0.0430310
1: -0.0133149, 0.0067611, -0.0151399, 0.0083981, -0.0217131, 0.0219010
2: -0.0073152, 0.0277913, -0.0109017, 0.0294701, -0.0367854, 0.0386930
3: -0.0061131, 0.0147476, -0.0072293, 0.0169131, -0.0230262, 0.0219769
4: -0.0150394, 0.0099349, -0.0164731, 0.0124494, -0.0274889, 0.0264080
5: -0.0098811, 0.0242113, -0.0136928, 0.0260826, -0.0359637, 0.0379041
6: -0.0175739, 0.0183272, -0.0188366, 0.0187052, -0.0362791, 0.0371637
7: -0.0386058, 0.0059683, -0.0414973, 0.0064738, -0.0450796, 0.0474656
8: 0.9417484, 0.9923577, 0.9403118, 0.9927152, -0.0509669, 0.0520459
9: -0.0072547, 0.0359048, -0.0072902, 0.0393496, -0.0466044, 0.0431950

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3766300
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3766300
time: 1.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0151050, 0.0257729, -0.0326248, 0.0337374, -0.0488424, 0.0583977
1: -0.0133149, 0.0067611, -0.0228376, 0.0182326, -0.0315475, 0.0295987
2: -0.0073152, 0.0277913, -0.0199897, 0.0442654, -0.0515806, 0.0477810
3: -0.0061131, 0.0147476, -0.0138262, 0.0234056, -0.0295186, 0.0285738
4: -0.0150394, 0.0099349, -0.0362488, 0.0248295, -0.0398689, 0.0461837
5: -0.0098811, 0.0242113, -0.0216912, 0.0403950, -0.0502761, 0.0459026
6: -0.0175739, 0.0183272, -0.0245827, 0.0256339, -0.0432078, 0.0429099
7: -0.0386058, 0.0059683, -0.0506919, 0.0207384, -0.0593441, 0.0566602
8: 0.9417484, 0.9923577, 0.9117983, 0.9946759, -0.0529275, 0.0805594
9: -0.0072547, 0.0359048, -0.0089768, 0.0776040, -0.0848588, 0.0448816

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836863
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836863
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0135347, 0.0248789, -0.0277992, 0.0309235, -0.0444582, 0.0526781
1: -0.0128549, 0.0054948, -0.0191015, 0.0147365, -0.0275915, 0.0245963
2: -0.0065000, 0.0263553, -0.0155597, 0.0390750, -0.0455750, 0.0419151
3: -0.0049636, 0.0142288, -0.0106296, 0.0195750, -0.0245386, 0.0248584
4: -0.0141117, 0.0065100, -0.0283745, 0.0193224, -0.0334341, 0.0348845
5: -0.0051146, 0.0229052, -0.0172086, 0.0359924, -0.0411070, 0.0401138
6: -0.0175344, 0.0179578, -0.0219858, 0.0227916, -0.0403260, 0.0399437
7: -0.0380482, 0.0062766, -0.0461802, 0.0135504, -0.0515986, 0.0524568
8: 0.9427510, 0.9921382, 0.9223605, 0.9938361, -0.0510851, 0.0697777
9: -0.0077489, 0.0330292, -0.0084755, 0.0630193, -0.0707683, 0.0415047

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836867
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836867
time: 1.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.64 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3809564
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3809564, upper bound: 0.3809564
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3778105, upper bound: 0.3791256
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3777142, upper bound: 0.3778668
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3725845, upper bound: 0.3789321
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3724938, upper bound: 0.3775928
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3791256, upper bound: 0.3778105
NS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3789322, upper bound: 0.3725845
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3778668, upper bound: 0.3777141
NS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3775929, upper bound: 0.3724937
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3928508, upper bound: 0.3928391
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3928508, upper bound: 0.3928392
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3791538, upper bound: 0.3844507
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3789153, upper bound: 0.3789139
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3798196, upper bound: 0.3765986
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3798196, upper bound: 0.3765986
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3800111, upper bound: 0.3824546
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3800111, upper bound: 0.3824546
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3773518, upper bound: 0.3731076
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3772248, upper bound: 0.3684283
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3764406, upper bound: 0.3730360
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3762802, upper bound: 0.3683722
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3792230, upper bound: 0.3800657
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3791095, upper bound: 0.3756382
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3780401, upper bound: 0.3799910
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3779219, upper bound: 0.3755863
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3731077, upper bound: 0.3773518
NS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3730360, upper bound: 0.3764406
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3684283, upper bound: 0.3772247
NS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3683723, upper bound: 0.3762802
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3824546, upper bound: 0.3800111
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3824546, upper bound: 0.3800111
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3800802, upper bound: 0.3791763
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3800087, upper bound: 0.3780150
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3756434, upper bound: 0.3790648
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3755903, upper bound: 0.3779077
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3766300
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3822887, upper bound: 0.3766300
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836863
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836863
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836867
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.64
Output dim: 8, lower bound: -0.3837191, upper bound: 0.3836867

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0265467, 0.0303759, -0.0265467, 0.0303759, -0.0569226, 0.0569226
1: -0.0182748, 0.0140857, -0.0182748, 0.0140857, -0.0323605, 0.0323605
2: -0.0148019, 0.0378118, -0.0148019, 0.0378118, -0.0526137, 0.0526137
3: -0.0109130, 0.0189780, -0.0109130, 0.0189780, -0.0298910, 0.0298910
4: -0.0267555, 0.0189954, -0.0267555, 0.0189954, -0.0457509, 0.0457509
5: -0.0179390, 0.0348805, -0.0179390, 0.0348805, -0.0528196, 0.0528196
6: -0.0211310, 0.0221200, -0.0211310, 0.0221200, -0.0432511, 0.0432511
7: -0.0453460, 0.0120630, -0.0453460, 0.0120630, -0.0574090, 0.0574090
8: 0.9249541, 0.9936405, 0.9249541, 0.9936405, -0.0686864, 0.0686864
9: -0.0079011, 0.0596590, -0.0079011, 0.0596590, -0.0675601, 0.0675601

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3863653, upper bound: 0.3792483
time: 1.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3802736, upper bound: 0.3789910
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0265467, 0.0303759, -0.0267401, 0.0329629, -0.0595096, 0.0571160
1: -0.0182748, 0.0140857, -0.0189058, 0.0146470, -0.0329218, 0.0329916
2: -0.0148019, 0.0378118, -0.0161642, 0.0386078, -0.0534097, 0.0539760
3: -0.0109130, 0.0189780, -0.0111078, 0.0198881, -0.0308011, 0.0300858
4: -0.0267555, 0.0189954, -0.0275032, 0.0190706, -0.0458261, 0.0464986
5: -0.0179390, 0.0348805, -0.0185541, 0.0350990, -0.0530381, 0.0534346
6: -0.0211310, 0.0221200, -0.0234157, 0.0222461, -0.0433771, 0.0455357
7: -0.0453460, 0.0120630, -0.0461166, 0.0125291, -0.0578751, 0.0581796
8: 0.9249541, 0.9936405, 0.9244128, 0.9948584, -0.0699044, 0.0692277
9: -0.0079011, 0.0596590, -0.0086460, 0.0601269, -0.0680281, 0.0683050

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3863653, upper bound: 0.3792506
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3802736, upper bound: 0.3789933
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0267401, 0.0329629, -0.0198488, 0.0246680, -0.0514081, 0.0528117
1: -0.0189058, 0.0146470, -0.0144075, 0.0091934, -0.0280992, 0.0290546
2: -0.0161642, 0.0386078, -0.0085656, 0.0305882, -0.0467523, 0.0471734
3: -0.0111078, 0.0198881, -0.0063451, 0.0155012, -0.0266090, 0.0262332
4: -0.0275032, 0.0190706, -0.0183087, 0.0124325, -0.0399358, 0.0373793
5: -0.0185541, 0.0350990, -0.0104524, 0.0284111, -0.0469653, 0.0455515
6: -0.0234157, 0.0222461, -0.0173740, 0.0190641, -0.0424798, 0.0396201
7: -0.0461166, 0.0125291, -0.0396932, 0.0063541, -0.0524707, 0.0522223
8: 0.9244128, 0.9948584, 0.9384336, 0.9919515, -0.0675387, 0.0564249
9: -0.0086460, 0.0601269, -0.0071906, 0.0410857, -0.0497318, 0.0673175

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3789153, upper bound: 0.3789138
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3789153, upper bound: 0.3789139
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0151050, 0.0257729, -0.0151050, 0.0257729, -0.0408778, 0.0408778
1: -0.0133149, 0.0067611, -0.0133149, 0.0067611, -0.0200761, 0.0200761
2: -0.0073152, 0.0277913, -0.0073152, 0.0277913, -0.0351066, 0.0351066
3: -0.0061131, 0.0147476, -0.0061131, 0.0147476, -0.0208607, 0.0208607
4: -0.0150394, 0.0099349, -0.0150394, 0.0099349, -0.0249744, 0.0249744
5: -0.0098811, 0.0242113, -0.0098811, 0.0242113, -0.0340924, 0.0340924
6: -0.0175739, 0.0183272, -0.0175739, 0.0183272, -0.0359011, 0.0359011
7: -0.0386058, 0.0059683, -0.0386058, 0.0059683, -0.0445741, 0.0445741
8: 0.9417484, 0.9923577, 0.9417484, 0.9923577, -0.0506094, 0.0506094
9: -0.0072547, 0.0359048, -0.0072547, 0.0359048, -0.0431595, 0.0431595

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3811100, upper bound: 0.3751234
time: 1.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3795676, upper bound: 0.3749997
time: 1.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0151050, 0.0257729, -0.0135347, 0.0248789, -0.0399838, 0.0393076
1: -0.0133149, 0.0067611, -0.0128549, 0.0054948, -0.0188098, 0.0196161
2: -0.0073152, 0.0277913, -0.0065000, 0.0263553, -0.0336706, 0.0342914
3: -0.0061131, 0.0147476, -0.0049636, 0.0142288, -0.0203419, 0.0197112
4: -0.0150394, 0.0099349, -0.0141117, 0.0065100, -0.0215495, 0.0240467
5: -0.0098811, 0.0242113, -0.0051146, 0.0229052, -0.0327863, 0.0293259
6: -0.0175739, 0.0183272, -0.0175344, 0.0179578, -0.0355317, 0.0358615
7: -0.0386058, 0.0059683, -0.0380482, 0.0062766, -0.0448823, 0.0440165
8: 0.9417484, 0.9923577, 0.9427510, 0.9921382, -0.0503898, 0.0496067
9: -0.0072547, 0.0359048, -0.0077489, 0.0330292, -0.0402840, 0.0436537

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3811100, upper bound: 0.3751235
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3795676, upper bound: 0.3749997
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0135347, 0.0248789, -0.0151050, 0.0257729, -0.0393076, 0.0399838
1: -0.0128549, 0.0054948, -0.0133149, 0.0067611, -0.0196161, 0.0188098
2: -0.0065000, 0.0263553, -0.0073152, 0.0277913, -0.0342914, 0.0336706
3: -0.0049636, 0.0142288, -0.0061131, 0.0147476, -0.0197112, 0.0203419
4: -0.0141117, 0.0065100, -0.0150394, 0.0099349, -0.0240467, 0.0215495
5: -0.0051146, 0.0229052, -0.0098811, 0.0242113, -0.0293259, 0.0327863
6: -0.0175344, 0.0179578, -0.0175739, 0.0183272, -0.0358615, 0.0355317
7: -0.0380482, 0.0062766, -0.0386058, 0.0059683, -0.0440165, 0.0448823
8: 0.9427510, 0.9921382, 0.9417484, 0.9923577, -0.0496067, 0.0503898
9: -0.0077489, 0.0330292, -0.0072547, 0.0359048, -0.0436537, 0.0402840

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3766373, upper bound: 0.3749277
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3748364, upper bound: 0.3748041
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0135347, 0.0248789, -0.0135347, 0.0248789, -0.0384136, 0.0384136
1: -0.0128549, 0.0054948, -0.0128549, 0.0054948, -0.0183498, 0.0183498
2: -0.0065000, 0.0263553, -0.0065000, 0.0263553, -0.0328554, 0.0328554
3: -0.0049636, 0.0142288, -0.0049636, 0.0142288, -0.0191924, 0.0191924
4: -0.0141117, 0.0065100, -0.0141117, 0.0065100, -0.0206217, 0.0206217
5: -0.0051146, 0.0229052, -0.0051146, 0.0229052, -0.0280198, 0.0280198
6: -0.0175344, 0.0179578, -0.0175344, 0.0179578, -0.0354922, 0.0354922
7: -0.0380482, 0.0062766, -0.0380482, 0.0062766, -0.0443248, 0.0443248
8: 0.9427510, 0.9921382, 0.9427510, 0.9921382, -0.0493872, 0.0493872
9: -0.0077489, 0.0330292, -0.0077489, 0.0330292, -0.0407782, 0.0407782

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3766373, upper bound: 0.3749277
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3748364, upper bound: 0.3748041
time: 1.30 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.58 seconds
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3863653, upper bound: 0.3792483
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3802736, upper bound: 0.3789910
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3863653, upper bound: 0.3792506
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3802736, upper bound: 0.3789933
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3789153, upper bound: 0.3789138
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3789153, upper bound: 0.3789139
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3811100, upper bound: 0.3751234
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3795676, upper bound: 0.3749997
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3811100, upper bound: 0.3751235
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3795676, upper bound: 0.3749997
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3766373, upper bound: 0.3749277
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3748364, upper bound: 0.3748041
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3766373, upper bound: 0.3749277
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.58
Output dim: 8, lower bound: -0.3748364, upper bound: 0.3748041

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0093526, 0.0239631, -0.0265467, 0.0303759, -0.0397285, 0.0505098
1: -0.0114082, 0.0046208, -0.0182748, 0.0140857, -0.0254939, 0.0228956
2: -0.0042287, 0.0241790, -0.0148019, 0.0378118, -0.0420405, 0.0389809
3: -0.0054254, 0.0131551, -0.0109130, 0.0189780, -0.0244034, 0.0240681
4: -0.0113314, 0.0069332, -0.0267555, 0.0189954, -0.0303268, 0.0336886
5: -0.0083753, 0.0191284, -0.0179390, 0.0348805, -0.0432559, 0.0370674
6: -0.0156208, 0.0174502, -0.0211310, 0.0221200, -0.0377408, 0.0385812
7: -0.0364518, 0.0038967, -0.0453460, 0.0120630, -0.0485148, 0.0492428
8: 0.9456504, 0.9915055, 0.9249541, 0.9936405, -0.0479901, 0.0665514
9: -0.0063288, 0.0292858, -0.0079011, 0.0596590, -0.0659878, 0.0371870

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3807576, upper bound: 0.3807569
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3807576, upper bound: 0.3807569
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0093526, 0.0239631, -0.0267401, 0.0329629, -0.0423155, 0.0507031
1: -0.0114082, 0.0046208, -0.0189058, 0.0146470, -0.0260552, 0.0235266
2: -0.0042287, 0.0241790, -0.0161642, 0.0386078, -0.0428366, 0.0403432
3: -0.0054254, 0.0131551, -0.0111078, 0.0198881, -0.0253135, 0.0242629
4: -0.0113314, 0.0069332, -0.0275032, 0.0190706, -0.0304020, 0.0344364
5: -0.0083753, 0.0191284, -0.0185541, 0.0350990, -0.0434743, 0.0376825
6: -0.0156208, 0.0174502, -0.0234157, 0.0222461, -0.0378669, 0.0408659
7: -0.0364518, 0.0038967, -0.0461166, 0.0125291, -0.0489809, 0.0500134
8: 0.9456504, 0.9915055, 0.9244128, 0.9948584, -0.0492080, 0.0670927
9: -0.0063288, 0.0292858, -0.0086460, 0.0601269, -0.0664557, 0.0379319

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 182

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3802733, upper bound: 0.3789933
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3802733, upper bound: 0.3789932
time: 1.26 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.77 seconds
NS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.77
Output dim: 8, lower bound: -0.3807576, upper bound: 0.3807569
NS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.77
Output dim: 8, lower bound: -0.3807576, upper bound: 0.3807569
NS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.77
Output dim: 8, lower bound: -0.3802733, upper bound: 0.3789933
NS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.77
Output dim: 8, lower bound: -0.3802733, upper bound: 0.3789932

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.40 + 305.67 = 310.07 seconds
