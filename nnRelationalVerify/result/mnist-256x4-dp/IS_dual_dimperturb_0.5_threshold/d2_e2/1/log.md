## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.35483528


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1944040, 0.2134438, -0.1944040, 0.2134438, -0.4063973, 0.4063973)
1: (-0.1162487, 0.0975982, -0.1162487, 0.0975982, -0.2138469, 0.2138469)
2: (-0.1161766, 0.1337720, -0.1161766, 0.1337720, -0.2499486, 0.2499486)
3: (-0.0922217, 0.1217764, -0.0922217, 0.1217764, -0.2139981, 0.2139981)
4: (-0.1167147, 0.0783617, -0.1167147, 0.0783617, -0.1950765, 0.1950765)
5: (-0.1301616, 0.1231892, -0.1301616, 0.1231892, -0.2533508, 0.2533508)
6: (-0.1587840, 0.1081306, -0.1587840, 0.1081306, -0.2634545, 0.2634544)
7: (-0.1143068, 0.0988387, -0.1143068, 0.0988387, -0.2131455, 0.2131455)
8: (0.4868668, 1.1453948, 0.4868668, 1.1453948, -0.6390972, 0.6390972)
9: (-0.0860214, 0.1395597, -0.0860214, 0.1395597, -0.2255811, 0.2255811)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.16 + 1.83 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3829977, upper bound: 0.3829977

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3759756, upper bound: 0.3655852
time: 0.83 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3779395, upper bound: 0.3779395
time: 0.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.91 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 8, lower bound: -0.3759756, upper bound: 0.3655852
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 8, lower bound: -0.3779395, upper bound: 0.3779395

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.1921257, 0.2013584, -0.1807004, 0.1522276, -0.3415424, 0.3803294
1: -0.1144488, 0.0959450, -0.1059122, 0.0880544, -0.2025032, 0.2018572
2: -0.1145849, 0.1312251, -0.1065026, 0.1207162, -0.2353012, 0.2377277
3: -0.0914912, 0.1193622, -0.0871360, 0.1085016, -0.1999929, 0.2064982
4: -0.1141362, 0.0766239, -0.1025703, 0.0686216, -0.1827578, 0.1791943
5: -0.1282380, 0.1217914, -0.1188868, 0.1148079, -0.2430459, 0.2406782
6: -0.1571007, 0.1035922, -0.1485240, 0.0916152, -0.2452888, 0.2486764
7: -0.1130500, 0.0967695, -0.1065407, 0.0869692, -0.2000192, 0.2033102
8: 0.5010729, 1.1442504, 0.5540407, 1.1367686, -0.6115847, 0.5624440
9: -0.0836619, 0.1377212, -0.0724507, 0.1288523, -0.2125142, 0.2101719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3655852
time: 0.86 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3655852
time: 0.81 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.1944040, 0.2134438, -0.1915396, 0.1989554, -0.3919139, 0.4035332
1: -0.1162487, 0.0975982, -0.1139856, 0.0956200, -0.2118687, 0.2115837
2: -0.1161766, 0.1337720, -0.1142173, 0.1307839, -0.2469605, 0.2479893
3: -0.0922217, 0.1217764, -0.0913179, 0.1187139, -0.2109357, 0.2130943
4: -0.1167147, 0.0783617, -0.1136799, 0.0762545, -0.1929693, 0.1920416
5: -0.1301616, 0.1231892, -0.1277903, 0.1215110, -0.2516725, 0.2509796
6: -0.1587840, 0.1081306, -0.1567937, 0.1007414, -0.2560784, 0.2614760
7: -0.1143068, 0.0988387, -0.1128079, 0.0962646, -0.2105714, 0.2116466
8: 0.4868668, 1.1453948, 0.5073789, 1.1438251, -0.6369944, 0.6166840
9: -0.0860214, 0.1395597, -0.0831536, 0.1373781, -0.2233994, 0.2227133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3759756
time: 0.74 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3779395
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.74 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3655852
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3655852
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3759756
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 8, lower bound: -0.3655852, upper bound: 0.3779395

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -0.1807004, 0.1522276, -0.1807004, 0.1522276, -0.3300432, 0.3300431
1: -0.1059122, 0.0880544, -0.1059122, 0.0880544, -0.1939666, 0.1939666
2: -0.1065026, 0.1207162, -0.1065026, 0.1207162, -0.2272188, 0.2272188
3: -0.0871360, 0.1085016, -0.0871360, 0.1085016, -0.1956376, 0.1956376
4: -0.1025703, 0.0686216, -0.1025703, 0.0686216, -0.1711920, 0.1711920
5: -0.1188868, 0.1148079, -0.1188868, 0.1148079, -0.2336947, 0.2336947
6: -0.1485240, 0.0916152, -0.1485240, 0.0916152, -0.2367138, 0.2367136
7: -0.1065407, 0.0869692, -0.1065407, 0.0869692, -0.1935099, 0.1935099
8: 0.5540407, 1.1367686, 0.5540407, 1.1367686, -0.5523357, 0.5523355
9: -0.0724507, 0.1288523, -0.0724507, 0.1288523, -0.2013030, 0.2013030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3478642, upper bound: 0.3504170
time: 0.71 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3456328, upper bound: 0.3455174
time: 0.71 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -0.1915396, 0.1989554, -0.1807004, 0.1522276, -0.3409547, 0.3781369
1: -0.1139856, 0.0956200, -0.1059122, 0.0880544, -0.2020400, 0.2015322
2: -0.1142173, 0.1307839, -0.1065026, 0.1207162, -0.2349335, 0.2372864
3: -0.0913179, 0.1187139, -0.0871360, 0.1085016, -0.1998195, 0.2058499
4: -0.1136799, 0.0762545, -0.1025703, 0.0686216, -0.1823015, 0.1788249
5: -0.1277903, 0.1215110, -0.1188868, 0.1148079, -0.2425982, 0.2403977
6: -0.1567937, 0.1007414, -0.1485240, 0.0916152, -0.2449815, 0.2458318
7: -0.1128079, 0.0962646, -0.1065407, 0.0869692, -0.1997771, 0.2028053
8: 0.5073789, 1.1438251, 0.5540407, 1.1367686, -0.6049290, 0.5619795
9: -0.0831536, 0.1373781, -0.0724507, 0.1288523, -0.2120059, 0.2098288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3510108, upper bound: 0.3477963
time: 0.78 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3456328, upper bound: 0.3455174
time: 0.88 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.1807004, 0.1522276, -0.1915396, 0.1989554, -0.3781369, 0.3409545
1: -0.1059122, 0.0880544, -0.1139856, 0.0956200, -0.2015322, 0.2020400
2: -0.1065026, 0.1207162, -0.1142173, 0.1307839, -0.2372864, 0.2349335
3: -0.0871360, 0.1085016, -0.0913179, 0.1187139, -0.2058499, 0.1998195
4: -0.1025703, 0.0686216, -0.1136799, 0.0762545, -0.1788249, 0.1823015
5: -0.1188868, 0.1148079, -0.1277903, 0.1215110, -0.2403977, 0.2425982
6: -0.1485240, 0.0916152, -0.1567937, 0.1007414, -0.2458319, 0.2449815
7: -0.1065407, 0.0869692, -0.1128079, 0.0962646, -0.2028053, 0.1997771
8: 0.5540407, 1.1367686, 0.5073789, 1.1438251, -0.5619793, 0.6049294
9: -0.0724507, 0.1288523, -0.0831536, 0.1373781, -0.2098288, 0.2120059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3477963, upper bound: 0.3651570
time: 0.84 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
time: 0.87 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.1915396, 0.1989554, -0.1915396, 0.1989554, -0.3890501, 0.3890500
1: -0.1139856, 0.0956200, -0.1139856, 0.0956200, -0.2096055, 0.2096055
2: -0.1142173, 0.1307839, -0.1142173, 0.1307839, -0.2450012, 0.2450012
3: -0.0913179, 0.1187139, -0.0913179, 0.1187139, -0.2100318, 0.2100318
4: -0.1136799, 0.0762545, -0.1136799, 0.0762545, -0.1899344, 0.1899344
5: -0.1277903, 0.1215110, -0.1277903, 0.1215110, -0.2493013, 0.2493013
6: -0.1567937, 0.1007414, -0.1567937, 0.1007414, -0.2541001, 0.2541000
7: -0.1128079, 0.0962646, -0.1128079, 0.0962646, -0.2090726, 0.2090726
8: 0.5073789, 1.1438251, 0.5073789, 1.1438251, -0.6145821, 0.6145821
9: -0.0831536, 0.1373781, -0.0831536, 0.1373781, -0.2205317, 0.2205317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3504170, upper bound: 0.3576674
time: 1.04 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3576401
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.98 seconds
IS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3478642, upper bound: 0.3504170
IS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3456328, upper bound: 0.3455174
IS_B1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3510108, upper bound: 0.3477963
IS_B1_A2_B2, status: Status.VERIFIED, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3456328, upper bound: 0.3455174
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3477963, upper bound: 0.3651570
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3504170, upper bound: 0.3576674
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3576401

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1748026, 0.1503645, -0.1915396, 0.1989554, -0.3722384, 0.3389206
1: -0.1022011, 0.0848079, -0.1139856, 0.0956200, -0.1978211, 0.1987935
2: -0.1019022, 0.1168443, -0.1142173, 0.1307839, -0.2326861, 0.2310616
3: -0.0835326, 0.1053720, -0.0913179, 0.1187139, -0.2022465, 0.1966899
4: -0.0988611, 0.0655613, -0.1136799, 0.0762545, -0.1751156, 0.1792412
5: -0.1140495, 0.1112707, -0.1277903, 0.1215110, -0.2355605, 0.2390611
6: -0.1436899, 0.0908413, -0.1567937, 0.1007414, -0.2409992, 0.2442080
7: -0.1027083, 0.0827986, -0.1128079, 0.0962646, -0.1989729, 0.1956065
8: 0.5571116, 1.1294914, 0.5073789, 1.1438251, -0.5587220, 0.5976696
9: -0.0675066, 0.1248847, -0.0831536, 0.1373781, -0.2048847, 0.2080383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
time: 0.86 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
time: 1.38 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.1600679, 0.1800335, -0.1864232, 0.1975511, -0.3564086, 0.3637888
1: -0.0940157, 0.0782800, -0.1108270, 0.0928625, -0.1868782, 0.1891070
2: -0.0898781, 0.1102740, -0.1102498, 0.1276012, -0.2174793, 0.2205238
3: -0.0721118, 0.1113509, -0.0880790, 0.1161532, -0.1882650, 0.1994300
4: -0.0936740, 0.0601832, -0.1106712, 0.0737457, -0.1674197, 0.1708544
5: -0.1015542, 0.1030096, -0.1235694, 0.1185036, -0.2200578, 0.2265790
6: -0.1312976, 0.0980126, -0.1526041, 0.0997530, -0.2278818, 0.2472163
7: -0.0924693, 0.0744901, -0.1094564, 0.0927728, -0.1852421, 0.1839465
8: 0.5209497, 1.1044066, 0.5096486, 1.1371789, -0.5919204, 0.5755007
9: -0.0564147, 0.1264862, -0.0788770, 0.1341128, -0.1905275, 0.2053632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
time: 0.83 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
time: 0.92 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1915396, 0.1989554, -0.1857528, 0.1969411, -0.3868687, 0.3832457
1: -0.1139856, 0.0956200, -0.1103954, 0.0924685, -0.2064541, 0.2060154
2: -0.1142173, 0.1307839, -0.1097248, 0.1270900, -0.2413073, 0.2405087
3: -0.0913179, 0.1187139, -0.0877349, 0.1157546, -0.2070725, 0.2064488
4: -0.1136799, 0.0762545, -0.1101119, 0.0733279, -0.1870078, 0.1863664
5: -0.1277903, 0.1215110, -0.1230468, 0.1180900, -0.2458803, 0.2445578
6: -0.1567937, 0.1007414, -0.1520598, 0.0990683, -0.2524282, 0.2493670
7: -0.1128079, 0.0962646, -0.1090471, 0.0922506, -0.2050586, 0.2053117
8: 0.5073789, 1.1438251, 0.5111675, 1.1365066, -0.6071262, 0.6104932
9: -0.0831536, 0.1373781, -0.0783470, 0.1335798, -0.2167334, 0.2157251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3647722, upper bound: 0.3576401
time: 0.90 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3647722, upper bound: 0.3576401
time: 0.96 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1864232, 0.1975511, -0.1702424, 0.2241769, -0.4089561, 0.3666861
1: -0.1108270, 0.0928625, -0.1017978, 0.0855294, -0.1963564, 0.1946603
2: -0.1102498, 0.1276012, -0.0971790, 0.1218757, -0.2321256, 0.2247802
3: -0.0880790, 0.1161532, -0.0758365, 0.1233683, -0.2114473, 0.1919897
4: -0.1106712, 0.0737457, -0.1047361, 0.0676941, -0.1783653, 0.1784818
5: -0.1235694, 0.1185036, -0.1101099, 0.1094930, -0.2330624, 0.2286135
6: -0.1526041, 0.0997530, -0.1391321, 0.1138505, -0.2630379, 0.2357084
7: -0.1094564, 0.0927728, -0.0983962, 0.0835397, -0.1929960, 0.1911690
8: 0.5096486, 1.1371789, 0.4715714, 1.1102450, -0.5840526, 0.6469877
9: -0.0788770, 0.1341128, -0.0667616, 0.1329898, -0.2118668, 0.2008744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2807649, upper bound: 0.2915878
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2442435, upper bound: 0.2408333
time: 0.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.39 seconds
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.3455174, upper bound: 0.3598232
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.3647722, upper bound: 0.3576401
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.3647722, upper bound: 0.3576401
IS_B2_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.2807649, upper bound: 0.2915878
IS_B2_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 2.39
Output dim: 8, lower bound: -0.2442435, upper bound: 0.2408333

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1748026, 0.1503645, -0.1857528, 0.1969411, -0.3700645, 0.3331161
1: -0.1022011, 0.0848079, -0.1103954, 0.0924685, -0.1946696, 0.1952033
2: -0.1019022, 0.1168443, -0.1097248, 0.1270900, -0.2289922, 0.2265691
3: -0.0835326, 0.1053720, -0.0877349, 0.1157546, -0.1992871, 0.1931069
4: -0.0988611, 0.0655613, -0.1101119, 0.0733279, -0.1721890, 0.1756732
5: -0.1140495, 0.1112707, -0.1230468, 0.1180900, -0.2321395, 0.2343175
6: -0.1436899, 0.0908413, -0.1520598, 0.0990683, -0.2393304, 0.2394750
7: -0.1027083, 0.0827986, -0.1090471, 0.0922506, -0.1949589, 0.1918457
8: 0.5571116, 1.1294914, 0.5111675, 1.1365066, -0.5512652, 0.5936201
9: -0.0675066, 0.1248847, -0.0783470, 0.1335798, -0.2010863, 0.2032317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3040992, upper bound: 0.3107902
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2465303, upper bound: 0.2962061
time: 0.67 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2778805, upper bound: 0.3218405
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3457573, upper bound: 0.3634043
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1748026, 0.1503645, -0.1702424, 0.2241769, -0.3972729, 0.3177204
1: -0.1022011, 0.0848079, -0.1017978, 0.0855294, -0.1877304, 0.1866058
2: -0.1019022, 0.1168443, -0.0971790, 0.1218757, -0.2237779, 0.2140233
3: -0.0835326, 0.1053720, -0.0758365, 0.1233683, -0.2069008, 0.1812086
4: -0.0988611, 0.0655613, -0.1047361, 0.0676941, -0.1665552, 0.1702974
5: -0.1140495, 0.1112707, -0.1101099, 0.1094930, -0.2235425, 0.2213806
6: -0.1436899, 0.0908413, -0.1391321, 0.1138505, -0.2541329, 0.2265450
7: -0.1027083, 0.0827986, -0.0983962, 0.0835397, -0.1862479, 0.1811948
8: 0.5571116, 1.1294914, 0.4715714, 1.1102450, -0.5275264, 0.6366770
9: -0.0675066, 0.1248847, -0.0667616, 0.1329898, -0.2004964, 0.1916463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A1_B2_B1

### Relational analysis result of IS_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3040992, upper bound: 0.3107902
time: 0.80 seconds

## Relational analysis of IS_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2465303, upper bound: 0.2962061
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2778805, upper bound: 0.3218405
time: 0.75 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3457573, upper bound: 0.3634043
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1600679, 0.1800335, -0.1857528, 0.1969411, -0.3554254, 0.3631053
1: -0.0940157, 0.0782800, -0.1103954, 0.0924685, -0.1864842, 0.1886754
2: -0.0898781, 0.1102740, -0.1097248, 0.1270900, -0.2169681, 0.2199988
3: -0.0721118, 0.1113509, -0.0877349, 0.1157546, -0.1878664, 0.1990858
4: -0.0936740, 0.0601832, -0.1101119, 0.0733279, -0.1670019, 0.1702951
5: -0.1015542, 0.1030096, -0.1230468, 0.1180900, -0.2196442, 0.2260564
6: -0.1312976, 0.0980126, -0.1520598, 0.0990683, -0.2269439, 0.2466763
7: -0.0924693, 0.0744901, -0.1090471, 0.0922506, -0.1847199, 0.1835372
8: 0.5209497, 1.1044066, 0.5111675, 1.1365066, -0.5910249, 0.5710881
9: -0.0564147, 0.1264862, -0.0783470, 0.1335798, -0.1899944, 0.2048332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2210657, upper bound: 0.2602408
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 245

## Relational analysis of IS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 245

## Relational analysis of IS_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 116
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 121
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 140
type: A, layer: 3, pos: 142
type: B, layer: 3, pos: 177
type: A, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 142
type: A, layer: 3, pos: 248

Time for candidate selection: 12.69 seconds

### Candidate
type: B, layer: 3, pos: 116

## Relational analysis of IS_B2_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3309002, upper bound: 0.3462213
time: 0.78 seconds

## Relational analysis of IS_B2_A1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3382464, upper bound: 0.3531583
time: 0.92 seconds

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1600679, 0.1800335, -0.1702424, 0.2241769, -0.3824415, 0.3475792
1: -0.0940157, 0.0782800, -0.1017978, 0.0855294, -0.1795451, 0.1800779
2: -0.0898781, 0.1102740, -0.0971790, 0.1218757, -0.2117538, 0.2074530
3: -0.0721118, 0.1113509, -0.0758365, 0.1233683, -0.1954800, 0.1871875
4: -0.0936740, 0.0601832, -0.1047361, 0.0676941, -0.1613682, 0.1649193
5: -0.1015542, 0.1030096, -0.1101099, 0.1094930, -0.2110472, 0.2131195
6: -0.1312976, 0.0980126, -0.1391321, 0.1138505, -0.2417296, 0.2337085
7: -0.0924693, 0.0744901, -0.0983962, 0.0835397, -0.1760090, 0.1728863
8: 0.5209497, 1.1044066, 0.4715714, 1.1102450, -0.5666361, 0.6132724
9: -0.0564147, 0.1264862, -0.0667616, 0.1329898, -0.1894044, 0.1932478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2210657, upper bound: 0.2602408
time: 0.74 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 245

## Relational analysis of IS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 245

## Relational analysis of IS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 250
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 71
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 255
type: A, layer: 3, pos: 121
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 142
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: A, layer: 3, pos: 248

Time for candidate selection: 13.18 seconds

### Candidate
type: B, layer: 3, pos: 116

## Relational analysis of IS_B2_A1_A2_B2_B1

### Relational analysis result of IS_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3309002, upper bound: 0.3462213
time: 0.99 seconds

## Relational analysis of IS_B2_A1_A2_B2_B2

### Relational analysis result of IS_B2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3382464, upper bound: 0.3531583
time: 0.94 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1857528, 0.1969411, -0.1857528, 0.1969411, -0.3810637, 0.3810639
1: -0.1103954, 0.0924685, -0.1103954, 0.0924685, -0.2028639, 0.2028639
2: -0.1097248, 0.1270900, -0.1097248, 0.1270900, -0.2368148, 0.2368148
3: -0.0877349, 0.1157546, -0.0877349, 0.1157546, -0.2034895, 0.2034895
4: -0.1101119, 0.0733279, -0.1101119, 0.0733279, -0.1834398, 0.1834398
5: -0.1230468, 0.1180900, -0.1230468, 0.1180900, -0.2411368, 0.2411368
6: -0.1520598, 0.0990683, -0.1520598, 0.0990683, -0.2476953, 0.2476952
7: -0.1090471, 0.0922506, -0.1090471, 0.0922506, -0.2012978, 0.2012978
8: 0.5111675, 1.1365066, 0.5111675, 1.1365066, -0.6030359, 0.6030364
9: -0.0783470, 0.1335798, -0.0783470, 0.1335798, -0.2119268, 0.2119268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3273165, upper bound: 0.3283743
time: 0.91 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2895104, upper bound: 0.2760324
time: 0.61 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1702424, 0.2241769, -0.1857528, 0.1969411, -0.3656683, 0.4082727
1: -0.1017978, 0.0855294, -0.1103954, 0.0924685, -0.1942664, 0.1959248
2: -0.0971790, 0.1218757, -0.1097248, 0.1270900, -0.2242690, 0.2316005
3: -0.0758365, 0.1233683, -0.0877349, 0.1157546, -0.1915911, 0.2111031
4: -0.1047361, 0.0676941, -0.1101119, 0.0733279, -0.1780640, 0.1778060
5: -0.1101099, 0.1094930, -0.1230468, 0.1180900, -0.2281999, 0.2325398
6: -0.1391321, 0.1138505, -0.1520598, 0.0990683, -0.2347653, 0.2624978
7: -0.0983962, 0.0835397, -0.1090471, 0.0922506, -0.1906469, 0.1925868
8: 0.4715714, 1.1102450, 0.5111675, 1.1365066, -0.6460915, 0.5792990
9: -0.0667616, 0.1329898, -0.0783470, 0.1335798, -0.2003414, 0.2113368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3073168, upper bound: 0.2862734
time: 0.65 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2895104, upper bound: 0.2760324
time: 0.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.61 seconds
IS_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.2778805, upper bound: 0.3218405
IS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3457573, upper bound: 0.3634043
IS_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.2778805, upper bound: 0.3218405
IS_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3457573, upper bound: 0.3634043
IS_B2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3309002, upper bound: 0.3462213
IS_B2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3382464, upper bound: 0.3531583
IS_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3309002, upper bound: 0.3462213
IS_B2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3382464, upper bound: 0.3531583
IS_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3273165, upper bound: 0.3283743
IS_B2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.2895104, upper bound: 0.2760324
IS_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.3073168, upper bound: 0.2862734
IS_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.61
Output dim: 8, lower bound: -0.2895104, upper bound: 0.2760324

## BFS IS instance: IS_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1745266, 0.1488576, -0.1857528, 0.1969411, -0.3697891, 0.3314025
1: -0.1020164, 0.0846543, -0.1103954, 0.0924685, -0.1944849, 0.1950497
2: -0.1017166, 0.1166485, -0.1097248, 0.1270900, -0.2288065, 0.2263733
3: -0.0834297, 0.1051395, -0.0877349, 0.1157546, -0.1991842, 0.1928743
4: -0.0986531, 0.0653850, -0.1101119, 0.0733279, -0.1719809, 0.1754969
5: -0.1138667, 0.1111073, -0.1230468, 0.1180900, -0.2319567, 0.2341541
6: -0.1435070, 0.0897572, -0.1520598, 0.0990683, -0.2391476, 0.2381918
7: -0.1025700, 0.0825600, -0.1090471, 0.0922506, -0.1948207, 0.1916071
8: 0.5609969, 1.1292918, 0.5111675, 1.1365066, -0.5446825, 0.5934117
9: -0.0672926, 0.1246840, -0.0783470, 0.1335798, -0.2008724, 0.2030310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3412967, upper bound: 0.3450604
time: 0.87 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3285942, upper bound: 0.3433527
time: 0.88 seconds

## BFS IS instance: IS_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1745266, 0.1488576, -0.1702424, 0.2241769, -0.3969976, 0.3160405
1: -0.1020164, 0.0846543, -0.1017978, 0.0855294, -0.1875458, 0.1864522
2: -0.1017166, 0.1166485, -0.0971790, 0.1218757, -0.2235923, 0.2138276
3: -0.0834297, 0.1051395, -0.0758365, 0.1233683, -0.2067979, 0.1809760
4: -0.0986531, 0.0653850, -0.1047361, 0.0676941, -0.1663472, 0.1701211
5: -0.1138667, 0.1111073, -0.1101099, 0.1094930, -0.2233597, 0.2212172
6: -0.1435070, 0.0897572, -0.1391321, 0.1138505, -0.2539500, 0.2252538
7: -0.1025700, 0.0825600, -0.0983962, 0.0835397, -0.1861097, 0.1809562
8: 0.5609969, 1.1292918, 0.4715714, 1.1102450, -0.5211372, 0.6364686
9: -0.0672926, 0.1246840, -0.0667616, 0.1329898, -0.2002824, 0.1914456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3024374, upper bound: 0.3089485
time: 0.90 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_B2_A1_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2447499, upper bound: 0.2947088
time: 0.71 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 245

## Relational analysis of IS_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 245

## Relational analysis of IS_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: A, layer: 3, pos: 59
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 121
type: A, layer: 3, pos: 121
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: B, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 142
type: B, layer: 3, pos: 248

Time for candidate selection: 11.81 seconds

### Candidate
type: B, layer: 3, pos: 116

## Relational analysis of IS_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3315910, upper bound: 0.3493200
time: 0.88 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3390049, upper bound: 0.3572287
time: 0.87 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.72 seconds
IS_B2_A1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.72
Output dim: 8, lower bound: -0.3412967, upper bound: 0.3450604
IS_B2_A1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 14.72
Output dim: 8, lower bound: -0.3285942, upper bound: 0.3433527
IS_B2_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.72
Output dim: 8, lower bound: -0.3315910, upper bound: 0.3493200
IS_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.72
Output dim: 8, lower bound: -0.3390049, upper bound: 0.3572287

## BFS IS instance: IS_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1745266, 0.1488576, -0.1687131, 0.2149181, -0.3878829, 0.3145111
1: -0.1020164, 0.0846543, -0.1005871, 0.0845264, -0.1865427, 0.1852414
2: -0.1017166, 0.1166485, -0.0961229, 0.1197526, -0.2214691, 0.2127714
3: -0.0834297, 0.1051395, -0.0754376, 0.1195531, -0.2029828, 0.1805770
4: -0.0986531, 0.0653850, -0.1030469, 0.0664640, -0.1651171, 0.1684318
5: -0.1138667, 0.1111073, -0.1089067, 0.1085317, -0.2223984, 0.2200140
6: -0.1435070, 0.0897572, -0.1381346, 0.1090102, -0.2491211, 0.2242557
7: -0.1025700, 0.0825600, -0.0976310, 0.0821825, -0.1847526, 0.1801910
8: 0.5609969, 1.1292918, 0.4855456, 1.1095729, -0.5202141, 0.6208205
9: -0.0672926, 0.1246840, -0.0653116, 0.1309905, -0.1982831, 0.1899956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: B, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 59
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 116
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 71
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 121
type: A, layer: 3, pos: 121
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: B, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 142
type: B, layer: 3, pos: 248

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3233571, upper bound: 0.3399925
time: 0.87 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_B2_A1_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3233571, upper bound: 0.3433337
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.04 seconds
IS_B2_A1_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.3233571, upper bound: 0.3399925
IS_B2_A1_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.04
Output dim: 8, lower bound: -0.3233571, upper bound: 0.3433337

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.99 + 110.83 = 113.83 seconds
