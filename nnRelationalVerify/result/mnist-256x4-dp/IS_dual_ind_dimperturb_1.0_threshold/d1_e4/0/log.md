## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.037597364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0949643, 0.0846384, -0.0949643, 0.0846384, -0.1796028, 0.1796028)
1: (-0.0607208, 0.0474269, -0.0607208, 0.0474269, -0.1081477, 0.1081477)
2: (-0.1246243, 0.0634956, -0.1246243, 0.0634956, -0.1881199, 0.1881199)
3: (0.9865127, 1.0434601, 0.9865127, 1.0434601, -0.0569474, 0.0569474)
4: (-0.0387707, 0.1049149, -0.0387707, 0.1049149, -0.1436856, 0.1436856)
5: (-0.0581584, 0.1370523, -0.0581584, 0.1370523, -0.1952107, 0.1952107)
6: (-0.1140255, 0.0998719, -0.1140255, 0.0998719, -0.2138974, 0.2138974)
7: (-0.0948062, 0.0074982, -0.0948062, 0.0074982, -0.1023044, 0.1023044)
8: (-0.0549271, 0.0979662, -0.0549271, 0.0979662, -0.1528933, 0.1528933)
9: (-0.0670059, 0.0810942, -0.0670059, 0.0810942, -0.1481002, 0.1481002)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 3.09 = 4.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0408667, upper bound: 0.0408667

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0405673, upper bound: 0.0402955
time: 2.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
time: 2.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.36 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.36
Output dim: 3, lower bound: -0.0405673, upper bound: 0.0402955
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.36
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0696140, 0.0585657, -0.0923202, 0.0819181, -0.1515321, 0.1508859
1: -0.0465374, 0.0365823, -0.0592436, 0.0462938, -0.0928312, 0.0958258
2: -0.0960034, 0.0439863, -0.1216386, 0.0614625, -0.1574659, 0.1656249
3: 0.9936385, 1.0356088, 0.9872570, 1.0426378, -0.0489993, 0.0483518
4: -0.0272908, 0.0755011, -0.0375746, 0.1018434, -0.1291341, 0.1130757
5: -0.0377102, 0.1128232, -0.0560224, 0.1345283, -0.1722385, 0.1688455
6: -0.0897659, 0.0676334, -0.1114936, 0.0965081, -0.1862740, 0.1791270
7: -0.0747000, 0.0012146, -0.0927079, 0.0067981, -0.0814981, 0.0939224
8: -0.0427751, 0.0650738, -0.0536578, 0.0945331, -0.1373082, 0.1187317
9: -0.0492805, 0.0637030, -0.0651546, 0.0792785, -0.1285590, 0.1288576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
time: 2.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
time: 2.58 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0565108, 0.0450152, -0.0751078, 0.0643354, -0.1208462, 0.1201230
1: -0.0391758, 0.0307427, -0.0496069, 0.0391389, -0.0783147, 0.0803496
2: -0.0807246, 0.0342839, -0.1025264, 0.0480599, -0.1287844, 0.1368103
3: 0.9972058, 1.0314329, 0.9921253, 1.0375504, -0.0403447, 0.0393075
4: -0.0215432, 0.0600531, -0.0296945, 0.0820377, -0.1035808, 0.0897476
5: -0.0272021, 0.0999487, -0.0422568, 0.1181504, -0.1453525, 0.1422054
6: -0.0769678, 0.0509182, -0.0952001, 0.0747651, -0.1517329, 0.1461183
7: -0.0642179, -0.0004016, -0.0790946, 0.0024410, -0.0666588, 0.0786930
8: -0.0364530, 0.0481090, -0.0455603, 0.0722544, -0.1087074, 0.0936692
9: -0.0402359, 0.0543274, -0.0530918, 0.0676570, -0.1078930, 0.1074192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397552, upper bound: 0.0400739
time: 2.12 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397538, upper bound: 0.0397538
time: 2.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 3, lower bound: -0.0402928, upper bound: 0.0402928
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 3, lower bound: -0.0397552, upper bound: 0.0400739
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 3, lower bound: -0.0397538, upper bound: 0.0397538

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0696140, 0.0585657, -0.0696140, 0.0585657, -0.1281797, 0.1281797
1: -0.0465374, 0.0365823, -0.0465374, 0.0365823, -0.0831197, 0.0831197
2: -0.0960034, 0.0439863, -0.0960034, 0.0439863, -0.1399896, 0.1399896
3: 0.9936385, 1.0356088, 0.9936385, 1.0356088, -0.0419703, 0.0419703
4: -0.0272908, 0.0755011, -0.0272908, 0.0755011, -0.1027919, 0.1027919
5: -0.0377102, 0.1128232, -0.0377102, 0.1128232, -0.1505333, 0.1505333
6: -0.0897659, 0.0676334, -0.0897659, 0.0676334, -0.1573994, 0.1573994
7: -0.0747000, 0.0012146, -0.0747000, 0.0012146, -0.0759146, 0.0759146
8: -0.0427751, 0.0650738, -0.0427751, 0.0650738, -0.1078490, 0.1078490
9: -0.0492805, 0.0637030, -0.0492805, 0.0637030, -0.1129835, 0.1129835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402232, upper bound: 0.0397580
time: 2.14 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400842, upper bound: 0.0397571
time: 3.34 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0696140, 0.0585657, -0.0565108, 0.0450152, -0.1146292, 0.1150765
1: -0.0465374, 0.0365823, -0.0391758, 0.0307427, -0.0772801, 0.0757581
2: -0.0960034, 0.0439863, -0.0807246, 0.0342839, -0.1302873, 0.1247108
3: 0.9936385, 1.0356088, 0.9972058, 1.0314329, -0.0377944, 0.0384030
4: -0.0272908, 0.0755011, -0.0215432, 0.0600531, -0.0873439, 0.0970443
5: -0.0377102, 0.1128232, -0.0272021, 0.0999487, -0.1376589, 0.1400252
6: -0.0897659, 0.0676334, -0.0769678, 0.0509182, -0.1406841, 0.1446012
7: -0.0747000, 0.0012146, -0.0642179, -0.0004016, -0.0742984, 0.0654324
8: -0.0427751, 0.0650738, -0.0364530, 0.0481090, -0.0908841, 0.1015268
9: -0.0492805, 0.0637030, -0.0402359, 0.0543274, -0.1036079, 0.1039389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402232, upper bound: 0.0397580
time: 2.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400842, upper bound: 0.0397571
time: 7.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0558250, 0.0443243, -0.0723343, 0.0614724, -0.1172974, 0.1166587
1: -0.0387921, 0.0304796, -0.0480554, 0.0380539, -0.0768459, 0.0785350
2: -0.0799999, 0.0337253, -0.0995707, 0.0457664, -0.1257663, 0.1332960
3: 0.9974039, 1.0312504, 0.9929337, 1.0367965, -0.0393926, 0.0383167
4: -0.0212149, 0.0592843, -0.0283419, 0.0789084, -0.1001233, 0.0876262
5: -0.0266599, 0.0993127, -0.0400219, 0.1155684, -0.1422282, 0.1393346
6: -0.0763383, 0.0500623, -0.0926114, 0.0712878, -0.1476262, 0.1426737
7: -0.0636824, -0.0005468, -0.0769189, 0.0016293, -0.0653118, 0.0763721
8: -0.0361420, 0.0472247, -0.0442890, 0.0686247, -0.1047666, 0.0915137
9: -0.0397507, 0.0538888, -0.0510966, 0.0658648, -0.1056155, 0.1049854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
time: 2.13 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
time: 2.08 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0550049, 0.0436525, -0.1277840, 0.1176255, -0.1726305, 0.1714365
1: -0.0383333, 0.0302028, -0.0791231, 0.0599273, -0.0982606, 0.1093259
2: -0.0791749, 0.0330657, -0.1592252, 0.0901367, -0.1693116, 0.1922909
3: 0.9976265, 1.0310214, 0.9770267, 1.0520164, -0.0543898, 0.0539947
4: -0.0208271, 0.0584422, -0.0544156, 0.1416642, -0.1624913, 0.1128578
5: -0.0260067, 0.0985595, -0.0839516, 0.1677361, -0.1937428, 0.1825111
6: -0.0755751, 0.0491446, -0.1441265, 0.1405656, -0.2161407, 0.1932711
7: -0.0630569, -0.0006938, -0.1204703, 0.0173695, -0.0804264, 0.1197766
8: -0.0357643, 0.0462891, -0.0696280, 0.1401767, -0.1759410, 0.1159170
9: -0.0391715, 0.0534456, -0.0902149, 0.1021638, -0.1413353, 0.1436604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391054, upper bound: 0.0393021
time: 2.03 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394364, upper bound: 0.0394364
time: 2.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0402232, upper bound: 0.0397580
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0400842, upper bound: 0.0397571
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0402232, upper bound: 0.0397580
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0400842, upper bound: 0.0397571
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0391054, upper bound: 0.0393021
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.63
Output dim: 3, lower bound: -0.0394364, upper bound: 0.0394364

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0668425, 0.0557111, -0.0688981, 0.0578326, -0.1246751, 0.1246092
1: -0.0449966, 0.0354954, -0.0461411, 0.0363026, -0.0812992, 0.0816365
2: -0.0930542, 0.0416954, -0.0952450, 0.0433949, -0.1364492, 0.1369404
3: 0.9944476, 1.0348458, 0.9938484, 1.0354127, -0.0409650, 0.0409974
4: -0.0259453, 0.0723664, -0.0269454, 0.0746912, -0.1006365, 0.0993118
5: -0.0354670, 0.1102619, -0.0371317, 0.1121647, -0.1476317, 0.1473936
6: -0.0871848, 0.0641510, -0.0891032, 0.0667348, -0.1539195, 0.1532542
7: -0.0725257, 0.0004283, -0.0741384, 0.0010090, -0.0735347, 0.0745667
8: -0.0415003, 0.0614468, -0.0424469, 0.0641402, -0.1056405, 0.1038938
9: -0.0472841, 0.0619192, -0.0487652, 0.0632455, -0.1105296, 0.1106844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401056, upper bound: 0.0397529
time: 2.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0403436, upper bound: 0.0401135
time: 3.47 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1221146, 0.1116300, -0.0680626, 0.0569737, -0.1790882, 0.1796926
1: -0.0759734, 0.0573093, -0.0456766, 0.0359662, -0.1119395, 0.1029859
2: -0.1525731, 0.0858038, -0.0943405, 0.0427181, -0.1952912, 0.1801443
3: 0.9786255, 1.0500169, 0.9940899, 1.0351741, -0.0565487, 0.0559270
4: -0.0518756, 0.1349362, -0.0265485, 0.0737385, -0.1256141, 0.1614847
5: -0.0791879, 0.1623531, -0.0364555, 0.1113865, -0.1905745, 0.1988086
6: -0.1385444, 0.1331798, -0.0883195, 0.0656804, -0.2042249, 0.2214993
7: -0.1159383, 0.0154710, -0.0734804, 0.0007828, -0.1167211, 0.0889514
8: -0.0667496, 0.1326979, -0.0420579, 0.0630501, -0.1297996, 0.1747558
9: -0.0862323, 0.0981658, -0.0481685, 0.0626987, -0.1489311, 0.1463343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 88

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399651, upper bound: 0.0397524
time: 2.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401129
time: 3.81 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0668425, 0.0557111, -0.0558250, 0.0443243, -0.1111668, 0.1115361
1: -0.0449966, 0.0354954, -0.0387921, 0.0304796, -0.0754762, 0.0742875
2: -0.0930542, 0.0416954, -0.0799999, 0.0337253, -0.1267795, 0.1216953
3: 0.9944476, 1.0348458, 0.9974039, 1.0312504, -0.0368027, 0.0374420
4: -0.0259453, 0.0723664, -0.0212149, 0.0592843, -0.0852296, 0.0935813
5: -0.0354670, 0.1102619, -0.0266599, 0.0993127, -0.1347797, 0.1369217
6: -0.0871848, 0.0641510, -0.0763383, 0.0500623, -0.1372470, 0.1404894
7: -0.0725257, 0.0004283, -0.0636824, -0.0005468, -0.0719788, 0.0641107
8: -0.0415003, 0.0614468, -0.0361420, 0.0472247, -0.0887251, 0.0975888
9: -0.0472841, 0.0619192, -0.0397507, 0.0538888, -0.1011729, 0.1016699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396482, upper bound: 0.0391085
time: 2.27 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399029, upper bound: 0.0394411
time: 2.17 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1221146, 0.1116300, -0.0550049, 0.0436525, -0.1657670, 0.1666349
1: -0.0759734, 0.0573093, -0.0383333, 0.0302028, -0.1061762, 0.0956426
2: -0.1525731, 0.0858038, -0.0791749, 0.0330657, -0.1856388, 0.1649787
3: 0.9786255, 1.0500169, 0.9976265, 1.0310214, -0.0523959, 0.0523903
4: -0.0518756, 0.1349362, -0.0208271, 0.0584422, -0.1103178, 0.1557633
5: -0.0791879, 0.1623531, -0.0260067, 0.0985595, -0.1777475, 0.1883598
6: -0.1385444, 0.1331798, -0.0755751, 0.0491446, -0.1876891, 0.2087549
7: -0.1159383, 0.0154710, -0.0630569, -0.0006938, -0.1152445, 0.0785279
8: -0.0667496, 0.1326979, -0.0357643, 0.0462891, -0.1130386, 0.1684622
9: -0.0862323, 0.0981658, -0.0391715, 0.0534456, -0.1396779, 0.1373373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395507, upper bound: 0.0391076
time: 2.42 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397734, upper bound: 0.0394402
time: 2.49 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0216828, 0.0252347, -0.0529565, 0.0420325, -0.0637153, 0.0781913
1: -0.0169645, 0.0182203, -0.0371876, 0.0307413, -0.0477058, 0.0554079
2: -0.0403198, 0.0170826, -0.0792343, 0.0299837, -0.0703035, 0.0963169
3: 1.0034505, 1.0202062, 0.9985059, 1.0318223, -0.0283718, 0.0217003
4: -0.0069501, 0.0278998, -0.0190718, 0.0572570, -0.0642071, 0.0469715
5: -0.0050175, 0.0634644, -0.0248710, 0.0974278, -0.1024453, 0.0883355
6: -0.0436630, 0.0131957, -0.0748795, 0.0473195, -0.0909825, 0.0880752
7: -0.0403072, -0.0008022, -0.0617853, -0.0016035, -0.0387038, 0.0609378
8: -0.0198662, 0.0141133, -0.0356664, 0.0437783, -0.0636446, 0.0497797
9: -0.0208446, 0.0332611, -0.0374086, 0.0534823, -0.0743268, 0.0706697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
time: 2.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
time: 2.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0327194, 0.0292314, -0.0649940, 0.0540845, -0.0868040, 0.0942254
1: -0.0247176, 0.0229697, -0.0439411, 0.0352587, -0.0599763, 0.0669107
2: -0.0559002, 0.0173047, -0.0918396, 0.0397899, -0.0956902, 0.1091443
3: 1.0024749, 1.0246326, 0.9950500, 1.0348843, -0.0324094, 0.0295826
4: -0.0113277, 0.0373071, -0.0248299, 0.0706899, -0.0820176, 0.0621370
5: -0.0100238, 0.0764180, -0.0342502, 0.1087181, -0.1187419, 0.1106683
6: -0.0549601, 0.0260977, -0.0858776, 0.0621693, -0.1171294, 0.1119752
7: -0.0467530, -0.0012066, -0.0711823, 0.0000034, -0.0467564, 0.0699442
8: -0.0248449, 0.0244438, -0.0409939, 0.0591828, -0.0840276, 0.0654377
9: -0.0254109, 0.0416363, -0.0459050, 0.0611662, -0.0865771, 0.0875413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
time: 2.13 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
time: 2.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0212073, 0.0250504, -0.1085135, 0.0982499, -0.1194573, 0.1335639
1: -0.0166401, 0.0179915, -0.0683206, 0.0526661, -0.0693062, 0.0863121
2: -0.0396097, 0.0170727, -0.1390585, 0.0743607, -0.1139704, 0.1561312
3: 1.0034908, 1.0199831, 0.9826252, 1.0470580, -0.0435672, 0.0373578
4: -0.0067523, 0.0274882, -0.0451505, 0.1201636, -0.1269159, 0.0726387
5: -0.0047710, 0.0629229, -0.0688031, 0.1497925, -0.1545635, 0.1317260
6: -0.0431464, 0.0126005, -0.1265138, 0.1167045, -0.1598510, 0.1391143
7: -0.0400123, -0.0008940, -0.1054274, 0.0110111, -0.0510234, 0.1044800
8: -0.0196435, 0.0136412, -0.0610379, 0.1153680, -0.1350116, 0.0746791
9: -0.0206415, 0.0328816, -0.0765444, 0.0898936, -0.1105351, 0.1094260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390011, upper bound: 0.0391238
time: 3.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390303, upper bound: 0.0392255
time: 2.34 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0321044, 0.0290424, -0.1203999, 0.1101993, -0.1423037, 0.1494422
1: -0.0242587, 0.0227350, -0.0749818, 0.0571352, -0.0813940, 0.0977168
2: -0.0551016, 0.0172945, -0.1514713, 0.0841131, -0.1392147, 0.1687658
3: 1.0025700, 1.0244035, 0.9791689, 1.0501125, -0.0475425, 0.0452346
4: -0.0111260, 0.0367799, -0.0508782, 0.1334118, -0.1445378, 0.0876581
5: -0.0097675, 0.0756730, -0.0781490, 0.1608412, -0.1706087, 0.1538220
6: -0.0543732, 0.0254884, -0.1373643, 0.1314214, -0.1857946, 0.1628527
7: -0.0464502, -0.0013041, -0.1146975, 0.0148268, -0.0612770, 0.1133473
8: -0.0245136, 0.0239522, -0.0663340, 0.1306713, -0.1551849, 0.0902863
9: -0.0251488, 0.0412468, -0.0849822, 0.0974424, -0.1225912, 0.1262290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393374, upper bound: 0.0392576
time: 2.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
time: 2.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.73 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0401056, upper bound: 0.0397529
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0403436, upper bound: 0.0401135
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0399651, upper bound: 0.0397524
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0401129, upper bound: 0.0401129
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0396482, upper bound: 0.0391085
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0399029, upper bound: 0.0394411
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0395507, upper bound: 0.0391076
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0397734, upper bound: 0.0394402
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0390011, upper bound: 0.0391238
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0390303, upper bound: 0.0392255
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0393374, upper bound: 0.0392576
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.73
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0473184, 0.0375590, -0.0293278, 0.0283580, -0.0756764, 0.0668868
1: -0.0340481, 0.0285257, -0.0221011, 0.0221390, -0.0561871, 0.0506268
2: -0.0730296, 0.0257835, -0.0521207, 0.0172129, -0.0902425, 0.0779042
3: 0.9999244, 1.0297885, 1.0029987, 1.0240458, -0.0241215, 0.0267898
4: -0.0165945, 0.0513115, -0.0099069, 0.0348776, -0.0514721, 0.0612184
5: -0.0201543, 0.0921151, -0.0092713, 0.0725738, -0.0927282, 0.1013864
6: -0.0692368, 0.0409976, -0.0524611, 0.0224128, -0.0916496, 0.0934587
7: -0.0574200, -0.0018060, -0.0452337, -0.0014012, -0.0559682, 0.0434278
8: -0.0327722, 0.0375090, -0.0236245, 0.0214766, -0.0542488, 0.0611335
9: -0.0334622, 0.0502693, -0.0242300, 0.0395894, -0.0730516, 0.0744993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 134

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399517, upper bound: 0.0396105
time: 2.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399471, upper bound: 0.0396106
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0595775, 0.0483798, -0.0439685, 0.0350896, -0.0946671, 0.0923483
1: -0.0409217, 0.0327340, -0.0321587, 0.0275680, -0.0684897, 0.0648927
2: -0.0853977, 0.0357665, -0.0699056, 0.0230057, -0.1084034, 0.1056721
3: 0.9965381, 1.0329731, 1.0006377, 1.0289888, -0.0324507, 0.0323354
4: -0.0224595, 0.0642284, -0.0149608, 0.0480891, -0.0705486, 0.0791892
5: -0.0297583, 0.1034627, -0.0175778, 0.0890832, -0.1188415, 0.1210405
6: -0.0805103, 0.0551392, -0.0662205, 0.0374685, -0.1179787, 0.1213597
7: -0.0668376, -0.0009325, -0.0549176, -0.0015909, -0.0652467, 0.0539850
8: -0.0382568, 0.0520879, -0.0313172, 0.0339080, -0.0721648, 0.0834051
9: -0.0421354, 0.0572616, -0.0311023, 0.0486688, -0.0908042, 0.0883639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401580, upper bound: 0.0399998
time: 2.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402796, upper bound: 0.0400522
time: 2.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1024909, 0.0919174, -0.0288377, 0.0281667, -0.1306577, 0.1207551
1: -0.0649670, 0.0498685, -0.0217673, 0.0219028, -0.0868698, 0.0716358
2: -0.1319299, 0.0698654, -0.0513877, 0.0172027, -0.1491326, 0.1212531
3: 0.9843022, 1.0449358, 1.0030406, 1.0238159, -0.0395136, 0.0418953
4: -0.0425114, 0.1129876, -0.0097056, 0.0344507, -0.0769621, 0.1226932
5: -0.0637939, 0.1439905, -0.0090147, 0.0720095, -0.1358034, 0.1530052
6: -0.1205648, 0.1088765, -0.0519263, 0.0218048, -0.1423696, 0.1608028
7: -0.1006028, 0.0094477, -0.0449281, -0.0014966, -0.0990483, 0.0543758
8: -0.0579819, 0.1074714, -0.0233942, 0.0209869, -0.0789688, 0.1308656
9: -0.0723464, 0.0855769, -0.0240202, 0.0391951, -0.1115415, 0.1095972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397327, upper bound: 0.0396240
time: 2.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399039, upper bound: 0.0396832
time: 2.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1147854, 0.1042883, -0.0431291, 0.0344751, -0.1492604, 0.1474174
1: -0.0718642, 0.0545541, -0.0316904, 0.0273057, -0.0991699, 0.0862445
2: -0.1449149, 0.0798298, -0.0690878, 0.0223292, -0.1672441, 0.1489176
3: 0.9807511, 1.0481323, 1.0008309, 1.0287516, -0.0480005, 0.0473014
4: -0.0483670, 0.1267659, -0.0145635, 0.0472652, -0.0956322, 0.1413294
5: -0.0734463, 0.1555286, -0.0169079, 0.0883203, -0.1617666, 0.1724365
6: -0.1318637, 0.1241068, -0.0654359, 0.0365824, -0.1684461, 0.1895427
7: -0.1102248, 0.0129447, -0.0542854, -0.0016995, -0.1085248, 0.0672300
8: -0.0634813, 0.1232869, -0.0309273, 0.0330125, -0.0964938, 0.1542143
9: -0.0810491, 0.0934995, -0.0305058, 0.0482596, -0.1293088, 0.1240053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 112

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398801, upper bound: 0.0399943
time: 2.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400515, upper bound: 0.0400515
time: 2.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0473184, 0.0375590, -0.0216828, 0.0252347, -0.0725532, 0.0592418
1: -0.0340481, 0.0285257, -0.0169645, 0.0182203, -0.0522684, 0.0454902
2: -0.0730296, 0.0257835, -0.0403198, 0.0170826, -0.0901122, 0.0661032
3: 0.9999244, 1.0297885, 1.0034505, 1.0202062, -0.0202819, 0.0263380
4: -0.0165945, 0.0513115, -0.0069501, 0.0278998, -0.0444942, 0.0582616
5: -0.0201543, 0.0921151, -0.0050175, 0.0634644, -0.0836188, 0.0971326
6: -0.0692368, 0.0409976, -0.0436630, 0.0131957, -0.0824325, 0.0846606
7: -0.0574200, -0.0018060, -0.0403072, -0.0008022, -0.0565702, 0.0385013
8: -0.0327722, 0.0375090, -0.0198662, 0.0141133, -0.0468855, 0.0573752
9: -0.0334622, 0.0502693, -0.0208446, 0.0332611, -0.0667233, 0.0711138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 134

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395020, upper bound: 0.0389794
time: 2.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394948, upper bound: 0.0389794
time: 2.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0595775, 0.0483798, -0.0327194, 0.0292314, -0.0888090, 0.0810993
1: -0.0409217, 0.0327340, -0.0247176, 0.0229697, -0.0638913, 0.0574516
2: -0.0853977, 0.0357665, -0.0559002, 0.0173047, -0.1027024, 0.0916667
3: 0.9965381, 1.0329731, 1.0024749, 1.0246326, -0.0280945, 0.0304981
4: -0.0224595, 0.0642284, -0.0113277, 0.0373071, -0.0597667, 0.0755561
5: -0.0297583, 0.1034627, -0.0100238, 0.0764180, -0.1061763, 0.1134866
6: -0.0805103, 0.0551392, -0.0549601, 0.0260977, -0.1066079, 0.1100993
7: -0.0668376, -0.0009325, -0.0467530, -0.0012066, -0.0655966, 0.0458205
8: -0.0382568, 0.0520879, -0.0248449, 0.0244438, -0.0627005, 0.0769327
9: -0.0421354, 0.0572616, -0.0254109, 0.0416363, -0.0837718, 0.0826725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397377, upper bound: 0.0393476
time: 2.47 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398371, upper bound: 0.0393660
time: 2.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1024909, 0.0919174, -0.0212073, 0.0250504, -0.1275413, 0.1131248
1: -0.0649670, 0.0498685, -0.0166401, 0.0179915, -0.0829585, 0.0665086
2: -0.1319299, 0.0698654, -0.0396097, 0.0170727, -0.1490026, 0.1094751
3: 0.9843022, 1.0449358, 1.0034908, 1.0199831, -0.0356808, 0.0414450
4: -0.0425114, 0.1129876, -0.0067523, 0.0274882, -0.0699996, 0.1197399
5: -0.0637939, 0.1439905, -0.0047710, 0.0629229, -0.1267167, 0.1487616
6: -0.1205648, 0.1088765, -0.0431464, 0.0126005, -0.1331652, 0.1520229
7: -0.1006028, 0.0094477, -0.0400123, -0.0008940, -0.0996535, 0.0494601
8: -0.0579819, 0.1074714, -0.0196435, 0.0136412, -0.0716231, 0.1271149
9: -0.0723464, 0.0855769, -0.0206415, 0.0328816, -0.1052280, 0.1062185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393666, upper bound: 0.0390069
time: 2.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394790, upper bound: 0.0390326
time: 2.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1147854, 0.1042883, -0.0321044, 0.0290424, -0.1438277, 0.1363927
1: -0.0718642, 0.0545541, -0.0242587, 0.0227350, -0.0945992, 0.0788129
2: -0.1449149, 0.0798298, -0.0551016, 0.0172945, -0.1622094, 0.1349314
3: 0.9807511, 1.0481323, 1.0025700, 1.0244035, -0.0436524, 0.0455623
4: -0.0483670, 0.1267659, -0.0111260, 0.0367799, -0.0851470, 0.1378919
5: -0.0734463, 0.1555286, -0.0097675, 0.0756730, -0.1491193, 0.1652961
6: -0.1318637, 0.1241068, -0.0543732, 0.0254884, -0.1573522, 0.1784800
7: -0.1102248, 0.0129447, -0.0464502, -0.0013041, -0.1088711, 0.0593949
8: -0.0634813, 0.1232869, -0.0245136, 0.0239522, -0.0874335, 0.1478006
9: -0.0810491, 0.0934995, -0.0251488, 0.0412468, -0.1222959, 0.1186483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 112

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395855, upper bound: 0.0393438
time: 2.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
time: 2.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0216828, 0.0252347, -0.0473184, 0.0375590, -0.0592418, 0.0725532
1: -0.0169645, 0.0182203, -0.0340481, 0.0285257, -0.0454902, 0.0522684
2: -0.0403198, 0.0170826, -0.0730296, 0.0257835, -0.0661032, 0.0901122
3: 1.0034505, 1.0202062, 0.9999244, 1.0297885, -0.0263380, 0.0202819
4: -0.0069501, 0.0278998, -0.0165945, 0.0513115, -0.0582616, 0.0444942
5: -0.0050175, 0.0634644, -0.0201543, 0.0921151, -0.0971326, 0.0836188
6: -0.0436630, 0.0131957, -0.0692368, 0.0409976, -0.0846606, 0.0824325
7: -0.0403072, -0.0008022, -0.0574200, -0.0018060, -0.0385013, 0.0565702
8: -0.0198662, 0.0141133, -0.0327722, 0.0375090, -0.0573752, 0.0468855
9: -0.0208446, 0.0332611, -0.0334622, 0.0502693, -0.0711138, 0.0667233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 229

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
time: 2.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
time: 2.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0216828, 0.0252347, -0.0356858, 0.0306686, -0.0523515, 0.0609205
1: -0.0169645, 0.0182203, -0.0269274, 0.0241701, -0.0411346, 0.0451477
2: -0.0403198, 0.0170826, -0.0597549, 0.0186891, -0.0590089, 0.0768375
3: 1.0034505, 1.0202062, 1.0020181, 1.0257523, -0.0223018, 0.0181881
4: -0.0069501, 0.0278998, -0.0122964, 0.0399373, -0.0468874, 0.0401962
5: -0.0050175, 0.0634644, -0.0117912, 0.0800145, -0.0850320, 0.0752556
6: -0.0436630, 0.0131957, -0.0579057, 0.0290308, -0.0726939, 0.0711014
7: -0.0403072, -0.0008022, -0.0487722, -0.0014500, -0.0388547, 0.0479266
8: -0.0198662, 0.0141133, -0.0266855, 0.0267920, -0.0466583, 0.0407988
9: -0.0208446, 0.0332611, -0.0266701, 0.0435678, -0.0644124, 0.0599312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 229

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
time: 2.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
time: 2.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0327194, 0.0292314, -0.0595775, 0.0483798, -0.0810993, 0.0888090
1: -0.0247176, 0.0229697, -0.0409217, 0.0327340, -0.0574516, 0.0638913
2: -0.0559002, 0.0173047, -0.0853977, 0.0357665, -0.0916667, 0.1027024
3: 1.0024749, 1.0246326, 0.9965381, 1.0329731, -0.0304981, 0.0280945
4: -0.0113277, 0.0373071, -0.0224595, 0.0642284, -0.0755561, 0.0597667
5: -0.0100238, 0.0764180, -0.0297583, 0.1034627, -0.1134866, 0.1061763
6: -0.0549601, 0.0260977, -0.0805103, 0.0551392, -0.1100993, 0.1066079
7: -0.0467530, -0.0012066, -0.0668376, -0.0009325, -0.0458205, 0.0655966
8: -0.0248449, 0.0244438, -0.0382568, 0.0520879, -0.0769327, 0.0627005
9: -0.0254109, 0.0416363, -0.0421354, 0.0572616, -0.0826725, 0.0837718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
time: 2.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
time: 2.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0327194, 0.0292314, -0.0465183, 0.0373918, -0.0701113, 0.0757498
1: -0.0247176, 0.0229697, -0.0335749, 0.0276616, -0.0523792, 0.0565445
2: -0.0559002, 0.0173047, -0.0710695, 0.0261156, -0.0820158, 0.0883743
3: 1.0024749, 1.0246326, 0.9998189, 1.0287998, -0.0263249, 0.0248137
4: -0.0113277, 0.0373071, -0.0167343, 0.0501887, -0.0615164, 0.0540414
5: -0.0100238, 0.0764180, -0.0193297, 0.0908468, -0.1008706, 0.0957478
6: -0.0549601, 0.0260977, -0.0677604, 0.0401952, -0.0951553, 0.0938581
7: -0.0467530, -0.0012066, -0.0566827, -0.0012160, -0.0455370, 0.0554464
8: -0.0248449, 0.0244438, -0.0319545, 0.0371330, -0.0619778, 0.0563982
9: -0.0254109, 0.0416363, -0.0331519, 0.0493323, -0.0747432, 0.0747882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
time: 2.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
time: 2.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0179789, 0.0238478, -0.1001221, 0.0897806, -0.1077596, 0.1239699
1: -0.0144489, 0.0165294, -0.0636103, 0.0495779, -0.0640269, 0.0801397
2: -0.0349618, 0.0169927, -0.1303466, 0.0674166, -0.1023784, 0.1473392
3: 1.0037543, 1.0187299, 0.9850950, 1.0449622, -0.0412079, 0.0336350
4: -0.0053322, 0.0249047, -0.0410536, 0.1108585, -0.1161908, 0.0659584
5: -0.0036129, 0.0593339, -0.0622334, 0.1419284, -0.1455413, 0.1215673
6: -0.0400352, 0.0084516, -0.1188241, 0.1064136, -0.1464488, 0.1272757
7: -0.0380665, -0.0010053, -0.0988765, 0.0086156, -0.0466821, 0.0978091
8: -0.0183787, 0.0104304, -0.0573538, 0.1045519, -0.1229306, 0.0677842
9: -0.0194084, 0.0303634, -0.0705357, 0.0845647, -0.1039731, 0.1008991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 156

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
time: 2.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
time: 2.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0203675, 0.0247481, -0.1042095, 0.0937921, -0.1141596, 0.1289576
1: -0.0160615, 0.0176227, -0.0659058, 0.0509598, -0.0670212, 0.0835285
2: -0.0384187, 0.0170492, -0.1343883, 0.0708728, -0.1092915, 0.1514375
3: 1.0035615, 1.0196501, 0.9838902, 1.0458777, -0.0423162, 0.0357599
4: -0.0063695, 0.0268282, -0.0430810, 0.1152810, -0.1216505, 0.0699092
5: -0.0044350, 0.0620220, -0.0653560, 0.1456878, -0.1501228, 0.1273780
6: -0.0423404, 0.0114958, -0.1224358, 0.1113529, -0.1536933, 0.1339316
7: -0.0395188, -0.0009217, -0.1020320, 0.0097847, -0.0493035, 0.1010571
8: -0.0192973, 0.0127833, -0.0590791, 0.1097480, -0.1290453, 0.0718625
9: -0.0203190, 0.0322437, -0.0734478, 0.0870376, -0.1073566, 0.1056915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 229

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
time: 2.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389113, upper bound: 0.0390831
time: 2.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0284745, 0.0278481, -0.1119975, 0.1017796, -0.1302541, 0.1398456
1: -0.0216257, 0.0212799, -0.0702647, 0.0540942, -0.0757199, 0.0915447
2: -0.0502256, 0.0172155, -0.1428189, 0.0771507, -0.1273763, 0.1600344
3: 1.0030092, 1.0230105, 0.9816423, 1.0480590, -0.0450498, 0.0413682
4: -0.0097405, 0.0337567, -0.0467721, 0.1241359, -0.1338764, 0.0805288
5: -0.0081980, 0.0714750, -0.0716093, 0.1529770, -0.1611750, 0.1430843
6: -0.0508575, 0.0214233, -0.1297130, 0.1211560, -0.1720135, 0.1511362
7: -0.0445212, -0.0014287, -0.1081489, 0.0118673, -0.0563885, 0.1066641
8: -0.0227324, 0.0206989, -0.0626796, 0.1198800, -0.1426123, 0.0833785
9: -0.0236525, 0.0387552, -0.0789721, 0.0921490, -0.1158015, 0.1177273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
time: 1.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
time: 2.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0310245, 0.0287394, -0.1160757, 0.1057376, -0.1367622, 0.1448151
1: -0.0234358, 0.0223645, -0.0725582, 0.0554191, -0.0788550, 0.0949227
2: -0.0537736, 0.0172716, -0.1467839, 0.0806166, -0.1343902, 0.1640555
3: 1.0027440, 1.0240462, 0.9804376, 1.0489149, -0.0461709, 0.0436085
4: -0.0107536, 0.0358998, -0.0488067, 0.1285111, -0.1392647, 0.0847065
5: -0.0093648, 0.0744055, -0.0746838, 0.1567380, -0.1661029, 0.1490893
6: -0.0534193, 0.0244086, -0.1332822, 0.1260363, -0.1794555, 0.1576907
7: -0.0459578, -0.0013402, -0.1112925, 0.0133000, -0.0592578, 0.1099043
8: -0.0239441, 0.0230900, -0.0643554, 0.1250360, -0.1489801, 0.0874454
9: -0.0247082, 0.0406120, -0.0818826, 0.0945841, -0.1192923, 0.1224946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 207

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
time: 2.31 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
time: 2.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.70 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0399517, upper bound: 0.0396105
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0399471, upper bound: 0.0396106
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0401580, upper bound: 0.0399998
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0402796, upper bound: 0.0400522
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0397327, upper bound: 0.0396240
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0399039, upper bound: 0.0396832
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0398801, upper bound: 0.0399943
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0400515, upper bound: 0.0400515
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0395020, upper bound: 0.0389794
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0394948, upper bound: 0.0389794
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0397377, upper bound: 0.0393476
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0398371, upper bound: 0.0393660
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0393666, upper bound: 0.0390069
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0394790, upper bound: 0.0390326
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0395855, upper bound: 0.0393438
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0391065, upper bound: 0.0395022
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0397458
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0389113, upper bound: 0.0390831
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.70
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0356621, 0.0307983, -0.0262107, 0.0271555, -0.0628176, 0.0570089
1: -0.0267519, 0.0245987, -0.0199685, 0.0206717, -0.0474236, 0.0445671
2: -0.0604595, 0.0183181, -0.0475118, 0.0171464, -0.0776060, 0.0658299
3: 1.0022124, 1.0262544, 1.0032272, 1.0226369, -0.0204245, 0.0230272
4: -0.0121335, 0.0403235, -0.0086116, 0.0321899, -0.0443234, 0.0489351
5: -0.0121901, 0.0802611, -0.0076890, 0.0689946, -0.0811846, 0.0879502
6: -0.0586168, 0.0289866, -0.0491081, 0.0185433, -0.0771601, 0.0780947
7: -0.0489651, -0.0020960, -0.0433068, -0.0014225, -0.0475119, 0.0412016
8: -0.0268403, 0.0267875, -0.0221991, 0.0183762, -0.0452165, 0.0489866
9: -0.0268460, 0.0439348, -0.0229177, 0.0370975, -0.0639435, 0.0668524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399517, upper bound: 0.0396105
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399517, upper bound: 0.0396105
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0398966, 0.0328716, -0.0250556, 0.0267188, -0.0666154, 0.0579272
1: -0.0298796, 0.0263470, -0.0191736, 0.0201430, -0.0500226, 0.0455206
2: -0.0660028, 0.0203138, -0.0458223, 0.0171211, -0.0831239, 0.0661361
3: 1.0015155, 1.0279118, 1.0033100, 1.0221281, -0.0206126, 0.0246018
4: -0.0134940, 0.0441161, -0.0081235, 0.0312072, -0.0447012, 0.0522396
5: -0.0147772, 0.0853848, -0.0071103, 0.0676822, -0.0824594, 0.0924951
6: -0.0628564, 0.0331799, -0.0478824, 0.0170989, -0.0799553, 0.0810622
7: -0.0518797, -0.0016801, -0.0425958, -0.0014263, -0.0504254, 0.0409157
8: -0.0294817, 0.0301624, -0.0216788, 0.0172244, -0.0467060, 0.0518412
9: -0.0286730, 0.0467009, -0.0224376, 0.0361825, -0.0648555, 0.0691385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399471, upper bound: 0.0396106
time: 3.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399471, upper bound: 0.0396106
time: 2.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0512675, 0.0401800, -0.0393187, 0.0325895, -0.0838570, 0.0794988
1: -0.0362606, 0.0298107, -0.0294764, 0.0260899, -0.0623505, 0.0592871
2: -0.0769992, 0.0286935, -0.0652480, 0.0200456, -0.0970448, 0.0939415
3: 0.9989788, 1.0309430, 1.0016311, 1.0276669, -0.0286881, 0.0293119
4: -0.0182797, 0.0552120, -0.0133235, 0.0435735, -0.0618532, 0.0685355
5: -0.0231991, 0.0957533, -0.0144129, 0.0846928, -0.1078919, 0.1101662
6: -0.0728762, 0.0451669, -0.0622762, 0.0326003, -0.1054765, 0.1074431
7: -0.0604233, -0.0019071, -0.0514872, -0.0017837, -0.0586287, 0.0495802
8: -0.0346328, 0.0415302, -0.0291191, 0.0297031, -0.0643359, 0.0706493
9: -0.0361069, 0.0522441, -0.0284203, 0.0463141, -0.0824210, 0.0806644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401580, upper bound: 0.0399998
time: 3.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0401580, upper bound: 0.0399998
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0553680, 0.0438941, -0.0425585, 0.0339838, -0.0893518, 0.0864526
1: -0.0385630, 0.0310579, -0.0313774, 0.0271428, -0.0657058, 0.0624354
2: -0.0808494, 0.0322231, -0.0685673, 0.0217988, -0.1026482, 0.1007904
3: 0.9977880, 1.0318027, 1.0009673, 1.0286173, -0.0308293, 0.0308354
4: -0.0203493, 0.0594503, -0.0142453, 0.0467070, -0.0670563, 0.0736956
5: -0.0263120, 0.0994807, -0.0164341, 0.0878075, -0.1141195, 0.1159148
6: -0.0764613, 0.0498778, -0.0648877, 0.0359855, -0.1124469, 0.1147655
7: -0.0635244, -0.0015009, -0.0538557, -0.0016454, -0.0618790, 0.0523548
8: -0.0363263, 0.0464990, -0.0306836, 0.0323485, -0.0686748, 0.0771827
9: -0.0390428, 0.0544935, -0.0300543, 0.0479915, -0.0870343, 0.0845478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402796, upper bound: 0.0400522
time: 3.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0402796, upper bound: 0.0400522
time: 1.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0940477, 0.0834077, -0.0258904, 0.0270197, -0.1210674, 0.1092980
1: -0.0602217, 0.0467756, -0.0197556, 0.0205031, -0.0807248, 0.0665312
2: -0.1231740, 0.0628734, -0.0470174, 0.0171331, -0.1403071, 0.1098908
3: 0.9867822, 1.0428518, 1.0032858, 1.0224743, -0.0356921, 0.0395660
4: -0.0383727, 0.1036446, -0.0084912, 0.0318903, -0.0702630, 0.1121359
5: -0.0572027, 0.1360724, -0.0075144, 0.0685981, -0.1258008, 0.1435867
6: -0.1128270, 0.0985318, -0.0487342, 0.0181729, -0.1310000, 0.1472659
7: -0.0940092, 0.0070301, -0.0431002, -0.0016087, -0.0923361, 0.0501303
8: -0.0542983, 0.0965770, -0.0220238, 0.0180645, -0.0723628, 0.1186009
9: -0.0662906, 0.0802313, -0.0227722, 0.0368250, -0.1031156, 0.1030035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397327, upper bound: 0.0396240
time: 2.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397327, upper bound: 0.0396240
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0981705, 0.0874231, -0.0280142, 0.0278467, -0.1260172, 0.1154373
1: -0.0625420, 0.0481569, -0.0212040, 0.0215146, -0.0840566, 0.0693609
2: -0.1272501, 0.0663354, -0.0501705, 0.0171811, -0.1444312, 0.1165059
3: 0.9855738, 1.0437498, 1.0031137, 1.0234460, -0.0378722, 0.0406361
4: -0.0404107, 0.1080975, -0.0093636, 0.0337385, -0.0741492, 0.1174612
5: -0.0603188, 0.1398840, -0.0086003, 0.0710569, -0.1313757, 0.1484844
6: -0.1164647, 0.1035022, -0.0510392, 0.0207883, -0.1372530, 0.1545414
7: -0.0971975, 0.0082115, -0.0444189, -0.0015247, -0.0956177, 0.0526303
8: -0.0560152, 0.1018091, -0.0230111, 0.0201682, -0.0761834, 0.1248202
9: -0.0692248, 0.0827210, -0.0236722, 0.0385346, -0.1077593, 0.1063932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399039, upper bound: 0.0396832
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399039, upper bound: 0.0396832
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1063897, 0.0958242, -0.0387070, 0.0322901, -0.1386798, 0.1345313
1: -0.0671441, 0.0514755, -0.0290207, 0.0258417, -0.0929859, 0.0804962
2: -0.1361907, 0.0728996, -0.0644504, 0.0197562, -0.1559469, 0.1373500
3: 0.9832167, 1.0460705, 1.0017363, 1.0274359, -0.0442192, 0.0443342
4: -0.0442696, 0.1174557, -0.0131242, 0.0430308, -0.0873004, 0.1305799
5: -0.0668942, 0.1476282, -0.0140459, 0.0839497, -0.1508439, 0.1616741
6: -0.1241606, 0.1138213, -0.0616679, 0.0319961, -0.1561567, 0.1754892
7: -0.1036599, 0.0104045, -0.0510663, -0.0018934, -0.1017444, 0.0614709
8: -0.0598169, 0.1124613, -0.0287401, 0.0292139, -0.0890309, 0.1412014
9: -0.0750334, 0.0881590, -0.0281582, 0.0459140, -0.1209474, 0.1163172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398801, upper bound: 0.0399943
time: 2.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398801, upper bound: 0.0399943
time: 2.26 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1104476, 0.0997926, -0.0417263, 0.0335349, -0.1439824, 0.1415188
1: -0.0694289, 0.0528380, -0.0309123, 0.0268825, -0.0963113, 0.0837503
2: -0.1402137, 0.0763082, -0.0677525, 0.0212578, -0.1614715, 0.1440607
3: 0.9820240, 1.0469406, 1.0011519, 1.0283810, -0.0463570, 0.0457886
4: -0.0462756, 0.1218497, -0.0139517, 0.0458908, -0.0921664, 0.1358014
5: -0.0699693, 0.1513969, -0.0158537, 0.0870488, -0.1570182, 0.1672506
6: -0.1277540, 0.1187123, -0.0641915, 0.0351060, -0.1628600, 0.1829038
7: -0.1068071, 0.0115572, -0.0532349, -0.0017541, -0.1050516, 0.0647922
8: -0.0615029, 0.1176212, -0.0302969, 0.0315858, -0.0930887, 0.1479180
9: -0.0779267, 0.0906255, -0.0295611, 0.0475823, -0.1255090, 0.1201867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400515, upper bound: 0.0400515
time: 2.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400515, upper bound: 0.0400515
time: 2.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0356621, 0.0307983, -0.0183925, 0.0239597, -0.0596218, 0.0491908
1: -0.0267519, 0.0245987, -0.0147505, 0.0166621, -0.0434141, 0.0393491
2: -0.0604595, 0.0183181, -0.0354612, 0.0170118, -0.0774714, 0.0537792
3: 1.0022124, 1.0262544, 1.0036948, 1.0188454, -0.0166330, 0.0225596
4: -0.0121335, 0.0403235, -0.0055724, 0.0251437, -0.0372772, 0.0458959
5: -0.0121901, 0.0802611, -0.0037242, 0.0596742, -0.0718642, 0.0839853
6: -0.0586168, 0.0289866, -0.0403166, 0.0090857, -0.0677025, 0.0693032
7: -0.0489651, -0.0020960, -0.0382623, -0.0008231, -0.0481144, 0.0361663
8: -0.0268403, 0.0267875, -0.0185197, 0.0109056, -0.0377459, 0.0453073
9: -0.0268460, 0.0439348, -0.0195371, 0.0306222, -0.0574682, 0.0634719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 156

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393522, upper bound: 0.0388168
time: 2.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0389144
time: 2.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0398966, 0.0328716, -0.0173171, 0.0235508, -0.0634474, 0.0501887
1: -0.0298796, 0.0263470, -0.0140248, 0.0161794, -0.0460589, 0.0403718
2: -0.0660028, 0.0203138, -0.0339076, 0.0169875, -0.0829903, 0.0542214
3: 1.0015155, 1.0279118, 1.0037733, 1.0184531, -0.0169376, 0.0241385
4: -0.0134940, 0.0441161, -0.0051159, 0.0242665, -0.0377606, 0.0492320
5: -0.0147772, 0.0853848, -0.0033836, 0.0584269, -0.0732041, 0.0887683
6: -0.0628564, 0.0331799, -0.0392754, 0.0077583, -0.0706147, 0.0724553
7: -0.0518797, -0.0016801, -0.0375998, -0.0008267, -0.0510281, 0.0359197
8: -0.0294817, 0.0301624, -0.0181313, 0.0098967, -0.0393784, 0.0482938
9: -0.0286730, 0.0467009, -0.0191259, 0.0297604, -0.0584333, 0.0658268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 107

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393425, upper bound: 0.0388168
time: 2.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394310, upper bound: 0.0389144
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0512675, 0.0401800, -0.0289453, 0.0280316, -0.0792992, 0.0691254
1: -0.0362606, 0.0298107, -0.0219467, 0.0215074, -0.0577680, 0.0517575
2: -0.0769992, 0.0286935, -0.0509304, 0.0172253, -0.0942245, 0.0796240
3: 0.9989788, 1.0309430, 1.0029684, 1.0232321, -0.0242533, 0.0279746
4: -0.0182797, 0.0552120, -0.0099342, 0.0341668, -0.0524465, 0.0651462
5: -0.0231991, 0.0957533, -0.0084457, 0.0720150, -0.0952141, 0.1041990
6: -0.0728762, 0.0451669, -0.0513719, 0.0220090, -0.0948852, 0.0965388
7: -0.0604233, -0.0019071, -0.0448147, -0.0013354, -0.0590390, 0.0429077
8: -0.0346328, 0.0415302, -0.0229553, 0.0211693, -0.0558021, 0.0644855
9: -0.0361069, 0.0522441, -0.0238543, 0.0391329, -0.0752398, 0.0760984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397377, upper bound: 0.0393476
time: 2.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397377, upper bound: 0.0393476
time: 2.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0553680, 0.0438941, -0.0316357, 0.0289277, -0.0842957, 0.0755298
1: -0.0385630, 0.0310579, -0.0238917, 0.0225981, -0.0611612, 0.0549496
2: -0.0808494, 0.0322231, -0.0545678, 0.0172817, -0.0981311, 0.0867909
3: 0.9977880, 1.0318027, 1.0026494, 1.0242743, -0.0264863, 0.0291532
4: -0.0203493, 0.0594503, -0.0109537, 0.0364245, -0.0567738, 0.0704040
5: -0.0263120, 0.0994807, -0.0096197, 0.0751469, -0.1014588, 0.1091004
6: -0.0764613, 0.0498778, -0.0540035, 0.0250129, -0.1014742, 0.1038813
7: -0.0635244, -0.0015009, -0.0462591, -0.0012430, -0.0622421, 0.0447582
8: -0.0363263, 0.0464990, -0.0242733, 0.0235781, -0.0599044, 0.0707723
9: -0.0390428, 0.0544935, -0.0249689, 0.0409997, -0.0800425, 0.0794624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398371, upper bound: 0.0393660
time: 2.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398371, upper bound: 0.0393660
time: 2.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0940477, 0.0834077, -0.0179789, 0.0238478, -0.1178956, 0.1013866
1: -0.0602217, 0.0467756, -0.0144489, 0.0165294, -0.0767510, 0.0612246
2: -0.1231740, 0.0628734, -0.0349618, 0.0169927, -0.1401667, 0.0978352
3: 0.9867822, 1.0428518, 1.0037543, 1.0187299, -0.0319477, 0.0390975
4: -0.0383727, 0.1036446, -0.0053322, 0.0249047, -0.0632774, 0.1089768
5: -0.0572027, 0.1360724, -0.0036129, 0.0593339, -0.1165365, 0.1396853
6: -0.1128270, 0.0985318, -0.0400352, 0.0084516, -0.1212786, 0.1385670
7: -0.0940092, 0.0070301, -0.0380665, -0.0010053, -0.0929415, 0.0450965
8: -0.0542983, 0.0965770, -0.0183787, 0.0104304, -0.0647287, 0.1149558
9: -0.0662906, 0.0802313, -0.0194084, 0.0303634, -0.0966539, 0.0996397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 156

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392277, upper bound: 0.0388884
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392282, upper bound: 0.0388835
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0981705, 0.0874231, -0.0203675, 0.0247481, -0.1229186, 0.1077906
1: -0.0625420, 0.0481569, -0.0160615, 0.0176227, -0.0801647, 0.0642184
2: -0.1272501, 0.0663354, -0.0384187, 0.0170492, -0.1442993, 0.1047541
3: 0.9855738, 1.0437498, 1.0035615, 1.0196501, -0.0340763, 0.0401883
4: -0.0404107, 0.1080975, -0.0063695, 0.0268282, -0.0672388, 0.1144670
5: -0.0603188, 0.1398840, -0.0044350, 0.0620220, -0.1223407, 0.1443190
6: -0.1164647, 0.1035022, -0.0423404, 0.0114958, -0.1279605, 0.1458427
7: -0.0971975, 0.0082115, -0.0395188, -0.0009217, -0.0962239, 0.0477303
8: -0.0560152, 0.1018091, -0.0192973, 0.0127833, -0.0687985, 0.1211064
9: -0.0692248, 0.0827210, -0.0203190, 0.0322437, -0.1014685, 0.1030400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 229

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393337, upper bound: 0.0389144
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393347, upper bound: 0.0389137
time: 2.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1063897, 0.0958242, -0.0284745, 0.0278481, -0.1342378, 0.1242987
1: -0.0671441, 0.0514755, -0.0216257, 0.0212799, -0.0884241, 0.0731012
2: -0.1361907, 0.0728996, -0.0502256, 0.0172155, -0.1534062, 0.1231252
3: 0.9832167, 1.0460705, 1.0030092, 1.0230105, -0.0397938, 0.0430613
4: -0.0442696, 0.1174557, -0.0097405, 0.0337567, -0.0780264, 0.1271963
5: -0.0668942, 0.1476282, -0.0081980, 0.0714750, -0.1383692, 0.1558262
6: -0.1241606, 0.1138213, -0.0508575, 0.0214233, -0.1455838, 0.1646788
7: -0.1036599, 0.0104045, -0.0445212, -0.0014287, -0.1021743, 0.0549257
8: -0.0598169, 0.1124613, -0.0227324, 0.0206989, -0.0805158, 0.1351937
9: -0.0750334, 0.0881590, -0.0236525, 0.0387552, -0.1137886, 0.1118115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394184, upper bound: 0.0392057
time: 2.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394153, upper bound: 0.0391696
time: 2.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1104476, 0.0997926, -0.0310245, 0.0287394, -0.1391870, 0.1308171
1: -0.0694289, 0.0528380, -0.0234358, 0.0223645, -0.0917934, 0.0762738
2: -0.1402137, 0.0763082, -0.0537736, 0.0172716, -0.1574852, 0.1300817
3: 0.9820240, 1.0469406, 1.0027440, 1.0240462, -0.0420222, 0.0441966
4: -0.0462756, 0.1218497, -0.0107536, 0.0358998, -0.0821754, 0.1326033
5: -0.0699693, 0.1513969, -0.0093648, 0.0744055, -0.1443748, 0.1607617
6: -0.1277540, 0.1187123, -0.0534193, 0.0244086, -0.1521625, 0.1721315
7: -0.1068071, 0.0115572, -0.0459578, -0.0013402, -0.1054200, 0.0575150
8: -0.0615029, 0.1176212, -0.0239441, 0.0230900, -0.0845929, 0.1415652
9: -0.0779267, 0.0906255, -0.0247082, 0.0406120, -0.1185387, 0.1153338

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=8, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 207

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
time: 2.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
time: 2.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0205359, 0.0247998, -0.0473184, 0.0375590, -0.0580949, 0.0721182
1: -0.0161813, 0.0176860, -0.0340481, 0.0285257, -0.0447070, 0.0517341
2: -0.0386367, 0.0170577, -0.0730296, 0.0257835, -0.0644202, 0.0900873
3: 1.0035532, 1.0197047, 0.9999244, 1.0297885, -0.0262353, 0.0197803
4: -0.0064616, 0.0269371, -0.0165945, 0.0513115, -0.0577731, 0.0435316
5: -0.0044843, 0.0621732, -0.0201543, 0.0921151, -0.0965994, 0.0823276
6: -0.0424718, 0.0117458, -0.0692368, 0.0409976, -0.0834694, 0.0809826
7: -0.0396054, -0.0011025, -0.0574200, -0.0018060, -0.0377994, 0.0562711
8: -0.0193642, 0.0129632, -0.0327722, 0.0375090, -0.0568731, 0.0457354
9: -0.0203737, 0.0323550, -0.0334622, 0.0502693, -0.0706430, 0.0658172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0395020
time: 3.20 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0394948
time: 2.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0534545, 0.0376497, -0.0473184, 0.0375590, -0.0910135, 0.0849681
1: -0.0386228, 0.0334600, -0.0340481, 0.0285257, -0.0671485, 0.0675081
2: -0.0877613, 0.0177251, -0.0730296, 0.0257835, -0.1135448, 0.0907547
3: 1.0012128, 1.0348840, 0.9999244, 1.0297885, -0.0285757, 0.0349596
4: -0.0199383, 0.0556605, -0.0165945, 0.0513115, -0.0712498, 0.0722550
5: -0.0214940, 0.1002712, -0.0201543, 0.0921151, -0.1136091, 0.1204255
6: -0.0783689, 0.0523942, -0.0692368, 0.0409976, -0.1193664, 0.1216309
7: -0.0601508, -0.0011302, -0.0574200, -0.0018060, -0.0583448, 0.0562404
8: -0.0346088, 0.0457349, -0.0327722, 0.0375090, -0.0721177, 0.0785071
9: -0.0343839, 0.0588547, -0.0334622, 0.0502693, -0.0846532, 0.0923169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 134

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0395019
time: 2.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0394948
time: 2.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0205359, 0.0247998, -0.0356858, 0.0306686, -0.0512045, 0.0604855
1: -0.0161813, 0.0176860, -0.0269274, 0.0241701, -0.0403514, 0.0446134
2: -0.0386367, 0.0170577, -0.0597549, 0.0186891, -0.0573258, 0.0768126
3: 1.0035532, 1.0197047, 1.0020181, 1.0257523, -0.0221992, 0.0176866
4: -0.0064616, 0.0269371, -0.0122964, 0.0399373, -0.0463989, 0.0392335
5: -0.0044843, 0.0621732, -0.0117912, 0.0800145, -0.0844988, 0.0739644
6: -0.0424718, 0.0117458, -0.0579057, 0.0290308, -0.0715027, 0.0696515
7: -0.0396054, -0.0011025, -0.0487722, -0.0014500, -0.0381527, 0.0476276
8: -0.0193642, 0.0129632, -0.0266855, 0.0267920, -0.0461562, 0.0396487
9: -0.0203737, 0.0323550, -0.0266701, 0.0435678, -0.0639415, 0.0590250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393570
time: 2.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393458
time: 3.28 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0534545, 0.0376497, -0.0356858, 0.0306686, -0.0841231, 0.0733354
1: -0.0386228, 0.0334600, -0.0269274, 0.0241701, -0.0627929, 0.0603874
2: -0.0877613, 0.0177251, -0.0597549, 0.0186891, -0.1064504, 0.0774800
3: 1.0012128, 1.0348840, 1.0020181, 1.0257523, -0.0245395, 0.0328659
4: -0.0199383, 0.0556605, -0.0122964, 0.0399373, -0.0598757, 0.0679569
5: -0.0214940, 0.1002712, -0.0117912, 0.0800145, -0.1015085, 0.1120623
6: -0.0783689, 0.0523942, -0.0579057, 0.0290308, -0.1073997, 0.1102999
7: -0.0601508, -0.0011302, -0.0487722, -0.0014500, -0.0586910, 0.0475969
8: -0.0346088, 0.0457349, -0.0266855, 0.0267920, -0.0614008, 0.0724204
9: -0.0343839, 0.0588547, -0.0266701, 0.0435678, -0.0779517, 0.0855248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 175

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393570
time: 2.22 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393458
time: 2.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0312271, 0.0287823, -0.0595775, 0.0483798, -0.0796069, 0.0883598
1: -0.0235987, 0.0224179, -0.0409217, 0.0327340, -0.0563328, 0.0633395
2: -0.0539885, 0.0172792, -0.0853977, 0.0357665, -0.0897549, 0.1026769
3: 1.0027127, 1.0240983, 0.9965381, 1.0329731, -0.0302603, 0.0275602
4: -0.0108316, 0.0360449, -0.0224595, 0.0642284, -0.0750600, 0.0585044
5: -0.0094237, 0.0746200, -0.0297583, 0.1034627, -0.1128864, 0.1043783
6: -0.0535643, 0.0246167, -0.0805103, 0.0551392, -0.1087035, 0.1051270
7: -0.0460299, -0.0015055, -0.0668376, -0.0009325, -0.0450974, 0.0652909
8: -0.0240532, 0.0232495, -0.0382568, 0.0520879, -0.0761410, 0.0615062
9: -0.0247818, 0.0407032, -0.0421354, 0.0572616, -0.0820434, 0.0828387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393476, upper bound: 0.0397377
time: 2.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393660, upper bound: 0.0398371
time: 3.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0725280, 0.0416143, -0.0595775, 0.0483798, -0.1209079, 0.1011918
1: -0.0543594, 0.0382207, -0.0409217, 0.0327340, -0.0870934, 0.0791424
2: -0.1077576, 0.0179483, -0.0853977, 0.0357665, -0.1435241, 0.1033460
3: 0.9963483, 1.0393746, 0.9965381, 1.0329731, -0.0366247, 0.0428365
4: -0.0242545, 0.0716848, -0.0224595, 0.0642284, -0.0884829, 0.0941443
5: -0.0265977, 0.1251206, -0.0297583, 0.1034627, -0.1300605, 0.1548789
6: -0.0932490, 0.0652061, -0.0805103, 0.0551392, -0.1483882, 0.1457163
7: -0.0665347, 0.0047171, -0.0668376, -0.0009325, -0.0656021, 0.0715547
8: -0.0461102, 0.0561807, -0.0382568, 0.0520879, -0.0981980, 0.0944375
9: -0.0424301, 0.0671374, -0.0421354, 0.0572616, -0.0996917, 0.1092729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393476, upper bound: 0.0397377
time: 1.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393660, upper bound: 0.0398371
time: 4.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0312271, 0.0287823, -0.0465183, 0.0373918, -0.0686189, 0.0753006
1: -0.0235987, 0.0224179, -0.0335749, 0.0276616, -0.0512603, 0.0559927
2: -0.0539885, 0.0172792, -0.0710695, 0.0261156, -0.0801040, 0.0883487
3: 1.0027127, 1.0240983, 0.9998189, 1.0287998, -0.0260870, 0.0242794
4: -0.0108316, 0.0360449, -0.0167343, 0.0501887, -0.0610203, 0.0527792
5: -0.0094237, 0.0746200, -0.0193297, 0.0908468, -0.1002705, 0.0939497
6: -0.0535643, 0.0246167, -0.0677604, 0.0401952, -0.0937595, 0.0923771
7: -0.0460299, -0.0015055, -0.0566827, -0.0012160, -0.0448139, 0.0551406
8: -0.0240532, 0.0232495, -0.0319545, 0.0371330, -0.0611861, 0.0552039
9: -0.0247818, 0.0407032, -0.0331519, 0.0493323, -0.0741141, 0.0738551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0394905
time: 2.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0397458
time: 2.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0725280, 0.0416143, -0.0465183, 0.0373918, -0.1099198, 0.0881326
1: -0.0543594, 0.0382207, -0.0335749, 0.0276616, -0.0820210, 0.0717956
2: -0.1077576, 0.0179483, -0.0710695, 0.0261156, -0.1338732, 0.0890179
3: 0.9963483, 1.0393746, 0.9998189, 1.0287998, -0.0324515, 0.0395557
4: -0.0242545, 0.0716848, -0.0167343, 0.0501887, -0.0744432, 0.0884191
5: -0.0265977, 0.1251206, -0.0193297, 0.0908468, -0.1174445, 0.1444503
6: -0.0932490, 0.0652061, -0.0677604, 0.0401952, -0.1334442, 0.1329665
7: -0.0665347, 0.0047171, -0.0566827, -0.0012160, -0.0653187, 0.0613998
8: -0.0461102, 0.0561807, -0.0319545, 0.0371330, -0.0832431, 0.0881352
9: -0.0424301, 0.0671374, -0.0331519, 0.0493323, -0.0917624, 0.1002893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0394905
time: 2.20 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0397458
time: 2.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0121865, 0.0205237, -0.0956114, 0.0852114, -0.0973980, 0.1161350
1: -0.0111123, 0.0124408, -0.0610830, 0.0478253, -0.0589377, 0.0735238
2: -0.0241002, 0.0168150, -0.1255418, 0.0637645, -0.0878647, 0.1423569
3: 1.0043809, 1.0153571, 0.9863974, 1.0437478, -0.0393668, 0.0289598
4: -0.0042961, 0.0178014, -0.0389031, 0.1057843, -0.1100803, 0.0567045
5: -0.0016726, 0.0494892, -0.0586600, 0.1377079, -0.1393805, 0.1081492
6: -0.0314768, 0.0039164, -0.1146517, 0.1007943, -0.1322711, 0.1185681
7: -0.0331831, -0.0010622, -0.0953456, 0.0073792, -0.0405624, 0.0942204
8: -0.0149946, 0.0068247, -0.0553064, 0.0987257, -0.1137203, 0.0621311
9: -0.0164003, 0.0235109, -0.0673409, 0.0816421, -0.0980424, 0.0908518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
time: 2.45 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
time: 3.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0143201, 0.0220417, -0.0933724, 0.0829113, -0.0972313, 0.1154141
1: -0.0121890, 0.0143456, -0.0598320, 0.0469447, -0.0591337, 0.0741775
2: -0.0287677, 0.0168884, -0.1231435, 0.0619351, -0.0907028, 0.1400319
3: 1.0040791, 1.0169575, 0.9870480, 1.0431416, -0.0390625, 0.0299095
4: -0.0043068, 0.0210396, -0.0378312, 0.1032441, -0.1075510, 0.0588708
5: -0.0023487, 0.0539216, -0.0568663, 0.1356083, -0.1379571, 0.1107880
6: -0.0354061, 0.0049157, -0.1125616, 0.0979996, -0.1334058, 0.1174773
7: -0.0353275, -0.0007037, -0.0935821, 0.0067657, -0.0420932, 0.0928191
8: -0.0165780, 0.0077708, -0.0542875, 0.0958094, -0.1123874, 0.0620583
9: -0.0177263, 0.0266035, -0.0657372, 0.0801824, -0.0979087, 0.0923408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
time: 2.18 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
time: 11.20 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0134517, 0.0214168, -0.0996872, 0.0891886, -0.1026403, 0.1211040
1: -0.0117587, 0.0135254, -0.0633713, 0.0491912, -0.0609499, 0.0768967
2: -0.0268147, 0.0168706, -0.1295504, 0.0672139, -0.0940286, 0.1464210
3: 1.0041903, 1.0162247, 0.9851959, 1.0446527, -0.0404624, 0.0310288
4: -0.0043048, 0.0197152, -0.0409268, 0.1101795, -0.1144842, 0.0606420
5: -0.0019877, 0.0521660, -0.0617637, 0.1414413, -0.1434290, 0.1139298
6: -0.0337665, 0.0044656, -0.1182348, 0.1057122, -0.1394787, 0.1227003
7: -0.0344475, -0.0009788, -0.0984844, 0.0085486, -0.0429961, 0.0974530
8: -0.0159007, 0.0073213, -0.0570217, 0.1038916, -0.1197922, 0.0643430
9: -0.0171721, 0.0253812, -0.0702377, 0.0840900, -0.1012621, 0.0956189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 112

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
time: 2.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
time: 2.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0156270, 0.0229312, -0.0974974, 0.0869774, -0.1026044, 0.1204287
1: -0.0128742, 0.0154324, -0.0621466, 0.0483558, -0.0612300, 0.0775790
2: -0.0315041, 0.0169430, -0.1272560, 0.0654154, -0.0969194, 0.1441991
3: 1.0038898, 1.0178411, 0.9858336, 1.0440760, -0.0401862, 0.0320075
4: -0.0043605, 0.0229400, -0.0398745, 0.1077275, -0.1120880, 0.0628145
5: -0.0028968, 0.0565681, -0.0600259, 0.1394215, -0.1423183, 0.1165940
6: -0.0376878, 0.0055793, -0.1162339, 0.1029829, -0.1406706, 0.1218132
7: -0.0365916, -0.0006188, -0.0967773, 0.0079449, -0.0445365, 0.0961068
8: -0.0174932, 0.0083985, -0.0560312, 0.1010663, -0.1185595, 0.0644298
9: -0.0184999, 0.0284567, -0.0686864, 0.0826992, -0.1011990, 0.0971431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389112, upper bound: 0.0390831
time: 2.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0389112, upper bound: 0.0390831
time: 2.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0203980, 0.0246911, -0.1075425, 0.0972581, -0.1176561, 0.1322336
1: -0.0161409, 0.0173656, -0.0677672, 0.0523658, -0.0685067, 0.0851328
2: -0.0381279, 0.0170492, -0.1380780, 0.0735335, -0.1116614, 0.1551273
3: 1.0035952, 1.0192623, 0.9829326, 1.0468626, -0.0432674, 0.0363297
4: -0.0064184, 0.0267415, -0.0446413, 0.1191261, -0.1255445, 0.0713828
5: -0.0041185, 0.0621888, -0.0680776, 0.1488127, -0.1529311, 0.1302663
6: -0.0420887, 0.0113773, -0.1255923, 0.1156009, -0.1576896, 0.1369696
7: -0.0394776, -0.0014890, -0.1046605, 0.0106514, -0.0501290, 0.1031145
8: -0.0189823, 0.0127371, -0.0606599, 0.1141150, -0.1330972, 0.0733970
9: -0.0202348, 0.0322708, -0.0758117, 0.0892660, -0.1095008, 0.1080825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 112

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
time: 2.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
time: 2.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0233905, 0.0258951, -0.1052199, 0.0948660, -0.1182565, 0.1311150
1: -0.0181360, 0.0189495, -0.0664688, 0.0514254, -0.0695613, 0.0854183
2: -0.0427787, 0.0171068, -0.1355453, 0.0716678, -0.1144466, 0.1526520
3: 1.0033559, 1.0208445, 0.9836038, 1.0462104, -0.0428545, 0.0372406
4: -0.0076240, 0.0293885, -0.0435467, 0.1164721, -0.1240960, 0.0729352
5: -0.0057408, 0.0655814, -0.0662094, 0.1466204, -0.1523612, 0.1317908
6: -0.0454427, 0.0151637, -0.1234052, 0.1126876, -0.1581304, 0.1385689
7: -0.0413779, -0.0011744, -0.1028267, 0.0100237, -0.0514016, 0.1015996
8: -0.0204936, 0.0157064, -0.0595844, 0.1110906, -0.1315841, 0.0752909
9: -0.0215343, 0.0346871, -0.0741590, 0.0877248, -0.1092591, 0.1088461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
time: 2.30 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
time: 2.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0301566, 0.0284800, -0.1160757, 0.1057376, -0.1358943, 0.1445557
1: -0.0227842, 0.0220479, -0.0725582, 0.0554191, -0.0782033, 0.0946061
2: -0.0526679, 0.0172565, -0.1467839, 0.0806166, -0.1332845, 0.1640404
3: 1.0028855, 1.0237423, 0.9804376, 1.0489149, -0.0460294, 0.0433047
4: -0.0104643, 0.0351683, -0.0488067, 0.1285111, -0.1389754, 0.0839750
5: -0.0090224, 0.0733596, -0.0746838, 0.1567380, -0.1657605, 0.1480435
6: -0.0526137, 0.0235492, -0.1332822, 0.1260363, -0.1786500, 0.1568314
7: -0.0455391, -0.0015419, -0.1112925, 0.0133000, -0.0588391, 0.1097057
8: -0.0234877, 0.0223963, -0.0643554, 0.1250360, -0.1485237, 0.0867516
9: -0.0243440, 0.0400705, -0.0818826, 0.0945841, -0.1189281, 0.1219530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 118

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
time: 2.24 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
time: 2.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0714762, 0.0413009, -0.1160757, 0.1057376, -0.1772138, 0.1573766
1: -0.0535659, 0.0378393, -0.0725582, 0.0554191, -0.1089851, 0.1103975
2: -0.1064235, 0.0179265, -0.1467839, 0.0806166, -0.1870401, 0.1647104
3: 0.9965479, 1.0390097, 0.9804376, 1.0489149, -0.0523670, 0.0585721
4: -0.0239041, 0.0707982, -0.0488067, 0.1285111, -0.1524152, 0.1196048
5: -0.0261869, 0.1238516, -0.0746838, 0.1567380, -0.1829249, 0.1985354
6: -0.0922762, 0.0641711, -0.1332822, 0.1260363, -0.2183125, 0.1974532
7: -0.0660324, 0.0043139, -0.1112925, 0.0133000, -0.0793324, 0.1156064
8: -0.0455584, 0.0553422, -0.0643554, 0.1250360, -0.1705944, 0.1196975
9: -0.0419880, 0.0664949, -0.0818826, 0.0945841, -0.1365720, 0.1483774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
time: 2.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
time: 2.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.14 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0399517, upper bound: 0.0396105
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0399517, upper bound: 0.0396105
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0399471, upper bound: 0.0396106
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0399471, upper bound: 0.0396106
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0401580, upper bound: 0.0399998
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0401580, upper bound: 0.0399998
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0402796, upper bound: 0.0400522
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0402796, upper bound: 0.0400522
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0397327, upper bound: 0.0396240
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0397327, upper bound: 0.0396240
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0399039, upper bound: 0.0396832
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0399039, upper bound: 0.0396832
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0398801, upper bound: 0.0399943
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0398801, upper bound: 0.0399943
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0400515, upper bound: 0.0400515
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0400515, upper bound: 0.0400515
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393522, upper bound: 0.0388168
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0394382, upper bound: 0.0389144
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393425, upper bound: 0.0388168
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0394310, upper bound: 0.0389144
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0397377, upper bound: 0.0393476
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0397377, upper bound: 0.0393476
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0398371, upper bound: 0.0393660
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0398371, upper bound: 0.0393660
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0392277, upper bound: 0.0388884
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0392282, upper bound: 0.0388835
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393337, upper bound: 0.0389144
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393347, upper bound: 0.0389137
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0394184, upper bound: 0.0392057
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0394153, upper bound: 0.0391696
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0395020
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0394948
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0395019
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0394948
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393570
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393458
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393570
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393458
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393476, upper bound: 0.0397377
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393660, upper bound: 0.0398371
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393476, upper bound: 0.0397377
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393660, upper bound: 0.0398371
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0394905
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0397458
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0394905
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0397458
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389112, upper bound: 0.0390831
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0389112, upper bound: 0.0390831
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.14
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0356621, 0.0307983, -0.0250272, 0.0267002, -0.0623623, 0.0558255
1: -0.0267519, 0.0245987, -0.0191591, 0.0201152, -0.0468671, 0.0437577
2: -0.0604595, 0.0183181, -0.0457600, 0.0171206, -0.0775802, 0.0640781
3: 1.0022124, 1.0262544, 1.0033333, 1.0221000, -0.0198876, 0.0229211
4: -0.0121335, 0.0403235, -0.0081187, 0.0311718, -0.0433053, 0.0484422
5: -0.0121901, 0.0802611, -0.0070874, 0.0676379, -0.0798280, 0.0873485
6: -0.0586168, 0.0289866, -0.0478379, 0.0170702, -0.0756870, 0.0768245
7: -0.0489651, -0.0020960, -0.0425749, -0.0017134, -0.0472196, 0.0404693
8: -0.0268403, 0.0267875, -0.0216547, 0.0171896, -0.0440300, 0.0484423
9: -0.0268460, 0.0439348, -0.0224177, 0.0361498, -0.0629958, 0.0663524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386078, upper bound: 0.0388961
time: 2.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397111, upper bound: 0.0393539
time: 2.49 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0356621, 0.0307983, -0.0577346, 0.0394543, -0.0751164, 0.0885329
1: -0.0267519, 0.0245987, -0.0414695, 0.0357946, -0.0625465, 0.0660681
2: -0.0604595, 0.0183181, -0.0945698, 0.0177855, -0.0782450, 0.1128879
3: 1.0022124, 1.0262544, 1.0010238, 1.0372254, -0.0350130, 0.0252306
4: -0.0121335, 0.0403235, -0.0215215, 0.0596957, -0.0718292, 0.0618450
5: -0.0121901, 0.0802611, -0.0240918, 0.1054191, -0.1176092, 0.1043529
6: -0.0586168, 0.0289866, -0.0835227, 0.0575276, -0.1161443, 0.1125093
7: -0.0489651, -0.0020960, -0.0629803, -0.0017463, -0.0471917, 0.0608709
8: -0.0268403, 0.0267875, -0.0368783, 0.0499386, -0.0767789, 0.0636659
9: -0.0268460, 0.0439348, -0.0363607, 0.0624569, -0.0893029, 0.0802954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386078, upper bound: 0.0388961
time: 2.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397111, upper bound: 0.0393539
time: 2.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0398966, 0.0328716, -0.0238675, 0.0262615, -0.0661581, 0.0567391
1: -0.0298796, 0.0263470, -0.0183609, 0.0195844, -0.0494640, 0.0447079
2: -0.0660028, 0.0203138, -0.0440633, 0.0170951, -0.0830979, 0.0643772
3: 1.0015155, 1.0279118, 1.0034167, 1.0215904, -0.0200748, 0.0244951
4: -0.0134940, 0.0441161, -0.0076301, 0.0301832, -0.0436773, 0.0517462
5: -0.0147772, 0.0853848, -0.0065072, 0.0663181, -0.0810953, 0.0918920
6: -0.0628564, 0.0331799, -0.0466062, 0.0156243, -0.0784807, 0.0797861
7: -0.0518797, -0.0016801, -0.0418605, -0.0017172, -0.0501331, 0.0401804
8: -0.0294817, 0.0301624, -0.0211332, 0.0160363, -0.0455179, 0.0512956
9: -0.0286730, 0.0467009, -0.0219358, 0.0352306, -0.0639036, 0.0686367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386113, upper bound: 0.0388983
time: 2.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397027, upper bound: 0.0393540
time: 2.23 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0398966, 0.0328716, -0.0565157, 0.0389831, -0.0788796, 0.0893873
1: -0.0298796, 0.0263470, -0.0406380, 0.0352039, -0.0650835, 0.0669851
2: -0.0660028, 0.0203138, -0.0927484, 0.0177588, -0.0837616, 0.1130622
3: 1.0015155, 1.0279118, 1.0011088, 1.0366430, -0.0351275, 0.0268030
4: -0.0134940, 0.0441161, -0.0210163, 0.0586342, -0.0721282, 0.0651324
5: -0.0147772, 0.0853848, -0.0234332, 0.1040324, -0.1188096, 0.1088179
6: -0.0628564, 0.0331799, -0.0821826, 0.0560043, -0.1188607, 0.1153624
7: -0.0518797, -0.0016801, -0.0622224, -0.0017501, -0.0501040, 0.0605403
8: -0.0294817, 0.0301624, -0.0362895, 0.0486989, -0.0781806, 0.0664520
9: -0.0286730, 0.0467009, -0.0358343, 0.0614845, -0.0901574, 0.0825352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0386113, upper bound: 0.0388983
time: 2.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397027, upper bound: 0.0393540
time: 2.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0512675, 0.0401800, -0.0377910, 0.0318453, -0.0831129, 0.0779710
1: -0.0362606, 0.0298107, -0.0283347, 0.0254821, -0.0617427, 0.0581454
2: -0.0769992, 0.0286935, -0.0632745, 0.0193157, -0.0963149, 0.0919681
3: 0.9989788, 1.0309430, 1.0019002, 1.0271072, -0.0281284, 0.0290428
4: -0.0182797, 0.0552120, -0.0128227, 0.0422268, -0.0605065, 0.0680347
5: -0.0231991, 0.0957533, -0.0135104, 0.0828392, -0.1060383, 0.1092637
6: -0.0728762, 0.0451669, -0.0607735, 0.0310942, -0.1039705, 0.1059403
7: -0.0604233, -0.0019071, -0.0504398, -0.0020992, -0.0583102, 0.0485328
8: -0.0346328, 0.0415302, -0.0281799, 0.0284831, -0.0631158, 0.0697101
9: -0.0361069, 0.0522441, -0.0277702, 0.0453215, -0.0814284, 0.0800143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 175

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0398207
time: 2.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0399996
time: 2.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0512675, 0.0401800, -0.0792004, 0.0520750, -0.1033426, 0.1193805
1: -0.0362606, 0.0298107, -0.0592202, 0.0420269, -0.0782875, 0.0890309
2: -0.0769992, 0.0286935, -0.1169795, 0.0388324, -0.1158316, 0.1456730
3: 0.9989788, 1.0309430, 0.9944446, 1.0422674, -0.0432886, 0.0364984
4: -0.0182797, 0.0552120, -0.0263230, 0.0789191, -0.0971988, 0.0815350
5: -0.0231991, 0.0957533, -0.0379314, 0.1334038, -0.1566029, 0.1336847
6: -0.0728762, 0.0451669, -0.1017381, 0.0718113, -0.1446875, 0.1469049
7: -0.0604233, -0.0019071, -0.0788997, 0.0065793, -0.0670025, 0.0769926
8: -0.0346328, 0.0415302, -0.0533904, 0.0615884, -0.0962211, 0.0949207
9: -0.0361069, 0.0522441, -0.0454294, 0.0724868, -0.1085937, 0.0976735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 175

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0398207
time: 2.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0399996
time: 2.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0553680, 0.0438941, -0.0405003, 0.0330921, -0.0884601, 0.0843944
1: -0.0385630, 0.0310579, -0.0302261, 0.0265154, -0.0650785, 0.0612841
2: -0.0808494, 0.0322231, -0.0665776, 0.0206201, -0.1014695, 0.0988007
3: 0.9977880, 1.0318027, 1.0014248, 1.0280524, -0.0302644, 0.0303779
4: -0.0203493, 0.0594503, -0.0136480, 0.0446986, -0.0650479, 0.0730983
5: -0.0263120, 0.0994807, -0.0151219, 0.0859428, -0.1122548, 0.1146026
6: -0.0764613, 0.0498778, -0.0633018, 0.0338077, -0.1102690, 0.1131796
7: -0.0635244, -0.0015009, -0.0523336, -0.0019657, -0.0615587, 0.0508327
8: -0.0363263, 0.0464990, -0.0297361, 0.0306273, -0.0669536, 0.0762351
9: -0.0390428, 0.0544935, -0.0289670, 0.0469911, -0.0860339, 0.0834605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0399327
time: 2.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0400520
time: 6.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0553680, 0.0438941, -0.0956678, 0.0725853, -0.1279533, 0.1395619
1: -0.0385630, 0.0310579, -0.0611435, 0.0436170, -0.0821801, 0.0922014
2: -0.0808494, 0.0322231, -0.1203444, 0.0641779, -0.1450273, 0.1525676
3: 0.9977880, 1.0318027, 0.9872721, 1.0432123, -0.0454243, 0.0445306
4: -0.0203493, 0.0594503, -0.0391596, 0.0987305, -0.1190798, 0.0986099
5: -0.0263120, 0.0994807, -0.0584030, 0.1365726, -0.1628845, 0.1578837
6: -0.0764613, 0.0498778, -0.1143327, 0.0918781, -0.1683395, 0.1642105
7: -0.0635244, -0.0015009, -0.0939292, 0.0075380, -0.0710624, 0.0924283
8: -0.0363263, 0.0464990, -0.0549539, 0.0890818, -0.1254082, 0.1014529
9: -0.0390428, 0.0544935, -0.0674605, 0.0741920, -0.1132348, 0.1219540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0399327
time: 2.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0400520
time: 2.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0940477, 0.0834077, -0.0251845, 0.0267494, -0.1207971, 0.1085922
1: -0.0602217, 0.0467756, -0.0192715, 0.0201772, -0.0803989, 0.0660471
2: -0.1231740, 0.0628734, -0.0459809, 0.0171176, -0.1402916, 0.1088543
3: 0.9867822, 1.0428518, 1.0033507, 1.0221643, -0.0353822, 0.0395011
4: -0.0383727, 0.1036446, -0.0081960, 0.0312864, -0.0696591, 0.1118406
5: -0.0572027, 0.1360724, -0.0071659, 0.0677863, -0.1249889, 0.1432383
6: -0.1128270, 0.0985318, -0.0479844, 0.0172974, -0.1301244, 0.1465162
7: -0.0940092, 0.0070301, -0.0426658, -0.0018021, -0.0921427, 0.0496959
8: -0.0542983, 0.0965770, -0.0217069, 0.0173566, -0.0716549, 0.1182840
9: -0.0662906, 0.0802313, -0.0224762, 0.0362602, -0.1025508, 0.1027075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397215, upper bound: 0.0395833
time: 1.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396998, upper bound: 0.0395867
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0940477, 0.0834077, -0.0577853, 0.0394578, -0.1335055, 0.1411930
1: -0.0602217, 0.0467756, -0.0415120, 0.0357895, -0.0960112, 0.0882876
2: -0.1231740, 0.0628734, -0.0945962, 0.0177832, -0.1409572, 0.1574696
3: 0.9867822, 1.0428518, 1.0010442, 1.0372179, -0.0504357, 0.0418077
4: -0.0383727, 0.1036446, -0.0215620, 0.0597089, -0.0980816, 0.1252067
5: -0.0572027, 0.1360724, -0.0240930, 0.1054471, -0.1626498, 0.1601654
6: -0.1128270, 0.0985318, -0.0835354, 0.0576128, -0.1704398, 0.1820672
7: -0.0940092, 0.0070301, -0.0629907, -0.0018376, -0.0920995, 0.0700207
8: -0.0542983, 0.0965770, -0.0368848, 0.0500079, -0.1043062, 0.1334618
9: -0.0662906, 0.0802313, -0.0363714, 0.0624804, -0.1287710, 0.1166027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0397215, upper bound: 0.0395833
time: 2.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0396998, upper bound: 0.0395867
time: 2.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0981705, 0.0874231, -0.0273076, 0.0275765, -0.1257470, 0.1147307
1: -0.0625420, 0.0481569, -0.0207197, 0.0211886, -0.0837306, 0.0688766
2: -0.1272501, 0.0663354, -0.0491328, 0.0171653, -0.1444155, 0.1154682
3: 0.9855738, 1.0437498, 1.0031796, 1.0231351, -0.0375613, 0.0405703
4: -0.0404107, 0.1080975, -0.0090681, 0.0331341, -0.0735448, 0.1171656
5: -0.0603188, 0.1398840, -0.0082509, 0.0702450, -0.1305638, 0.1481349
6: -0.1164647, 0.1035022, -0.0502891, 0.0199115, -0.1363762, 0.1537913
7: -0.0971975, 0.0082115, -0.0439838, -0.0017210, -0.0954214, 0.0521952
8: -0.0560152, 0.1018091, -0.0226924, 0.0194601, -0.0754754, 0.1245016
9: -0.0692248, 0.0827210, -0.0233765, 0.0379689, -0.1071936, 0.1060974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398954, upper bound: 0.0396489
time: 2.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398741, upper bound: 0.0396524
time: 2.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0981705, 0.0874231, -0.0599408, 0.0403003, -0.1384708, 0.1473639
1: -0.0625420, 0.0481569, -0.0429826, 0.0368170, -0.0993590, 0.0911395
2: -0.1272501, 0.0663354, -0.0978096, 0.0178299, -0.1450801, 0.1641450
3: 0.9855738, 1.0437498, 1.0008794, 1.0381991, -0.0526253, 0.0428704
4: -0.0404107, 0.1080975, -0.0224415, 0.0615874, -0.1019980, 0.1305390
5: -0.0603188, 0.1398840, -0.0251939, 0.1079515, -0.1682703, 0.1650779
6: -0.1164647, 0.1035022, -0.0858758, 0.0602673, -0.1767320, 0.1893781
7: -0.0971975, 0.0082115, -0.0643386, -0.0017520, -0.0953848, 0.0725501
8: -0.0560152, 0.1018091, -0.0378762, 0.0521693, -0.1081845, 0.1396853
9: -0.0692248, 0.0827210, -0.0372865, 0.0642175, -0.1334423, 0.1200075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398954, upper bound: 0.0396489
time: 2.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398741, upper bound: 0.0396524
time: 2.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1063897, 0.0958242, -0.0377910, 0.0318453, -0.1382350, 0.1336152
1: -0.0671441, 0.0514755, -0.0283347, 0.0254821, -0.0926262, 0.0798102
2: -0.1361907, 0.0728996, -0.0632745, 0.0193157, -0.1555064, 0.1361741
3: 0.9832167, 1.0460705, 1.0019002, 1.0271072, -0.0438905, 0.0441703
4: -0.0442696, 0.1174557, -0.0128227, 0.0422268, -0.0864965, 0.1302784
5: -0.0668942, 0.1476282, -0.0135104, 0.0828392, -0.1497334, 0.1611387
6: -0.1241606, 0.1138213, -0.0607735, 0.0310942, -0.1552548, 0.1745948
7: -0.1036599, 0.0104045, -0.0504398, -0.0020992, -0.1015404, 0.0608444
8: -0.0598169, 0.1124613, -0.0281799, 0.0284831, -0.0883000, 0.1406412
9: -0.0750334, 0.0881590, -0.0277702, 0.0453215, -0.1203549, 0.1159291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=9, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398793, upper bound: 0.0399656
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398564, upper bound: 0.0399694
time: 3.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1063897, 0.0958242, -0.0792004, 0.0520750, -0.1584648, 0.1750247
1: -0.0671441, 0.0514755, -0.0592202, 0.0420269, -0.1091711, 0.1106957
2: -0.1361907, 0.0728996, -0.1169795, 0.0388324, -0.1750231, 0.1898791
3: 0.9832167, 1.0460705, 0.9944446, 1.0422674, -0.0590507, 0.0516258
4: -0.0442696, 0.1174557, -0.0263230, 0.0789191, -0.1231887, 0.1437787
5: -0.0668942, 0.1476282, -0.0379314, 0.1334038, -0.2002980, 0.1855596
6: -0.1241606, 0.1138213, -0.1017381, 0.0718113, -0.1959719, 0.2155594
7: -0.1036599, 0.0104045, -0.0788997, 0.0065793, -0.1102391, 0.0893043
8: -0.0598169, 0.1124613, -0.0533904, 0.0615884, -0.1214053, 0.1658518
9: -0.0750334, 0.0881590, -0.0454294, 0.0724868, -0.1475202, 0.1335884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398794, upper bound: 0.0399656
time: 2.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0398564, upper bound: 0.0399694
time: 2.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1104476, 0.0997926, -0.0405003, 0.0330921, -0.1435397, 0.1402929
1: -0.0694289, 0.0528380, -0.0302261, 0.0265154, -0.0959443, 0.0830641
2: -0.1402137, 0.0763082, -0.0665776, 0.0206201, -0.1608338, 0.1428858
3: 0.9820240, 1.0469406, 1.0014248, 1.0280524, -0.0460284, 0.0455158
4: -0.0462756, 0.1218497, -0.0136480, 0.0446986, -0.0909742, 0.1354977
5: -0.0699693, 0.1513969, -0.0151219, 0.0859428, -0.1559122, 0.1665188
6: -0.1277540, 0.1187123, -0.0633018, 0.0338077, -0.1615616, 0.1820141
7: -0.1068071, 0.0115572, -0.0523336, -0.0019657, -0.1048371, 0.0638908
8: -0.0615029, 0.1176212, -0.0297361, 0.0306273, -0.0921302, 0.1473572
9: -0.0779267, 0.0906255, -0.0289670, 0.0469911, -0.1249178, 0.1195925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 118

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400508, upper bound: 0.0400227
time: 3.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400260, upper bound: 0.0400260
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1104476, 0.0997926, -0.0956678, 0.0533411, -0.1637887, 0.1954604
1: -0.0694289, 0.0528380, -0.0611435, 0.0434944, -0.1129233, 0.1139814
2: -0.1402137, 0.0763082, -0.1203444, 0.0488115, -0.1890252, 0.1966526
3: 0.9820240, 1.0469406, 0.9880551, 1.0432123, -0.0611883, 0.0588855
4: -0.0462756, 0.1218497, -0.0271599, 0.0987305, -0.1450061, 0.1490096
5: -0.0699693, 0.1513969, -0.0482919, 0.1365726, -0.2065419, 0.1996888
6: -0.1277540, 0.1187123, -0.1042998, 0.0918781, -0.2196321, 0.2230121
7: -0.1068071, 0.0115572, -0.0930553, 0.0075380, -0.1143451, 0.1046125
8: -0.0615029, 0.1176212, -0.0549539, 0.0738458, -0.1353486, 0.1725751
9: -0.0779267, 0.0906255, -0.0558420, 0.0741920, -0.1521187, 0.1464676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 118

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400508, upper bound: 0.0400227
time: 2.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0400260, upper bound: 0.0400260
time: 2.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0323423, 0.0293621, -0.0135594, 0.0215129, -0.0538552, 0.0429215
1: -0.0242865, 0.0232687, -0.0117872, 0.0137546, -0.0380411, 0.0350559
2: -0.0561229, 0.0172628, -0.0272168, 0.0168462, -0.0729691, 0.0444796
3: 1.0027455, 1.0250092, 1.0042467, 1.0165145, -0.0137690, 0.0207624
4: -0.0110613, 0.0373845, -0.0042985, 0.0199106, -0.0309720, 0.0416830
5: -0.0104181, 0.0761803, -0.0020940, 0.0522521, -0.0626703, 0.0782743
6: -0.0553316, 0.0257319, -0.0340945, 0.0046407, -0.0599724, 0.0598263
7: -0.0468878, -0.0022359, -0.0345848, -0.0010719, -0.0457807, 0.0323129
8: -0.0248888, 0.0241516, -0.0160756, 0.0073887, -0.0322774, 0.0402273
9: -0.0254282, 0.0417491, -0.0172738, 0.0254502, -0.0508783, 0.0590228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380221, upper bound: 0.0381320
time: 2.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390614, upper bound: 0.0385066
time: 2.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0348341, 0.0303889, -0.0156453, 0.0229412, -0.0577754, 0.0460342
1: -0.0261389, 0.0242566, -0.0128902, 0.0154068, -0.0415458, 0.0371468
2: -0.0593694, 0.0179328, -0.0315002, 0.0169361, -0.0763055, 0.0494330
3: 1.0023539, 1.0259376, 1.0039310, 1.0177686, -0.0154147, 0.0220065
4: -0.0118677, 0.0395787, -0.0043523, 0.0229701, -0.0348379, 0.0439310
5: -0.0116959, 0.0792381, -0.0028287, 0.0566645, -0.0683604, 0.0820668
6: -0.0577808, 0.0281757, -0.0376988, 0.0055274, -0.0633082, 0.0658745
7: -0.0483926, -0.0021392, -0.0366135, -0.0009189, -0.0474407, 0.0344724
8: -0.0263276, 0.0261301, -0.0174173, 0.0083258, -0.0346534, 0.0435474
9: -0.0264900, 0.0433799, -0.0184969, 0.0284940, -0.0549841, 0.0618769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381229, upper bound: 0.0382281
time: 2.21 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391523, upper bound: 0.0386136
time: 2.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0365966, 0.0312554, -0.0129782, 0.0211037, -0.0577003, 0.0442336
1: -0.0274474, 0.0249970, -0.0114912, 0.0132558, -0.0407031, 0.0364882
2: -0.0616938, 0.0187717, -0.0259675, 0.0168219, -0.0785157, 0.0447392
3: 1.0020648, 1.0266639, 1.0043237, 1.0161139, -0.0140491, 0.0223402
4: -0.0124369, 0.0411509, -0.0042953, 0.0190331, -0.0314700, 0.0454462
5: -0.0128008, 0.0813589, -0.0019317, 0.0510301, -0.0638309, 0.0832906
6: -0.0595555, 0.0299285, -0.0330412, 0.0043883, -0.0639438, 0.0629697
7: -0.0495987, -0.0018553, -0.0340016, -0.0010757, -0.0484913, 0.0321404
8: -0.0274614, 0.0275391, -0.0156626, 0.0071667, -0.0346281, 0.0432017
9: -0.0272522, 0.0445225, -0.0169214, 0.0245959, -0.0518481, 0.0614439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0380208, upper bound: 0.0381330
time: 2.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0390508, upper bound: 0.0385066
time: 2.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0390647, 0.0324708, -0.0150195, 0.0225331, -0.0615978, 0.0474904
1: -0.0292793, 0.0260101, -0.0125517, 0.0149244, -0.0442037, 0.0385618
2: -0.0649295, 0.0199232, -0.0302400, 0.0169114, -0.0818410, 0.0501632
3: 1.0016618, 1.0275995, 1.0040096, 1.0173913, -0.0157295, 0.0235898
4: -0.0132332, 0.0433605, -0.0043093, 0.0220947, -0.0353280, 0.0476698
5: -0.0142762, 0.0843875, -0.0025826, 0.0554203, -0.0696965, 0.0869701
6: -0.0620364, 0.0323564, -0.0366609, 0.0051943, -0.0672306, 0.0690173
7: -0.0513006, -0.0017259, -0.0360249, -0.0009226, -0.0503480, 0.0342990
8: -0.0289793, 0.0295020, -0.0170282, 0.0080179, -0.0369972, 0.0465302
9: -0.0283100, 0.0461598, -0.0181411, 0.0276325, -0.0559425, 0.0643009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0381234, upper bound: 0.0382282
time: 2.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391421, upper bound: 0.0386136
time: 2.49 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0512675, 0.0401800, -0.0277889, 0.0275861, -0.0788537, 0.0679690
1: -0.0362606, 0.0298107, -0.0211555, 0.0209606, -0.0572212, 0.0509663
2: -0.0769992, 0.0286935, -0.0492166, 0.0172006, -0.0941998, 0.0779101
3: 0.9989788, 1.0309430, 1.0030740, 1.0227048, -0.0237260, 0.0278690
4: -0.0182797, 0.0552120, -0.0094530, 0.0331715, -0.0514512, 0.0646650
5: -0.0231991, 0.0957533, -0.0078538, 0.0706952, -0.0938943, 0.1036071
6: -0.0728762, 0.0451669, -0.0501274, 0.0205665, -0.0934427, 0.0952943
7: -0.0604233, -0.0019071, -0.0440996, -0.0016210, -0.0587547, 0.0421925
8: -0.0346328, 0.0415302, -0.0224196, 0.0200098, -0.0546426, 0.0639498
9: -0.0361069, 0.0522441, -0.0233640, 0.0382118, -0.0743187, 0.0756081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 175

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0392064
time: 2.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0393476
time: 3.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0512675, 0.0401800, -0.0606929, 0.0404093, -0.0916769, 0.1008729
1: -0.0362606, 0.0298107, -0.0435978, 0.0367567, -0.0730172, 0.0734085
2: -0.0769992, 0.0286935, -0.0983211, 0.0178730, -0.0948722, 0.1270147
3: 0.9989788, 1.0309430, 1.0007681, 1.0379744, -0.0389956, 0.0301750
4: -0.0182797, 0.0552120, -0.0229516, 0.0618462, -0.0801259, 0.0781636
5: -0.0231991, 0.0957533, -0.0250221, 0.1086368, -0.1318359, 0.1207754
6: -0.0728762, 0.0451669, -0.0860204, 0.0613235, -0.1341997, 0.1311873
7: -0.0604233, -0.0019071, -0.0646124, -0.0016547, -0.0587225, 0.0627053
8: -0.0346328, 0.0415302, -0.0377957, 0.0530261, -0.0876588, 0.0793259
9: -0.0361069, 0.0522441, -0.0373992, 0.0646541, -0.1007610, 0.0896433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 175

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0392064
time: 2.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0393475
time: 2.10 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0553680, 0.0438941, -0.0301566, 0.0284800, -0.0838480, 0.0740507
1: -0.0385630, 0.0310579, -0.0227842, 0.0220479, -0.0606110, 0.0538421
2: -0.0808494, 0.0322231, -0.0526679, 0.0172565, -0.0981058, 0.0848910
3: 0.9977880, 1.0318027, 1.0028855, 1.0237423, -0.0259543, 0.0289172
4: -0.0203493, 0.0594503, -0.0104643, 0.0351683, -0.0555176, 0.0699146
5: -0.0263120, 0.0994807, -0.0090224, 0.0733596, -0.0996716, 0.1085031
6: -0.0764613, 0.0498778, -0.0526137, 0.0235492, -0.1000106, 0.1024915
7: -0.0635244, -0.0015009, -0.0455391, -0.0015419, -0.0619441, 0.0440382
8: -0.0363263, 0.0464990, -0.0234877, 0.0223963, -0.0587226, 0.0699867
9: -0.0390428, 0.0544935, -0.0243440, 0.0400705, -0.0791132, 0.0788376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0392649
time: 2.46 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0393659
time: 2.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0553680, 0.0438941, -0.0714762, 0.0413009, -0.0966689, 0.1153702
1: -0.0385630, 0.0310579, -0.0535659, 0.0378393, -0.0764023, 0.0846239
2: -0.0808494, 0.0322231, -0.1064235, 0.0179265, -0.0987758, 0.1386466
3: 0.9977880, 1.0318027, 0.9965479, 1.0390097, -0.0412217, 0.0352548
4: -0.0203493, 0.0594503, -0.0239041, 0.0707982, -0.0911475, 0.0833544
5: -0.0263120, 0.0994807, -0.0261869, 0.1238516, -0.1501635, 0.1256676
6: -0.0764613, 0.0498778, -0.0922762, 0.0641711, -0.1406324, 0.1421540
7: -0.0635244, -0.0015009, -0.0660324, 0.0043139, -0.0678383, 0.0645316
8: -0.0363263, 0.0464990, -0.0455584, 0.0553422, -0.0916685, 0.0920574
9: -0.0390428, 0.0544935, -0.0419880, 0.0664949, -0.1055377, 0.0964815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 114

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0392649
time: 2.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0393659
time: 2.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0894885, 0.0787659, -0.0121865, 0.0205237, -0.1100122, 0.0909524
1: -0.0576692, 0.0449978, -0.0111123, 0.0124408, -0.0701100, 0.0561102
2: -0.1183090, 0.0591695, -0.0241002, 0.0168150, -0.1351241, 0.0832696
3: 0.9881021, 1.0416235, 1.0043809, 1.0153571, -0.0272551, 0.0372425
4: -0.0361962, 0.0985007, -0.0042961, 0.0178014, -0.0539977, 0.1027968
5: -0.0535794, 0.1317999, -0.0016726, 0.0494892, -0.1030687, 0.1334725
6: -0.1085988, 0.0928492, -0.0314768, 0.0039164, -0.1125152, 0.1243259
7: -0.0904320, 0.0057776, -0.0331831, -0.0010622, -0.0893063, 0.0389607
8: -0.0522285, 0.0906712, -0.0149946, 0.0068247, -0.0590532, 0.1056658
9: -0.0630476, 0.0772705, -0.0164003, 0.0235109, -0.0865585, 0.0936708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392071, upper bound: 0.0388374
time: 2.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391909, upper bound: 0.0388402
time: 2.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0872955, 0.0765317, -0.0143201, 0.0220417, -0.1093371, 0.0908518
1: -0.0564416, 0.0441290, -0.0121890, 0.0143456, -0.0707871, 0.0563181
2: -0.1159485, 0.0574054, -0.0287677, 0.0168884, -0.1328370, 0.0861731
3: 0.9887336, 1.0410101, 1.0040791, 1.0169575, -0.0282239, 0.0369310
4: -0.0351594, 0.0960179, -0.0043068, 0.0210396, -0.0561990, 0.1003248
5: -0.0518323, 0.1297403, -0.0023487, 0.0539216, -0.1057539, 0.1320890
6: -0.1065541, 0.0901084, -0.0354061, 0.0049157, -0.1114698, 0.1255146
7: -0.0887117, 0.0051873, -0.0353275, -0.0007037, -0.0879486, 0.0405148
8: -0.0512196, 0.0878354, -0.0165780, 0.0077708, -0.0589904, 0.1044133
9: -0.0614965, 0.0758333, -0.0177263, 0.0266035, -0.0881000, 0.0935596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0392106, upper bound: 0.0388459
time: 2.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391962, upper bound: 0.0388484
time: 2.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0935839, 0.0827773, -0.0134517, 0.0214168, -0.1150007, 0.0962290
1: -0.0599721, 0.0463798, -0.0117587, 0.0135254, -0.0734975, 0.0581385
2: -0.1223688, 0.0626158, -0.0268147, 0.0168706, -0.1392394, 0.0894305
3: 0.9868993, 1.0425212, 1.0041903, 1.0162247, -0.0293255, 0.0383309
4: -0.0382215, 0.1029404, -0.0043048, 0.0197152, -0.0579367, 0.1072452
5: -0.0566846, 0.1355952, -0.0019877, 0.0521660, -0.1088506, 0.1375829
6: -0.1122284, 0.0977860, -0.0337665, 0.0044656, -0.1166939, 0.1315525
7: -0.0936050, 0.0069467, -0.0344475, -0.0009788, -0.0925726, 0.0413942
8: -0.0539397, 0.0958789, -0.0159007, 0.0073213, -0.0612610, 0.1117795
9: -0.0659713, 0.0797567, -0.0171721, 0.0253812, -0.0913526, 0.0969288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393154, upper bound: 0.0388691
time: 2.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393000, upper bound: 0.0388721
time: 2.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0914584, 0.0805594, -0.0156270, 0.0229312, -0.1143897, 0.0961864
1: -0.0587833, 0.0455179, -0.0128742, 0.0154324, -0.0742157, 0.0583921
2: -0.1200574, 0.0608957, -0.0315041, 0.0169430, -0.1370004, 0.0923997
3: 0.9875135, 1.0419132, 1.0038898, 1.0178411, -0.0303276, 0.0380234
4: -0.0372065, 0.1005062, -0.0043605, 0.0229400, -0.0601465, 0.1048667
5: -0.0549635, 0.1335959, -0.0028968, 0.0565681, -0.1115316, 0.1364927
6: -0.1102139, 0.0951006, -0.0376878, 0.0055793, -0.1157932, 0.1327884
7: -0.0919319, 0.0063675, -0.0365916, -0.0006188, -0.0912624, 0.0429591
8: -0.0529477, 0.0930899, -0.0174932, 0.0083985, -0.0613463, 0.1105831
9: -0.0644512, 0.0783430, -0.0184999, 0.0284567, -0.0929078, 0.0968428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393189, upper bound: 0.0388783
time: 2.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393057, upper bound: 0.0388811
time: 3.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1019184, 0.0912900, -0.0203980, 0.0246911, -0.1266095, 0.1116880
1: -0.0646408, 0.0497407, -0.0161409, 0.0173656, -0.0820064, 0.0658816
2: -0.1314352, 0.0692700, -0.0381279, 0.0170492, -0.1484844, 0.1073979
3: 0.9845102, 1.0448663, 1.0035952, 1.0192623, -0.0347521, 0.0412711
4: -0.0421351, 0.1124249, -0.0064184, 0.0267415, -0.0688766, 0.1188433
5: -0.0633498, 0.1434441, -0.0041185, 0.0621888, -0.1255385, 0.1475625
6: -0.1200230, 0.1082578, -0.0420887, 0.0113773, -0.1314003, 0.1503465
7: -0.1001596, 0.0091771, -0.0394776, -0.0014890, -0.0986132, 0.0486547
8: -0.0577889, 0.1066850, -0.0189823, 0.0127371, -0.0705260, 0.1256673
9: -0.0718590, 0.0852649, -0.0202348, 0.0322708, -0.1041297, 0.1054997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394064, upper bound: 0.0391726
time: 3.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0391909, upper bound: 0.0391761
time: 3.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0995675, 0.0888540, -0.0233905, 0.0258951, -0.1254627, 0.1122445
1: -0.0633234, 0.0487788, -0.0181360, 0.0189495, -0.0822728, 0.0669147
2: -0.1288474, 0.0673923, -0.0427787, 0.0171068, -0.1459542, 0.1101711
3: 0.9851870, 1.0441976, 1.0033559, 1.0208445, -0.0356575, 0.0408417
4: -0.0410283, 0.1097278, -0.0076240, 0.0293885, -0.0704168, 0.1173518
5: -0.0614561, 0.1412105, -0.0057408, 0.0655814, -0.1270376, 0.1469513
6: -0.1177940, 0.1052974, -0.0454427, 0.0151637, -0.1329578, 0.1507401
7: -0.0982993, 0.0085422, -0.0413779, -0.0011744, -0.0970713, 0.0499202
8: -0.0566974, 0.1036123, -0.0204936, 0.0157064, -0.0724038, 0.1241058
9: -0.0701854, 0.0836862, -0.0215343, 0.0346871, -0.1048725, 0.1052205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 112

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0394038, upper bound: 0.0391526
time: 2.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0393920, upper bound: 0.0391565
time: 2.34 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.40 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0386078, upper bound: 0.0388961
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0397111, upper bound: 0.0393539
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0386078, upper bound: 0.0388961
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0397111, upper bound: 0.0393539
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0386113, upper bound: 0.0388983
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0397027, upper bound: 0.0393540
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0386113, upper bound: 0.0388983
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0397027, upper bound: 0.0393540
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0398207
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0399996
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0398207
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398212, upper bound: 0.0399996
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0399327
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0400520
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0399327
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0399694, upper bound: 0.0400520
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0397215, upper bound: 0.0395833
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0396998, upper bound: 0.0395867
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0397215, upper bound: 0.0395833
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0396998, upper bound: 0.0395867
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398954, upper bound: 0.0396489
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398741, upper bound: 0.0396524
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398954, upper bound: 0.0396489
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398741, upper bound: 0.0396524
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398793, upper bound: 0.0399656
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398564, upper bound: 0.0399694
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398794, upper bound: 0.0399656
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0398564, upper bound: 0.0399694
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0400508, upper bound: 0.0400227
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0400260, upper bound: 0.0400260
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0400508, upper bound: 0.0400227
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0400260, upper bound: 0.0400260
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0380221, upper bound: 0.0381320
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0390614, upper bound: 0.0385066
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0381229, upper bound: 0.0382281
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0391523, upper bound: 0.0386136
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0380208, upper bound: 0.0381330
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0390508, upper bound: 0.0385066
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0381234, upper bound: 0.0382282
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0391421, upper bound: 0.0386136
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0392064
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0393476
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0392064
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0394197, upper bound: 0.0393475
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0392649
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0393659
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0392649
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0395198, upper bound: 0.0393659
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0392071, upper bound: 0.0388374
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0391909, upper bound: 0.0388402
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0392106, upper bound: 0.0388459
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0391962, upper bound: 0.0388484
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0393154, upper bound: 0.0388691
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0393000, upper bound: 0.0388721
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0393189, upper bound: 0.0388783
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0393057, upper bound: 0.0388811
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0394064, upper bound: 0.0391726
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0391909, upper bound: 0.0391761
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0394038, upper bound: 0.0391526
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.40
Output dim: 3, lower bound: -0.0393920, upper bound: 0.0391565
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0397077, upper bound: 0.0393651
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0395020
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0394948
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0395019
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389794, upper bound: 0.0394948
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393570
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393458
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393570
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389772, upper bound: 0.0393458
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393476, upper bound: 0.0397377
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393660, upper bound: 0.0398371
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393476, upper bound: 0.0397377
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393660, upper bound: 0.0398371
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0394905
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0397458
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0394905
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393341, upper bound: 0.0397458
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0388818, upper bound: 0.0389916
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0388772, upper bound: 0.0389921
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389118, upper bound: 0.0390823
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389112, upper bound: 0.0390831
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0389112, upper bound: 0.0390831
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0391971, upper bound: 0.0391006
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0391642, upper bound: 0.0391004
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.40
Output dim: 3, lower bound: -0.0393617, upper bound: 0.0393617

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.32 + 596.33 = 600.64 seconds
