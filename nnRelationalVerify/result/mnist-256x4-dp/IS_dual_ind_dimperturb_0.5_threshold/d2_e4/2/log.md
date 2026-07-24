## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13486788544


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0699447, 0.0238695, -0.0699447, 0.0238695, -0.0938142, 0.0938142)
1: (-0.0650581, 0.0466669, -0.0650581, 0.0466669, -0.1117250, 0.1117250)
2: (-0.0702461, 0.1393103, -0.0702461, 0.1393103, -0.2095564, 0.2095564)
3: (-0.0228521, 0.0706787, -0.0228521, 0.0706787, -0.0935308, 0.0935308)
4: (-0.0660886, 0.0802853, -0.0660886, 0.0802853, -0.1463740, 0.1463740)
5: (-0.0528843, 0.0634429, -0.0528843, 0.0634429, -0.1163272, 0.1163272)
6: (-0.1225017, 0.0829012, -0.1225017, 0.0829012, -0.2054029, 0.2054029)
7: (0.8246818, 1.0194728, 0.8246818, 1.0194728, -0.1947911, 0.1947911)
8: (-0.0723004, 0.1192246, -0.0723004, 0.1192246, -0.1915250, 0.1915250)
9: (-0.0929379, 0.0865260, -0.0929379, 0.0865260, -0.1794639, 0.1794639)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.51 + 1.84 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.90 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.90
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.90
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0596306, 0.0210185, -0.0689286, 0.0228050, -0.0824356, 0.0899472
1: -0.0547545, 0.0388688, -0.0640430, 0.0458546, -0.1006091, 0.1029118
2: -0.0587167, 0.1274874, -0.0689136, 0.1381456, -0.1968624, 0.1964010
3: -0.0178391, 0.0620759, -0.0217621, 0.0698312, -0.0876703, 0.0838379
4: -0.0562320, 0.0673983, -0.0650575, 0.0790157, -0.1352478, 0.1324558
5: -0.0419723, 0.0530227, -0.0515186, 0.0624164, -0.1043887, 0.1045413
6: -0.1105427, 0.0712230, -0.1213236, 0.0812044, -0.1917471, 0.1925466
7: 0.8404446, 1.0132688, 0.8262346, 1.0187657, -0.1783211, 0.1870342
8: -0.0653486, 0.1055762, -0.0694905, 0.1178800, -0.1814894, 0.1750667
9: -0.0843642, 0.0674305, -0.0919130, 0.0846449, -0.1690090, 0.1593435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.80 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0681584, 0.0222832, -0.0692378, 0.0231290, -0.0912874, 0.0915210
1: -0.0632736, 0.0452607, -0.0643519, 0.0461018, -0.1093754, 0.1096125
2: -0.0680005, 0.1372627, -0.0693191, 0.1385000, -0.2065006, 0.2065818
3: -0.0210363, 0.0691887, -0.0220938, 0.0700890, -0.0911253, 0.0912825
4: -0.0643055, 0.0780534, -0.0653713, 0.0794020, -0.1437075, 0.1434246
5: -0.0506266, 0.0616383, -0.0519341, 0.0627287, -0.1133553, 0.1135724
6: -0.1204305, 0.0801875, -0.1216820, 0.0817208, -0.2021513, 0.2018695
7: 0.8274115, 1.0182769, 0.8257620, 1.0189806, -0.1915691, 0.1925150
8: -0.0684833, 0.1168608, -0.0707003, 0.1182891, -0.1867724, 0.1875611
9: -0.0912249, 0.0832189, -0.0922248, 0.0852173, -0.1764422, 0.1754437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.21 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0585701, 0.0200382, -0.0548163, 0.0202071, -0.0787772, 0.0748544
1: -0.0536950, 0.0380739, -0.0499450, 0.0352603, -0.0889553, 0.0880188
2: -0.0575621, 0.1262718, -0.0534753, 0.1219687, -0.1795308, 0.1797471
3: -0.0171048, 0.0611913, -0.0172313, 0.0580603, -0.0751651, 0.0784226
4: -0.0552279, 0.0660733, -0.0516741, 0.0613829, -0.1166108, 0.1177474
5: -0.0408961, 0.0519512, -0.0370864, 0.0481587, -0.0890548, 0.0890376
6: -0.1093131, 0.0701081, -0.1049606, 0.0661621, -0.1754752, 0.1750687
7: 0.8420653, 1.0126458, 0.8478023, 1.0104415, -0.1683762, 0.1648435
8: -0.0628008, 0.1041729, -0.0632398, 0.0992054, -0.1607398, 0.1646181
9: -0.0835109, 0.0654671, -0.0804909, 0.0585171, -0.1420280, 0.1459581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0590883, 0.0207603, -0.0674935, 0.0220221, -0.0811104, 0.0882538
1: -0.0542127, 0.0384623, -0.0626093, 0.0447622, -0.0989749, 0.1010716
2: -0.0581262, 0.1268657, -0.0672765, 0.1365005, -0.1946267, 0.1941423
3: -0.0176457, 0.0616236, -0.0204764, 0.0686341, -0.0862798, 0.0821000
4: -0.0557185, 0.0667206, -0.0636759, 0.0772226, -0.1329411, 0.1303965
5: -0.0414219, 0.0524747, -0.0499518, 0.0609664, -0.1023883, 0.1024265
6: -0.1099139, 0.0706529, -0.1196595, 0.0794885, -0.1894024, 0.1903124
7: 0.8412734, 1.0129503, 0.8284279, 1.0178863, -0.1766129, 0.1845224
8: -0.0646776, 0.1048585, -0.0677219, 0.1159808, -0.1787413, 0.1721071
9: -0.0839279, 0.0664264, -0.0906899, 0.0819878, -0.1659157, 0.1571162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0671083, 0.0218708, -0.0551136, 0.0207187, -0.0878269, 0.0769844
1: -0.0622245, 0.0444735, -0.0502420, 0.0354831, -0.0977076, 0.0947156
2: -0.0668572, 0.1360589, -0.0537991, 0.1223096, -0.1891668, 0.1898580
3: -0.0201521, 0.0683128, -0.0176145, 0.0583083, -0.0784604, 0.0859273
4: -0.0633112, 0.0767413, -0.0519555, 0.0617545, -0.1250657, 0.1286968
5: -0.0495609, 0.0605773, -0.0373882, 0.0484591, -0.0980199, 0.0979655
6: -0.1192130, 0.0790835, -0.1053054, 0.0664746, -0.1856875, 0.1843889
7: 0.8290167, 1.0176600, 0.8473479, 1.0106161, -0.1815994, 0.1703121
8: -0.0660105, 0.1154711, -0.0645693, 0.0995989, -0.1656094, 0.1768974
9: -0.0903801, 0.0812745, -0.0807301, 0.0590676, -0.1494476, 0.1620046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0675952, 0.0220620, -0.0678001, 0.0223565, -0.0899517, 0.0898622
1: -0.0627109, 0.0448384, -0.0629156, 0.0449921, -0.1077030, 0.1077541
2: -0.0673873, 0.1366171, -0.0676104, 0.1368520, -0.2042393, 0.2042274
3: -0.0205620, 0.0687190, -0.0207346, 0.0688899, -0.0894519, 0.0894536
4: -0.0637722, 0.0773496, -0.0639662, 0.0776057, -0.1413779, 0.1413159
5: -0.0500550, 0.0610692, -0.0502630, 0.0612762, -0.1113312, 0.1113322
6: -0.1197775, 0.0795954, -0.1200150, 0.0798109, -0.1995883, 0.1996104
7: 0.8282725, 1.0179460, 0.8279593, 1.0180664, -0.1897939, 0.1899867
8: -0.0677531, 0.1161154, -0.0688259, 0.1163867, -0.1839149, 0.1840995
9: -0.0907717, 0.0821761, -0.0909366, 0.0825554, -0.1733271, 0.1731127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0484563, 0.0174033, -0.0537605, 0.0190864, -0.0675427, 0.0711638
1: -0.0435982, 0.0304631, -0.0488903, 0.0344689, -0.0780671, 0.0793534
2: -0.0465783, 0.1146509, -0.0523261, 0.1207585, -0.1673368, 0.1669770
3: -0.0151312, 0.0527219, -0.0163919, 0.0571797, -0.0723109, 0.0691138
4: -0.0456562, 0.0535166, -0.0506745, 0.0600639, -0.1057201, 0.1041911
5: -0.0306031, 0.0418555, -0.0360150, 0.0470921, -0.0776951, 0.0778705
6: -0.0976646, 0.0594518, -0.1037366, 0.0650522, -0.1627168, 0.1631884
7: 0.8575585, 1.0067332, 0.8494159, 1.0098214, -0.1522629, 0.1573173
8: -0.0559528, 0.0909652, -0.0603272, 0.0978085, -0.1502527, 0.1489243
9: -0.0753681, 0.0467561, -0.0796415, 0.0565624, -0.1319305, 0.1263976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0574779, 0.0191493, -0.0544394, 0.0198922, -0.0773700, 0.0735888
1: -0.0526038, 0.0372553, -0.0495685, 0.0349778, -0.0875816, 0.0868238
2: -0.0563731, 0.1250197, -0.0530651, 0.1215367, -0.1779098, 0.1780848
3: -0.0164390, 0.0602802, -0.0169955, 0.0577460, -0.0741850, 0.0772757
4: -0.0541938, 0.0647085, -0.0513172, 0.0609121, -0.1151060, 0.1160258
5: -0.0397875, 0.0508478, -0.0367040, 0.0477780, -0.0875655, 0.0875518
6: -0.1080467, 0.0689600, -0.1045237, 0.0657659, -0.1738125, 0.1734836
7: 0.8437347, 1.0120044, 0.8483784, 1.0102201, -0.1664853, 0.1636260
8: -0.0604908, 0.1027275, -0.0624214, 0.0987068, -0.1568601, 0.1624322
9: -0.0826322, 0.0634447, -0.0801877, 0.0578194, -0.1404516, 0.1436324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0489318, 0.0181186, -0.0663726, 0.0215820, -0.0705138, 0.0844912
1: -0.0440652, 0.0308436, -0.0614896, 0.0439222, -0.0879874, 0.0923332
2: -0.0470747, 0.1152141, -0.0660564, 0.1352157, -0.1822903, 0.1812705
3: -0.0156670, 0.0531454, -0.0195327, 0.0676992, -0.0833662, 0.0726780
4: -0.0460953, 0.0540470, -0.0626148, 0.0758221, -0.1219174, 0.1166617
5: -0.0311064, 0.0422385, -0.0488143, 0.0598341, -0.0909404, 0.0910528
6: -0.0981537, 0.0599677, -0.1183599, 0.0783102, -0.1764639, 0.1783276
7: 0.8568081, 1.0069911, 0.8301411, 1.0172280, -0.1604199, 0.1768501
8: -0.0578119, 0.0914546, -0.0647884, 0.1144976, -0.1682356, 0.1562430
9: -0.0757555, 0.0476074, -0.0897882, 0.0799125, -0.1556680, 0.1373956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0579937, 0.0198462, -0.0670969, 0.0218664, -0.0798601, 0.0869431
1: -0.0531191, 0.0376419, -0.0622131, 0.0444649, -0.0975841, 0.0998549
2: -0.0569345, 0.1256109, -0.0668448, 0.1360459, -0.1929804, 0.1924557
3: -0.0169610, 0.0607105, -0.0201424, 0.0683033, -0.0852643, 0.0808529
4: -0.0546822, 0.0653531, -0.0633004, 0.0767270, -0.1314092, 0.1286534
5: -0.0403110, 0.0513688, -0.0495493, 0.0605657, -0.1008768, 0.1009181
6: -0.1086448, 0.0695022, -0.1191996, 0.0790716, -0.1877164, 0.1887017
7: 0.8429464, 1.0123074, 0.8290340, 1.0176533, -0.1747069, 0.1832734
8: -0.0623019, 0.1034102, -0.0668815, 0.1154560, -0.1747794, 0.1699249
9: -0.0830472, 0.0643997, -0.0903708, 0.0812536, -0.1643007, 0.1547705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0570505, 0.0186832, -0.0540576, 0.0196020, -0.0766525, 0.0727409
1: -0.0521770, 0.0369349, -0.0491871, 0.0346917, -0.0868686, 0.0861220
2: -0.0559077, 0.1245299, -0.0526495, 0.1210991, -0.1770068, 0.1771794
3: -0.0160899, 0.0599238, -0.0167781, 0.0574275, -0.0735174, 0.0767019
4: -0.0537893, 0.0641746, -0.0509558, 0.0604351, -0.1142244, 0.1151303
5: -0.0393539, 0.0504160, -0.0363165, 0.0473923, -0.0867462, 0.0867325
6: -0.1075512, 0.0685107, -0.1040810, 0.0653645, -0.1729157, 0.1725917
7: 0.8443877, 1.0117536, 0.8489619, 1.0099959, -0.1656083, 0.1627916
8: -0.0592793, 0.1021621, -0.0616671, 0.0982016, -0.1555525, 0.1607766
9: -0.0822884, 0.0626536, -0.0798806, 0.0571124, -0.1394008, 0.1425341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0659775, 0.0214269, -0.0547429, 0.0203948, -0.0863723, 0.0761698
1: -0.0610949, 0.0436260, -0.0498717, 0.0352053, -0.0963002, 0.0934977
2: -0.0656262, 0.1347628, -0.0533956, 0.1218847, -0.1875110, 0.1881583
3: -0.0192000, 0.0673697, -0.0173719, 0.0579991, -0.0771992, 0.0847416
4: -0.0622407, 0.0753285, -0.0516047, 0.0612914, -0.1235321, 0.1269331
5: -0.0484134, 0.0594348, -0.0370120, 0.0480846, -0.0964980, 0.0964469
6: -0.1179018, 0.0778949, -0.1048756, 0.0660850, -0.1839868, 0.1827706
7: 0.8307446, 1.0169960, 0.8479143, 1.0103984, -0.1796538, 0.1690817
8: -0.0636886, 0.1139749, -0.0637276, 0.0991084, -0.1625630, 0.1746783
9: -0.0894704, 0.0791810, -0.0804319, 0.0583814, -0.1478517, 0.1596129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0575544, 0.0193270, -0.0666843, 0.0217044, -0.0792588, 0.0860113
1: -0.0526803, 0.0373126, -0.0618009, 0.0441557, -0.0968360, 0.0991135
2: -0.0564564, 0.1251074, -0.0663956, 0.1355730, -0.1920293, 0.1915030
3: -0.0165721, 0.0603441, -0.0197951, 0.0679592, -0.0845313, 0.0801392
4: -0.0542663, 0.0648042, -0.0629098, 0.0762115, -0.1304778, 0.1277141
5: -0.0398652, 0.0509251, -0.0491306, 0.0601489, -0.1000141, 0.1000557
6: -0.1081355, 0.0690404, -0.1187213, 0.0786379, -0.1867734, 0.1877617
7: 0.8436177, 1.0120496, 0.8296646, 1.0174112, -0.1737936, 0.1823850
8: -0.0609524, 0.1028287, -0.0659201, 0.1149101, -0.1733587, 0.1682145
9: -0.0826938, 0.0635866, -0.0900390, 0.0804896, -0.1631834, 0.1536256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0664663, 0.0216188, -0.0674078, 0.0220439, -0.0885103, 0.0890266
1: -0.0615832, 0.0439924, -0.0625237, 0.0446980, -0.1062812, 0.1065161
2: -0.0661584, 0.1353231, -0.0671833, 0.1364024, -0.2025607, 0.2025064
3: -0.0196116, 0.0677775, -0.0204043, 0.0685627, -0.0881743, 0.0881818
4: -0.0627036, 0.0759392, -0.0635949, 0.0771155, -0.1398191, 0.1395340
5: -0.0489094, 0.0599287, -0.0498649, 0.0608799, -0.1097893, 0.1097936
6: -0.1184686, 0.0784088, -0.1195602, 0.0793984, -0.1978670, 0.1979690
7: 0.8299977, 1.0172830, 0.8285588, 1.0178360, -0.1878383, 0.1887242
8: -0.0654327, 0.1146217, -0.0680135, 0.1158675, -0.1802886, 0.1819011
9: -0.0898636, 0.0800861, -0.0906210, 0.0818292, -0.1716928, 0.1707071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.57 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0480745, 0.0169158, -0.0523349, 0.0173670, -0.0654415, 0.0692506
1: -0.0432522, 0.0301577, -0.0474661, 0.0334004, -0.0766526, 0.0776238
2: -0.0461798, 0.1142399, -0.0507740, 0.1191243, -0.1653041, 0.1650139
3: -0.0147661, 0.0523820, -0.0151040, 0.0559906, -0.0707567, 0.0674861
4: -0.0453956, 0.0530908, -0.0493248, 0.0582826, -0.1036782, 0.1024156
5: -0.0302254, 0.0415480, -0.0345682, 0.0456518, -0.0758772, 0.0761162
6: -0.0972719, 0.0590771, -0.1020836, 0.0635536, -0.1608255, 0.1611607
7: 0.8581051, 1.0065259, 0.8515948, 1.0089843, -0.1508793, 0.1549311
8: -0.0546859, 0.0905724, -0.0558586, 0.0959219, -0.1474220, 0.1450049
9: -0.0750571, 0.0462689, -0.0784946, 0.0539230, -0.1289801, 0.1247634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0473172, 0.0164421, -0.0690994, 0.0226526, -0.0699699, 0.0855415
1: -0.0425660, 0.0295518, -0.0642135, 0.0459659, -0.0885319, 0.0937653
2: -0.0453894, 0.1134249, -0.0690248, 0.1383413, -0.1837307, 0.1824497
3: -0.0144113, 0.0517078, -0.0218284, 0.0699735, -0.0843848, 0.0735362
4: -0.0448786, 0.0522462, -0.0651962, 0.0792290, -0.1241076, 0.1174424
5: -0.0294761, 0.0409379, -0.0515815, 0.0625888, -0.0920649, 0.0925194
6: -0.0964930, 0.0583339, -0.1215214, 0.0811766, -0.1776696, 0.1798554
7: 0.8591890, 1.0061147, 0.8259737, 1.0188295, -0.1596405, 0.1801410
8: -0.0534550, 0.0897932, -0.0550776, 0.1181058, -0.1684567, 0.1448708
9: -0.0744401, 0.0453023, -0.0919818, 0.0849609, -0.1594010, 0.1372841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0570910, 0.0186682, -0.0530305, 0.0181701, -0.0752611, 0.0716987
1: -0.0522174, 0.0369653, -0.0481611, 0.0339218, -0.0861392, 0.0851263
2: -0.0559518, 0.1245763, -0.0515314, 0.1199218, -0.1758735, 0.1761077
3: -0.0160787, 0.0599576, -0.0157056, 0.0565708, -0.0726495, 0.0756632
4: -0.0538276, 0.0642253, -0.0499834, 0.0591518, -0.1129794, 0.1142087
5: -0.0393950, 0.0504569, -0.0352742, 0.0463546, -0.0857496, 0.0857311
6: -0.1075982, 0.0685533, -0.1028901, 0.0642849, -0.1718831, 0.1714434
7: 0.8443259, 1.0117774, 0.8505316, 1.0093927, -0.1650668, 0.1612458
8: -0.0592404, 0.1022156, -0.0579457, 0.0968424, -0.1540485, 0.1583325
9: -0.0823210, 0.0627286, -0.0790542, 0.0552110, -0.1375320, 0.1417828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0563064, 0.0181736, -0.0698397, 0.0229433, -0.0792497, 0.0880133
1: -0.0514336, 0.0363772, -0.0649531, 0.0465208, -0.0979544, 0.1013302
2: -0.0550977, 0.1236768, -0.0698307, 0.1391900, -0.1942877, 0.1935076
3: -0.0157082, 0.0593032, -0.0224518, 0.0705910, -0.0862992, 0.0817550
4: -0.0530848, 0.0632448, -0.0658971, 0.0801540, -0.1332389, 0.1291420
5: -0.0385987, 0.0496642, -0.0523328, 0.0633368, -0.1019355, 0.1019970
6: -0.1066884, 0.0677285, -0.1223798, 0.0819548, -0.1886432, 0.1901083
7: 0.8455250, 1.0113165, 0.8248423, 1.0192642, -0.1737392, 0.1864743
8: -0.0579550, 0.1011773, -0.0571209, 0.1190856, -0.1749936, 0.1582982
9: -0.0816897, 0.0612760, -0.0925775, 0.0863315, -0.1680212, 0.1538535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0485505, 0.0176256, -0.0649075, 0.0210067, -0.0695572, 0.0825332
1: -0.0436835, 0.0305385, -0.0600260, 0.0428240, -0.0865075, 0.0905644
2: -0.0466766, 0.1147522, -0.0644614, 0.1335363, -0.1802129, 0.1792136
3: -0.0152978, 0.0528058, -0.0182992, 0.0664772, -0.0817750, 0.0711049
4: -0.0457205, 0.0536216, -0.0612277, 0.0739916, -0.1197120, 0.1148493
5: -0.0306963, 0.0419314, -0.0473275, 0.0583539, -0.0890502, 0.0892588
6: -0.0977614, 0.0595442, -0.1166612, 0.0767701, -0.1745315, 0.1762054
7: 0.8574239, 1.0067842, 0.8323801, 1.0163677, -0.1589438, 0.1744041
8: -0.0565309, 0.0910621, -0.0602355, 0.1125589, -0.1652323, 0.1512977
9: -0.0754448, 0.0468763, -0.0886095, 0.0772001, -0.1526450, 0.1354858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0477909, 0.0171418, -0.0820888, 0.0277531, -0.0755441, 0.0992306
1: -0.0429953, 0.0299308, -0.0771898, 0.0557019, -0.0986972, 0.1071206
2: -0.0458839, 0.1139347, -0.0831659, 0.1532310, -0.1991149, 0.1971007
3: -0.0149354, 0.0521296, -0.0327649, 0.0808078, -0.0957432, 0.0848945
4: -0.0452020, 0.0527745, -0.0774938, 0.0954588, -0.1406608, 0.1302684
5: -0.0299448, 0.0413195, -0.0647637, 0.0757121, -0.1056568, 0.1060832
6: -0.0969803, 0.0587989, -0.1365824, 0.0948313, -0.1918116, 0.1953813
7: 0.8585109, 1.0063720, 0.8061218, 1.0264577, -0.1679468, 0.2002501
8: -0.0552734, 0.0902807, -0.0595233, 0.1352944, -0.1864502, 0.1498039
9: -0.0748261, 0.0459069, -0.1024321, 0.1090098, -0.1838359, 0.1483390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0576058, 0.0193620, -0.0656508, 0.0212986, -0.0789043, 0.0850127
1: -0.0527317, 0.0373510, -0.0607685, 0.0433811, -0.0961127, 0.0981195
2: -0.0565122, 0.1251663, -0.0652706, 0.1343882, -0.1909004, 0.1904369
3: -0.0165983, 0.0603870, -0.0189249, 0.0670971, -0.0836954, 0.0793119
4: -0.0543148, 0.0648683, -0.0619314, 0.0749202, -0.1292350, 0.1267997
5: -0.0399173, 0.0509769, -0.0480817, 0.0591048, -0.0990221, 0.0990586
6: -0.1081949, 0.0690944, -0.1175229, 0.0775514, -0.1857463, 0.1866173
7: 0.8435392, 1.0120796, 0.8312440, 1.0168042, -0.1732650, 0.1808356
8: -0.0610433, 0.1028968, -0.0623362, 0.1135424, -0.1718861, 0.1652329
9: -0.0827351, 0.0636815, -0.0892074, 0.0785762, -0.1613112, 0.1528889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0568199, 0.0188682, -0.0828703, 0.0280600, -0.0848799, 0.1017385
1: -0.0519466, 0.0367621, -0.0779704, 0.0562876, -0.1082342, 0.1147326
2: -0.0556567, 0.1242655, -0.0840167, 0.1541268, -0.2097835, 0.2082822
3: -0.0162285, 0.0597315, -0.0334229, 0.0814597, -0.0976881, 0.0931544
4: -0.0535709, 0.0638865, -0.0782336, 0.0964351, -0.1500061, 0.1421201
5: -0.0391198, 0.0501830, -0.0655568, 0.0765016, -0.1156214, 0.1157398
6: -0.1072838, 0.0682684, -0.1374885, 0.0956528, -0.2029366, 0.2057568
7: 0.8447402, 1.0116181, 0.8049276, 1.0269166, -0.1821764, 0.2066905
8: -0.0597601, 0.1018568, -0.0615063, 0.1363285, -0.1931271, 0.1633632
9: -0.0821029, 0.0622266, -0.1030608, 0.1104565, -0.1925594, 0.1652875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0566322, 0.0181964, -0.0526319, 0.0178920, -0.0745242, 0.0708284
1: -0.0517591, 0.0366214, -0.0477629, 0.0336231, -0.0853821, 0.0843843
2: -0.0554524, 0.1240503, -0.0510974, 0.1194649, -0.1749173, 0.1751477
3: -0.0157253, 0.0595749, -0.0154973, 0.0562384, -0.0719637, 0.0750722
4: -0.0533932, 0.0636520, -0.0496061, 0.0586538, -0.1120470, 0.1132581
5: -0.0389293, 0.0499934, -0.0348697, 0.0459519, -0.0848812, 0.0848631
6: -0.1070662, 0.0680710, -0.1024280, 0.0638659, -0.1709321, 0.1704990
7: 0.8450271, 1.0115077, 0.8511406, 1.0091587, -0.1641316, 0.1603671
8: -0.0580142, 0.1016085, -0.0572231, 0.0963150, -0.1527374, 0.1566237
9: -0.0819518, 0.0618791, -0.0787336, 0.0544731, -0.1364249, 0.1406127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0558319, 0.0177137, -0.0694199, 0.0227785, -0.0786104, 0.0871335
1: -0.0509596, 0.0360216, -0.0645337, 0.0462061, -0.0971657, 0.1005553
2: -0.0545812, 0.1231329, -0.0693737, 0.1387087, -0.1932898, 0.1925066
3: -0.0153637, 0.0589074, -0.0220983, 0.0702409, -0.0856046, 0.0810058
4: -0.0526355, 0.0626521, -0.0654997, 0.0796295, -0.1322650, 0.1281518
5: -0.0381172, 0.0491849, -0.0519068, 0.0629127, -0.1010299, 0.1010917
6: -0.1061383, 0.0672297, -0.1218931, 0.0815135, -0.1876518, 0.1891228
7: 0.8462502, 1.0110378, 0.8254838, 1.0190176, -0.1727674, 0.1855540
8: -0.0567596, 0.1005495, -0.0563986, 0.1185300, -0.1735321, 0.1569481
9: -0.0813080, 0.0603974, -0.0922397, 0.0855543, -0.1668623, 0.1526371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0655801, 0.0212708, -0.0533342, 0.0186984, -0.0842785, 0.0746050
1: -0.0606978, 0.0433281, -0.0484645, 0.0341494, -0.0948472, 0.0917926
2: -0.0651935, 0.1343072, -0.0518620, 0.1202699, -0.1854634, 0.1861692
3: -0.0188654, 0.0670382, -0.0161013, 0.0568242, -0.0756897, 0.0831395
4: -0.0618645, 0.0748319, -0.0502709, 0.0595312, -0.1213957, 0.1251028
5: -0.0480099, 0.0590334, -0.0355824, 0.0466614, -0.0946713, 0.0946158
6: -0.1174410, 0.0774771, -0.1032422, 0.0646041, -0.1820451, 0.1807194
7: 0.8313521, 1.0167629, 0.8500674, 1.0095711, -0.1782190, 0.1666955
8: -0.0624426, 0.1134489, -0.0593190, 0.0972444, -0.1596870, 0.1705140
9: -0.0891505, 0.0784453, -0.0792986, 0.0557732, -0.1449237, 0.1577439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0648002, 0.0209646, -0.0701649, 0.0230711, -0.0878713, 0.0911295
1: -0.0599188, 0.0427436, -0.0652780, 0.0467645, -0.1066833, 0.1080215
2: -0.0643446, 0.1334132, -0.0701849, 0.1395627, -0.2039073, 0.2035982
3: -0.0182088, 0.0663878, -0.0227256, 0.0708623, -0.0890711, 0.0891133
4: -0.0611261, 0.0738574, -0.0662050, 0.0805604, -0.1416865, 0.1400624
5: -0.0472186, 0.0582454, -0.0526628, 0.0636654, -0.1108840, 0.1109082
6: -0.1165367, 0.0766573, -0.1227570, 0.0822967, -0.1988335, 0.1994143
7: 0.8325440, 1.0163046, 0.8243452, 1.0194550, -0.1869110, 0.1919594
8: -0.0610812, 0.1124169, -0.0584282, 0.1195159, -0.1805971, 0.1708451
9: -0.0885232, 0.0770015, -0.0928392, 0.0869336, -0.1754568, 0.1698406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0571336, 0.0188327, -0.0652201, 0.0211294, -0.0782630, 0.0840528
1: -0.0522599, 0.0369972, -0.0603382, 0.0430583, -0.0953182, 0.0973354
2: -0.0559982, 0.1246251, -0.0648015, 0.1338945, -0.1898927, 0.1894266
3: -0.0162019, 0.0599931, -0.0185623, 0.0667380, -0.0829399, 0.0785554
4: -0.0538679, 0.0642784, -0.0615237, 0.0743820, -0.1282500, 0.1258021
5: -0.0394381, 0.0504999, -0.0476447, 0.0586696, -0.0981077, 0.0981446
6: -0.1076476, 0.0685981, -0.1170235, 0.0770987, -0.1847463, 0.1856216
7: 0.8442609, 1.0118024, 0.8319024, 1.0165510, -0.1722901, 0.1799001
8: -0.0596679, 0.1022719, -0.0613846, 0.1129725, -0.1704470, 0.1636566
9: -0.0823553, 0.0628073, -0.0888610, 0.0777787, -0.1601340, 0.1516683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0563369, 0.0183569, -0.0824200, 0.0278831, -0.0842200, 0.1007769
1: -0.0514641, 0.0364000, -0.0775207, 0.0559501, -0.1074142, 0.1139207
2: -0.0551308, 0.1237118, -0.0835264, 0.1536107, -0.2087415, 0.2072382
3: -0.0158455, 0.0593286, -0.0330437, 0.0810841, -0.0969296, 0.0923724
4: -0.0531136, 0.0632829, -0.0778073, 0.0958726, -0.1489862, 0.1410902
5: -0.0386296, 0.0496950, -0.0651000, 0.0760466, -0.1146762, 0.1147950
6: -0.1067238, 0.0677605, -0.1369663, 0.0951795, -0.2019033, 0.2047268
7: 0.8454784, 1.0113344, 0.8056159, 1.0266521, -0.1811737, 0.2057185
8: -0.0584312, 0.1012177, -0.0607237, 0.1357327, -0.1914797, 0.1619414
9: -0.0817143, 0.0613323, -0.1026985, 0.1096229, -0.1913371, 0.1640307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0660666, 0.0214619, -0.0659610, 0.0214204, -0.0874870, 0.0874228
1: -0.0611839, 0.0436927, -0.0610783, 0.0436136, -0.1047974, 0.1047710
2: -0.0657232, 0.1348649, -0.0656082, 0.1347438, -0.2004669, 0.2004731
3: -0.0192750, 0.0674440, -0.0191861, 0.0673559, -0.0866309, 0.0866301
4: -0.0623250, 0.0754398, -0.0622250, 0.0753077, -0.1376328, 0.1376648
5: -0.0485037, 0.0595249, -0.0483965, 0.0594181, -0.1079218, 0.1079213
6: -0.1180051, 0.0779885, -0.1178826, 0.0778775, -0.1958826, 0.1958711
7: 0.8306086, 1.0170485, 0.8307701, 1.0169865, -0.1863779, 0.1862783
8: -0.0641849, 0.1140927, -0.0635223, 0.1139529, -0.1775084, 0.1776150
9: -0.0895420, 0.0793460, -0.0894569, 0.0791504, -0.1686924, 0.1688029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0652912, 0.0211574, -0.0832076, 0.0281924, -0.0934836, 0.1043650
1: -0.0604093, 0.0431116, -0.0783074, 0.0565404, -0.1169497, 0.1214190
2: -0.0648791, 0.1339761, -0.0843839, 0.1545134, -0.2193925, 0.2183600
3: -0.0186222, 0.0667973, -0.0337068, 0.0817410, -0.1003632, 0.1005041
4: -0.0615910, 0.0744709, -0.0785529, 0.0968566, -0.1584476, 0.1530238
5: -0.0477169, 0.0587415, -0.0658991, 0.0768423, -0.1245592, 0.1246406
6: -0.1171060, 0.0771735, -0.1378795, 0.0960074, -0.2131134, 0.2150529
7: 0.8317935, 1.0165930, 0.8044122, 1.0271146, -0.1953211, 0.2121808
8: -0.0628063, 0.1130667, -0.0626680, 0.1367748, -0.1986880, 0.1757347
9: -0.0889181, 0.0779106, -0.1033322, 0.1110810, -0.1999991, 0.1812427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.66 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.66
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0470737, 0.0156740, -0.0523349, 0.0173670, -0.0644407, 0.0680088
1: -0.0423454, 0.0293570, -0.0474661, 0.0334004, -0.0757458, 0.0768231
2: -0.0451352, 0.1131628, -0.0507740, 0.1191243, -0.1642595, 0.1639369
3: -0.0138359, 0.0514910, -0.0151040, 0.0559906, -0.0698266, 0.0665951
4: -0.0447125, 0.0519745, -0.0493248, 0.0582826, -0.1029951, 0.1012993
5: -0.0292351, 0.0407417, -0.0345682, 0.0456518, -0.0748869, 0.0753100
6: -0.0962425, 0.0580950, -0.1020836, 0.0635536, -0.1597961, 0.1601786
7: 0.8595375, 1.0059826, 0.8515948, 1.0089843, -0.1494468, 0.1543878
8: -0.0514586, 0.0895426, -0.0558586, 0.0959219, -0.1451250, 0.1439636
9: -0.0742418, 0.0449915, -0.0784946, 0.0539230, -0.1281648, 0.1234861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0643372, 0.0188382, -0.0523349, 0.0173670, -0.0817042, 0.0711731
1: -0.0579876, 0.0431690, -0.0474661, 0.0334004, -0.0913880, 0.0906351
2: -0.0631548, 0.1317425, -0.0507740, 0.1191243, -0.1822791, 0.1825165
3: -0.0136197, 0.0668611, -0.0151040, 0.0559906, -0.0696104, 0.0819651
4: -0.0564969, 0.0712292, -0.0493248, 0.0582826, -0.1147795, 0.1205540
5: -0.0463156, 0.0546484, -0.0345682, 0.0456518, -0.0919674, 0.0892166
6: -0.1139990, 0.0750375, -0.1020836, 0.0635536, -0.1775526, 0.1771210
7: 0.8348267, 1.0153522, 0.8515948, 1.0089843, -0.1741576, 0.1637574
8: -0.0507085, 0.1073061, -0.0558586, 0.0959219, -0.1466304, 0.1617801
9: -0.0883066, 0.0670248, -0.0784946, 0.0539230, -0.1422296, 0.1455194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0356602, 0.0150436, -0.0690994, 0.0226526, -0.0583128, 0.0841430
1: -0.0336167, 0.0188196, -0.0642135, 0.0459659, -0.0795825, 0.0830331
2: -0.0314463, 0.0999831, -0.0690248, 0.1383413, -0.1697877, 0.1690079
3: -0.0133638, 0.0437791, -0.0218284, 0.0699735, -0.0833373, 0.0656075
4: -0.0360880, 0.0397449, -0.0651962, 0.0792290, -0.1153170, 0.1049411
5: -0.0197772, 0.0301322, -0.0515815, 0.0625888, -0.0823660, 0.0817137
6: -0.0837465, 0.0460974, -0.1215214, 0.0811766, -0.1649231, 0.1676189
7: 0.8783208, 0.9997565, 0.8259737, 1.0188295, -0.1405087, 0.1737828
8: -0.0498204, 0.0770647, -0.0550776, 0.1181058, -0.1621614, 0.1321423
9: -0.0641747, 0.0302929, -0.0919818, 0.0849609, -0.1491355, 0.1222747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0470289, 0.0167566, -0.0690994, 0.0226526, -0.0696815, 0.0858560
1: -0.0423048, 0.0293211, -0.0642135, 0.0459659, -0.0882707, 0.0935346
2: -0.0450884, 0.1131146, -0.0690248, 0.1383413, -0.1834298, 0.1821394
3: -0.0146469, 0.0514512, -0.0218284, 0.0699735, -0.0846204, 0.0732796
4: -0.0446818, 0.0519245, -0.0651962, 0.0792290, -0.1239108, 0.1171207
5: -0.0291908, 0.0407057, -0.0515815, 0.0625888, -0.0917796, 0.0922871
6: -0.0961964, 0.0580510, -0.1215214, 0.0811766, -0.1773730, 0.1795724
7: 0.8596016, 1.0059586, 0.8259737, 1.0188295, -0.1592278, 0.1799849
8: -0.0542724, 0.0894965, -0.0550776, 0.1181058, -0.1685993, 0.1445741
9: -0.0742052, 0.0449343, -0.0919818, 0.0849609, -0.1591661, 0.1369161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0560730, 0.0175377, -0.0530305, 0.0181701, -0.0742430, 0.0705682
1: -0.0512004, 0.0362022, -0.0481611, 0.0339218, -0.0851222, 0.0843633
2: -0.0548435, 0.1234092, -0.0515314, 0.1199218, -0.1747653, 0.1749406
3: -0.0151620, 0.0591084, -0.0157056, 0.0565708, -0.0717328, 0.0748140
4: -0.0528637, 0.0629532, -0.0499834, 0.0591518, -0.1120155, 0.1129367
5: -0.0383618, 0.0494284, -0.0352742, 0.0463546, -0.0847164, 0.0847025
6: -0.1064177, 0.0674830, -0.1028901, 0.0642849, -0.1707026, 0.1703731
7: 0.8458818, 1.0111794, 0.8505316, 1.0093927, -0.1635109, 0.1606478
8: -0.0560597, 0.1008684, -0.0579457, 0.0968424, -0.1517769, 0.1569788
9: -0.0815019, 0.0608437, -0.0790542, 0.0552110, -0.1367129, 0.1398979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0734499, 0.0243610, -0.0530305, 0.0181701, -0.0916200, 0.0773915
1: -0.0685597, 0.0492267, -0.0481611, 0.0339218, -0.1024815, 0.0973878
2: -0.0737611, 0.1433283, -0.0515314, 0.1199218, -0.1936828, 0.1948597
3: -0.0254914, 0.0736023, -0.0157056, 0.0565708, -0.0820622, 0.0893079
4: -0.0693150, 0.0846649, -0.0499834, 0.0591518, -0.1284668, 0.1346483
5: -0.0559966, 0.0669842, -0.0352742, 0.0463546, -0.1023512, 0.1022584
6: -0.1265658, 0.0857500, -0.1028901, 0.0642849, -0.1908507, 0.1886400
7: 0.8193248, 1.0213845, 0.8505316, 1.0093927, -0.1900679, 0.1708528
8: -0.0551789, 0.1238629, -0.0579457, 0.0968424, -0.1520214, 0.1796775
9: -0.0954819, 0.0930156, -0.0790542, 0.0552110, -0.1506930, 0.1720698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0427713, 0.0167634, -0.0698397, 0.0229433, -0.0657146, 0.0866031
1: -0.0384922, 0.0258950, -0.0649531, 0.0465208, -0.0850130, 0.0908480
2: -0.0406204, 0.1085058, -0.0698307, 0.1391900, -0.1798104, 0.1783366
3: -0.0146520, 0.0477663, -0.0224518, 0.0705910, -0.0852429, 0.0702181
4: -0.0417703, 0.0471483, -0.0658971, 0.0801540, -0.1219243, 0.1130454
5: -0.0249982, 0.0372560, -0.0523328, 0.0633368, -0.0883350, 0.0895888
6: -0.0917918, 0.0538779, -0.1223798, 0.0819548, -0.1737466, 0.1762577
7: 0.8657314, 1.0036635, 0.8248423, 1.0192642, -0.1535329, 0.1788213
8: -0.0542900, 0.0850901, -0.0571209, 0.1190856, -0.1671235, 0.1422110
9: -0.0707375, 0.0394688, -0.0925775, 0.0863315, -0.1570689, 0.1320463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0559901, 0.0184617, -0.0698397, 0.0229433, -0.0789334, 0.0883013
1: -0.0511175, 0.0361401, -0.0649531, 0.0465208, -0.0976383, 0.1010932
2: -0.0547533, 0.1233142, -0.0698307, 0.1391900, -0.1939432, 0.1931450
3: -0.0159239, 0.0590393, -0.0224518, 0.0705910, -0.0865149, 0.0814911
4: -0.0527853, 0.0628496, -0.0658971, 0.0801540, -0.1329394, 0.1287467
5: -0.0382776, 0.0493446, -0.0523328, 0.0633368, -0.1016145, 0.1016774
6: -0.1063216, 0.0673960, -0.1223798, 0.0819548, -0.1882764, 0.1897758
7: 0.8460085, 1.0111308, 0.8248423, 1.0192642, -0.1732557, 0.1862885
8: -0.0587035, 0.1007587, -0.0571209, 0.1190856, -0.1748545, 0.1578796
9: -0.0814353, 0.0606903, -0.0925775, 0.0863315, -0.1677668, 0.1532677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0475435, 0.0163706, -0.0649075, 0.0210067, -0.0685502, 0.0812782
1: -0.0427711, 0.0297328, -0.0600260, 0.0428240, -0.0855951, 0.0897588
2: -0.0456256, 0.1136685, -0.0644614, 0.1335363, -0.1791619, 0.1781299
3: -0.0143578, 0.0519093, -0.0182992, 0.0664772, -0.0808350, 0.0702085
4: -0.0450331, 0.0524986, -0.0612277, 0.0739916, -0.1190247, 0.1137263
5: -0.0297000, 0.0411202, -0.0473275, 0.0583539, -0.0880539, 0.0884477
6: -0.0967258, 0.0585561, -0.1166612, 0.0767701, -0.1734958, 0.1752173
7: 0.8588650, 1.0062377, 0.8323801, 1.0163677, -0.1575027, 0.1738577
8: -0.0532692, 0.0900260, -0.0602355, 0.1125589, -0.1628099, 0.1502616
9: -0.0746245, 0.0455912, -0.0886095, 0.0772001, -0.1518246, 0.1342007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0647890, 0.0189602, -0.0649075, 0.0210067, -0.0857958, 0.0838678
1: -0.0583970, 0.0435304, -0.0600260, 0.0428240, -0.1012210, 0.1035564
2: -0.0636264, 0.1322288, -0.0644614, 0.1335363, -0.1971627, 0.1966902
3: -0.0141307, 0.0672634, -0.0182992, 0.0664772, -0.0806079, 0.0855625
4: -0.0568053, 0.0717332, -0.0612277, 0.0739916, -0.1307968, 0.1329609
5: -0.0467627, 0.0550124, -0.0473275, 0.0583539, -0.1051165, 0.1023399
6: -0.1144638, 0.0754808, -0.1166612, 0.0767701, -0.1912339, 0.1921420
7: 0.8341800, 1.0155976, 0.8323801, 1.0163677, -0.1821877, 0.1832175
8: -0.0524813, 0.1077710, -0.0602355, 0.1125589, -0.1650402, 0.1680065
9: -0.0886746, 0.0676016, -0.0886095, 0.0772001, -0.1658747, 0.1562111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0356602, 0.0150436, -0.0820888, 0.0277531, -0.0634133, 0.0971324
1: -0.0336167, 0.0188196, -0.0771898, 0.0557019, -0.0893185, 0.0960095
2: -0.0314463, 0.0999831, -0.0831659, 0.1532310, -0.1846773, 0.1831490
3: -0.0133638, 0.0437791, -0.0327649, 0.0808078, -0.0941716, 0.0765440
4: -0.0360880, 0.0397449, -0.0774938, 0.0954588, -0.1315468, 0.1172388
5: -0.0197772, 0.0301322, -0.0647637, 0.0757121, -0.0954892, 0.0948960
6: -0.0837465, 0.0460974, -0.1365824, 0.0948313, -0.1785778, 0.1826798
7: 0.8783208, 0.9997565, 0.8061218, 1.0264577, -0.1481369, 0.1936346
8: -0.0498204, 0.0770647, -0.0595233, 0.1352944, -0.1789804, 0.1365880
9: -0.0641747, 0.0302929, -0.1024321, 0.1090098, -0.1731845, 0.1327250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0470289, 0.0167566, -0.0820888, 0.0277531, -0.0747820, 0.0988455
1: -0.0423048, 0.0293211, -0.0771898, 0.0557019, -0.0980067, 0.1065109
2: -0.0450884, 0.1131146, -0.0831659, 0.1532310, -0.1983195, 0.1962805
3: -0.0146469, 0.0514512, -0.0327649, 0.0808078, -0.0954547, 0.0842161
4: -0.0446818, 0.0519245, -0.0774938, 0.0954588, -0.1401406, 0.1294183
5: -0.0291908, 0.0407057, -0.0647637, 0.0757121, -0.1049029, 0.1054694
6: -0.0961964, 0.0580510, -0.1365824, 0.0948313, -0.1910277, 0.1946334
7: 0.8596016, 1.0059586, 0.8061218, 1.0264577, -0.1668561, 0.1998367
8: -0.0542724, 0.0894965, -0.0595233, 0.1352944, -0.1836607, 0.1490198
9: -0.0742052, 0.0449343, -0.1024321, 0.1090098, -0.1832151, 0.1473664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0565825, 0.0181467, -0.0656508, 0.0212986, -0.0778811, 0.0837975
1: -0.0517095, 0.0365842, -0.0607685, 0.0433811, -0.0950905, 0.0973527
2: -0.0553984, 0.1239934, -0.0652706, 0.1343882, -0.1897866, 0.1892640
3: -0.0156881, 0.0595335, -0.0189249, 0.0670971, -0.0827852, 0.0784585
4: -0.0533463, 0.0635899, -0.0619314, 0.0749202, -0.1282665, 0.1255213
5: -0.0388789, 0.0499432, -0.0480817, 0.0591048, -0.0979837, 0.0980249
6: -0.1070086, 0.0680188, -0.1175229, 0.0775514, -0.1845600, 0.1855417
7: 0.8451029, 1.0114785, 0.8312440, 1.0168042, -0.1717013, 0.1802346
8: -0.0578851, 0.1015428, -0.0623362, 0.1135424, -0.1694786, 0.1638790
9: -0.0819119, 0.0617872, -0.0892074, 0.0785762, -0.1604880, 0.1509947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0739569, 0.0245600, -0.0656508, 0.0212986, -0.0952555, 0.0902108
1: -0.0690661, 0.0496067, -0.0607685, 0.0433811, -0.1124472, 0.1103752
2: -0.0743131, 0.1439094, -0.0652706, 0.1343882, -0.2087013, 0.2091800
3: -0.0259182, 0.0740251, -0.0189249, 0.0670971, -0.0930153, 0.0929500
4: -0.0697950, 0.0852984, -0.0619314, 0.0749202, -0.1447152, 0.1472298
5: -0.0565112, 0.0674964, -0.0480817, 0.0591048, -0.1156160, 0.1155781
6: -0.1271536, 0.0862829, -0.1175229, 0.0775514, -0.2047050, 0.2038058
7: 0.8185500, 1.0216819, 0.8312440, 1.0168042, -0.1982542, 0.1904379
8: -0.0569428, 0.1245337, -0.0623362, 0.1135424, -0.1704852, 0.1868699
9: -0.0958898, 0.0939542, -0.0892074, 0.0785762, -0.1744660, 0.1831616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0427713, 0.0167634, -0.0828703, 0.0280600, -0.0708312, 0.0996337
1: -0.0384922, 0.0258950, -0.0779704, 0.0562876, -0.0947798, 0.1038654
2: -0.0406204, 0.1085058, -0.0840167, 0.1541268, -0.1947472, 0.1925226
3: -0.0146520, 0.0477663, -0.0334229, 0.0814597, -0.0961116, 0.0811892
4: -0.0417703, 0.0471483, -0.0782336, 0.0964351, -0.1382054, 0.1253819
5: -0.0249982, 0.0372560, -0.0655568, 0.0765016, -0.1014998, 0.1028128
6: -0.0917918, 0.0538779, -0.1374885, 0.0956528, -0.1874446, 0.1913663
7: 0.8657314, 1.0036635, 0.8049276, 1.0269166, -0.1611853, 0.1987360
8: -0.0542900, 0.0850901, -0.0615063, 0.1363285, -0.1841309, 0.1465965
9: -0.0707375, 0.0394688, -0.1030608, 0.1104565, -0.1811939, 0.1425296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0559901, 0.0184617, -0.0828703, 0.0280600, -0.0840500, 0.1013319
1: -0.0511175, 0.0361401, -0.0779704, 0.0562876, -0.1074051, 0.1141106
2: -0.0547533, 0.1233142, -0.0840167, 0.1541268, -0.2088801, 0.2073310
3: -0.0159239, 0.0590393, -0.0334229, 0.0814597, -0.0973836, 0.0924622
4: -0.0527853, 0.0628496, -0.0782336, 0.0964351, -0.1492205, 0.1410832
5: -0.0382776, 0.0493446, -0.0655568, 0.0765016, -0.1147792, 0.1149013
6: -0.1063216, 0.0673960, -0.1374885, 0.0956528, -0.2019744, 0.2048844
7: 0.8460085, 1.0111308, 0.8049276, 1.0269166, -0.1809081, 0.2062032
8: -0.0587035, 0.1007587, -0.0615063, 0.1363285, -0.1903862, 0.1622650
9: -0.0814353, 0.0606903, -0.1030608, 0.1104565, -0.1918918, 0.1637511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0555254, 0.0173227, -0.0526319, 0.0178920, -0.0734174, 0.0699547
1: -0.0506534, 0.0357918, -0.0477629, 0.0336231, -0.0842765, 0.0835547
2: -0.0542474, 0.1227816, -0.0510974, 0.1194649, -0.1737123, 0.1738790
3: -0.0147985, 0.0586518, -0.0154973, 0.0562384, -0.0710369, 0.0741491
4: -0.0523453, 0.0622690, -0.0496061, 0.0586538, -0.1109991, 0.1118751
5: -0.0378061, 0.0488751, -0.0348697, 0.0459519, -0.0837580, 0.0837449
6: -0.1057829, 0.0669075, -0.1024280, 0.0638659, -0.1696488, 0.1693355
7: 0.8467187, 1.0108578, 0.8511406, 1.0091587, -0.1624401, 0.1597172
8: -0.0547986, 0.1001438, -0.0572231, 0.0963150, -0.1505035, 0.1551578
9: -0.0810614, 0.0598299, -0.0787336, 0.0544731, -0.1355345, 0.1385635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0741686, 0.0246432, -0.0526319, 0.0178920, -0.0920606, 0.0772752
1: -0.0692777, 0.0497655, -0.0477629, 0.0336231, -0.1029008, 0.0975284
2: -0.0745436, 0.1441523, -0.0510974, 0.1194649, -0.1940085, 0.1952497
3: -0.0260966, 0.0742018, -0.0154973, 0.0562384, -0.0823350, 0.0896991
4: -0.0699956, 0.0855629, -0.0496061, 0.0586538, -0.1286494, 0.1351690
5: -0.0567261, 0.0677103, -0.0348697, 0.0459519, -0.1026780, 0.1025801
6: -0.1273992, 0.0865056, -0.1024280, 0.0638659, -0.1912651, 0.1889336
7: 0.8182260, 1.0218064, 0.8511406, 1.0091587, -0.1909327, 0.1706657
8: -0.0539877, 0.1248140, -0.0572231, 0.0963150, -0.1503027, 0.1797006
9: -0.0960602, 0.0943464, -0.0787336, 0.0544731, -0.1505333, 0.1730800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0430042, 0.0163419, -0.0694199, 0.0227785, -0.0657827, 0.0857617
1: -0.0386638, 0.0260986, -0.0645337, 0.0462061, -0.0848699, 0.0906323
2: -0.0408845, 0.1087798, -0.0693737, 0.1387087, -0.1795932, 0.1781535
3: -0.0143362, 0.0478811, -0.0220983, 0.0702409, -0.0845771, 0.0699795
4: -0.0419338, 0.0474322, -0.0654997, 0.0796295, -0.1215633, 0.1129319
5: -0.0252113, 0.0374611, -0.0519068, 0.0629127, -0.0881240, 0.0893679
6: -0.0920536, 0.0541018, -0.1218931, 0.0815135, -0.1735671, 0.1759949
7: 0.8653668, 1.0037758, 0.8254838, 1.0190176, -0.1536508, 0.1782920
8: -0.0531944, 0.0853521, -0.0563986, 0.1185300, -0.1658208, 0.1417507
9: -0.0709264, 0.0397937, -0.0922397, 0.0855543, -0.1564806, 0.1320335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0555025, 0.0179202, -0.0694199, 0.0227785, -0.0782809, 0.0873401
1: -0.0506305, 0.0357746, -0.0645337, 0.0462061, -0.0968366, 0.1003083
2: -0.0542224, 0.1227553, -0.0693737, 0.1387087, -0.1929311, 0.1921291
3: -0.0155184, 0.0586326, -0.0220983, 0.0702409, -0.0857593, 0.0807310
4: -0.0523237, 0.0622403, -0.0654997, 0.0796295, -0.1319532, 0.1277400
5: -0.0377828, 0.0488520, -0.0519068, 0.0629127, -0.1006955, 0.1007588
6: -0.1057563, 0.0668834, -0.1218931, 0.0815135, -0.1872698, 0.1887765
7: 0.8467536, 1.0108443, 0.8254838, 1.0190176, -0.1722640, 0.1853606
8: -0.0572963, 0.1001135, -0.0563986, 0.1185300, -0.1732673, 0.1565121
9: -0.0810429, 0.0597876, -0.0922397, 0.0855543, -0.1665972, 0.1520273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0645275, 0.0208575, -0.0533342, 0.0186984, -0.0832259, 0.0741917
1: -0.0596463, 0.0425391, -0.0484645, 0.0341494, -0.0937957, 0.0910036
2: -0.0640476, 0.1331007, -0.0518620, 0.1202699, -0.1843175, 0.1849627
3: -0.0179792, 0.0661603, -0.0161013, 0.0568242, -0.0748034, 0.0822616
4: -0.0608679, 0.0735168, -0.0502709, 0.0595312, -0.1203991, 0.1237877
5: -0.0469418, 0.0579699, -0.0355824, 0.0466614, -0.0936032, 0.0935523
6: -0.1162205, 0.0763706, -0.1032422, 0.0646041, -0.1808247, 0.1796128
7: 0.8329608, 1.0161443, 0.8500674, 1.0095711, -0.1766103, 0.1660769
8: -0.0593606, 0.1120560, -0.0593190, 0.0972444, -0.1566050, 0.1691209
9: -0.0883038, 0.0764965, -0.0792986, 0.0557732, -0.1440769, 0.1557951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0817171, 0.0276071, -0.0533342, 0.0186984, -0.1004155, 0.0809414
1: -0.0768185, 0.0554232, -0.0484645, 0.0341494, -0.1109679, 0.1038876
2: -0.0827612, 0.1528048, -0.0518620, 0.1202699, -0.2030310, 0.2046668
3: -0.0324519, 0.0804978, -0.0161013, 0.0568242, -0.0892761, 0.0965991
4: -0.0771418, 0.0949943, -0.0502709, 0.0595312, -0.1366730, 0.1452652
5: -0.0643865, 0.0753365, -0.0355824, 0.0466614, -0.1110479, 0.1109189
6: -0.1361514, 0.0944405, -0.1032422, 0.0646041, -0.2007555, 0.1976827
7: 0.8066901, 1.0262394, 0.8500674, 1.0095711, -0.2028810, 0.1761720
8: -0.0582217, 0.1348025, -0.0593190, 0.0972444, -0.1554661, 0.1914582
9: -0.1021330, 0.1083214, -0.0792986, 0.0557732, -0.1579061, 0.1876200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0519553, 0.0179822, -0.0701649, 0.0230711, -0.0750263, 0.0881471
1: -0.0470869, 0.0331159, -0.0652780, 0.0467645, -0.0938514, 0.0983939
2: -0.0503608, 0.1186892, -0.0701849, 0.1395627, -0.1899236, 0.1888741
3: -0.0155648, 0.0556740, -0.0227256, 0.0708623, -0.0864271, 0.0783996
4: -0.0489655, 0.0578084, -0.0662050, 0.0805604, -0.1295258, 0.1240134
5: -0.0341830, 0.0452683, -0.0526628, 0.0636654, -0.0978484, 0.0979311
6: -0.1016435, 0.0631546, -0.1227570, 0.0822967, -0.1839402, 0.1859115
7: 0.8521746, 1.0087614, 0.8243452, 1.0194550, -0.1672803, 0.1844162
8: -0.0574574, 0.0954197, -0.0584282, 0.1195159, -0.1714795, 0.1538479
9: -0.0781892, 0.0532203, -0.0928392, 0.0869336, -0.1651228, 0.1460594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0644373, 0.0208221, -0.0701649, 0.0230711, -0.0875084, 0.0909870
1: -0.0595562, 0.0424715, -0.0652780, 0.0467645, -0.1063208, 0.1077495
2: -0.0639495, 0.1329973, -0.0701849, 0.1395627, -0.2035122, 0.2031822
3: -0.0179033, 0.0660851, -0.0227256, 0.0708623, -0.0887655, 0.0888107
4: -0.0607825, 0.0734040, -0.0662050, 0.0805604, -0.1413429, 0.1396090
5: -0.0468503, 0.0578788, -0.0526628, 0.0636654, -0.1105157, 0.1105416
6: -0.1161160, 0.0762758, -0.1227570, 0.0822967, -0.1984127, 0.1990328
7: 0.8330986, 1.0160918, 0.8243452, 1.0194550, -0.1863563, 0.1917466
8: -0.0617287, 0.1119367, -0.0584282, 0.1195159, -0.1802103, 0.1703649
9: -0.0882312, 0.0763295, -0.0928392, 0.0869336, -0.1751648, 0.1691687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0560204, 0.0175849, -0.0652201, 0.0211294, -0.0771498, 0.0828049
1: -0.0511479, 0.0361627, -0.0603382, 0.0430583, -0.0942062, 0.0965009
2: -0.0547863, 0.1233490, -0.0648015, 0.1338945, -0.1886808, 0.1881505
3: -0.0152672, 0.0590646, -0.0185623, 0.0667380, -0.0820052, 0.0776270
4: -0.0528140, 0.0628875, -0.0615237, 0.0743820, -0.1271960, 0.1244112
5: -0.0383084, 0.0493752, -0.0476447, 0.0586696, -0.0969780, 0.0970198
6: -0.1063568, 0.0674278, -0.1170235, 0.0770987, -0.1834554, 0.1844513
7: 0.8459622, 1.0111486, 0.8319024, 1.0165510, -0.1705888, 0.1792462
8: -0.0564248, 0.1007988, -0.0613846, 0.1129725, -0.1680288, 0.1621835
9: -0.0814596, 0.0607464, -0.0888610, 0.0777787, -0.1592384, 0.1496074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0746582, 0.0248354, -0.0652201, 0.0211294, -0.0957876, 0.0900555
1: -0.0697668, 0.0501324, -0.0603382, 0.0430583, -0.1128250, 0.1104706
2: -0.0750765, 0.1447134, -0.0648015, 0.1338945, -0.2089711, 0.2095149
3: -0.0265087, 0.0746102, -0.0185623, 0.0667380, -0.0932467, 0.0931725
4: -0.0704590, 0.0861745, -0.0615237, 0.0743820, -0.1448410, 0.1476982
5: -0.0572229, 0.0682049, -0.0476447, 0.0586696, -0.1158925, 0.1158496
6: -0.1279668, 0.0870201, -0.1170235, 0.0770987, -0.2050655, 0.2040437
7: 0.8174782, 1.0220940, 0.8319024, 1.0165510, -0.1990728, 0.1901916
8: -0.0556930, 0.1254617, -0.0613846, 0.1129725, -0.1686655, 0.1868463
9: -0.0964540, 0.0952526, -0.0888610, 0.0777787, -0.1742328, 0.1841136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0430042, 0.0163419, -0.0824200, 0.0278831, -0.0708873, 0.0987618
1: -0.0386638, 0.0260986, -0.0775207, 0.0559501, -0.0946139, 0.1036193
2: -0.0408845, 0.1087798, -0.0835264, 0.1536107, -0.1944952, 0.1923062
3: -0.0143362, 0.0478811, -0.0330437, 0.0810841, -0.0954203, 0.0809249
4: -0.0419338, 0.0474322, -0.0778073, 0.0958726, -0.1378064, 0.1252395
5: -0.0252113, 0.0374611, -0.0651000, 0.0760466, -0.1012579, 0.1025610
6: -0.0920536, 0.0541018, -0.1369663, 0.0951795, -0.1872331, 0.1910681
7: 0.8653668, 1.0037758, 0.8056159, 1.0266521, -0.1612853, 0.1981599
8: -0.0531944, 0.0853521, -0.0607237, 0.1357327, -0.1827430, 0.1460758
9: -0.0709264, 0.0397937, -0.1026985, 0.1096229, -0.1805492, 0.1424922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0555025, 0.0179202, -0.0824200, 0.0278831, -0.0833856, 0.1003402
1: -0.0506305, 0.0357746, -0.0775207, 0.0559501, -0.1065806, 0.1132953
2: -0.0542224, 0.1227553, -0.0835264, 0.1536107, -0.2078331, 0.2062818
3: -0.0155184, 0.0586326, -0.0330437, 0.0810841, -0.0966025, 0.0916764
4: -0.0523237, 0.0622403, -0.0778073, 0.0958726, -0.1481963, 0.1400476
5: -0.0377828, 0.0488520, -0.0651000, 0.0760466, -0.1138294, 0.1139520
6: -0.1057563, 0.0668834, -0.1369663, 0.0951795, -0.2009358, 0.2038497
7: 0.8467536, 1.0108443, 0.8056159, 1.0266521, -0.1798985, 0.2052284
8: -0.0572963, 0.1001135, -0.0607237, 0.1357327, -0.1887857, 0.1608372
9: -0.0810429, 0.0597876, -0.1026985, 0.1096229, -0.1906658, 0.1624860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0650108, 0.0210473, -0.0659610, 0.0214204, -0.0864311, 0.0870082
1: -0.0601291, 0.0429014, -0.0610783, 0.0436136, -0.1037427, 0.1039797
2: -0.0645737, 0.1336546, -0.0656082, 0.1347438, -0.1993175, 0.1992628
3: -0.0183861, 0.0665633, -0.0191861, 0.0673559, -0.0857420, 0.0857494
4: -0.0613255, 0.0741206, -0.0622250, 0.0753077, -0.1366332, 0.1363456
5: -0.0474323, 0.0584581, -0.0483965, 0.0594181, -0.1068504, 0.1068546
6: -0.1167808, 0.0768786, -0.1178826, 0.0778775, -0.1946583, 0.1947612
7: 0.8322222, 1.0164282, 0.8307701, 1.0169865, -0.1847643, 0.1856581
8: -0.0610462, 0.1126955, -0.0635223, 0.1139529, -0.1749991, 0.1762178
9: -0.0886925, 0.0773913, -0.0894569, 0.0791504, -0.1678429, 0.1668482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0822103, 0.0278008, -0.0659610, 0.0214204, -0.1036307, 0.0937618
1: -0.0773112, 0.0557929, -0.0610783, 0.0436136, -0.1209247, 0.1168713
2: -0.0832983, 0.1533702, -0.0656082, 0.1347438, -0.2180420, 0.2189785
3: -0.0328672, 0.0809092, -0.0191861, 0.0673559, -0.1002230, 0.1000953
4: -0.0776087, 0.0956105, -0.0622250, 0.0753077, -0.1529164, 0.1578355
5: -0.0648870, 0.0758348, -0.0483965, 0.0594181, -0.1243052, 0.1242313
6: -0.1367232, 0.0949589, -0.1178826, 0.0778775, -0.2146007, 0.2128415
7: 0.8059363, 1.0265290, 0.8307701, 1.0169865, -0.2110502, 0.1957588
8: -0.0599421, 0.1354552, -0.0635223, 0.1139529, -0.1738950, 0.1989774
9: -0.1025298, 0.1092346, -0.0894569, 0.0791504, -0.1816802, 0.1986915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0519553, 0.0179822, -0.0832076, 0.0281924, -0.0801477, 0.1011898
1: -0.0470869, 0.0331159, -0.0783074, 0.0565404, -0.1036273, 0.1114233
2: -0.0503608, 0.1186892, -0.0843839, 0.1545134, -0.2048743, 0.2030731
3: -0.0155648, 0.0556740, -0.0337068, 0.0817410, -0.0973058, 0.0893808
4: -0.0489655, 0.0578084, -0.0785529, 0.0968566, -0.1458221, 0.1363613
5: -0.0341830, 0.0452683, -0.0658991, 0.0768423, -0.1110253, 0.1111674
6: -0.1016435, 0.0631546, -0.1378795, 0.0960074, -0.1976508, 0.2010340
7: 0.8521746, 1.0087614, 0.8044122, 1.0271146, -0.1749400, 0.2043492
8: -0.0574574, 0.0954197, -0.0626680, 0.1367748, -0.1885159, 0.1580877
9: -0.0781892, 0.0532203, -0.1033322, 0.1110810, -0.1892702, 0.1565525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0644373, 0.0208221, -0.0832076, 0.0281924, -0.0926297, 0.1040297
1: -0.0595562, 0.0424715, -0.0783074, 0.0565404, -0.1160967, 0.1207789
2: -0.0639495, 0.1329973, -0.0843839, 0.1545134, -0.2184629, 0.2173811
3: -0.0179033, 0.0660851, -0.0337068, 0.0817410, -0.0996443, 0.0997919
4: -0.0607825, 0.0734040, -0.0785529, 0.0968566, -0.1576391, 0.1519569
5: -0.0468503, 0.0578788, -0.0658991, 0.0768423, -0.1236926, 0.1237779
6: -0.1161160, 0.0762758, -0.1378795, 0.0960074, -0.2121233, 0.2141553
7: 0.8330986, 1.0160918, 0.8044122, 1.0271146, -0.1940160, 0.2116796
8: -0.0617287, 0.1119367, -0.0626680, 0.1367748, -0.1963307, 0.1746048
9: -0.0882312, 0.0763295, -0.1033322, 0.1110810, -0.1993122, 0.1796617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.71 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0470737, 0.0156740, -0.0425346, 0.0157907, -0.0628643, 0.0582086
1: -0.0423454, 0.0293570, -0.0383179, 0.0256880, -0.0680334, 0.0676749
2: -0.0451352, 0.1131628, -0.0403522, 0.1082275, -0.1533627, 0.1535150
3: -0.0138359, 0.0514910, -0.0139233, 0.0476497, -0.0614857, 0.0654144
4: -0.0447125, 0.0519745, -0.0416041, 0.0468599, -0.0915723, 0.0935786
5: -0.0292351, 0.0407417, -0.0247817, 0.0370477, -0.0662828, 0.0655234
6: -0.0962425, 0.0580950, -0.0915258, 0.0536503, -0.1498928, 0.1496208
7: 0.8595375, 1.0059826, 0.8661014, 1.0035493, -0.1440119, 0.1398812
8: -0.0514586, 0.0895426, -0.0517619, 0.0848241, -0.1340920, 0.1375409
9: -0.0742418, 0.0449915, -0.0705455, 0.0391388, -0.1133806, 0.1155370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0470737, 0.0156740, -0.0516358, 0.0170662, -0.0641399, 0.0673098
1: -0.0423454, 0.0293570, -0.0467677, 0.0328764, -0.0752218, 0.0761247
2: -0.0451352, 0.1131628, -0.0500130, 0.1183230, -0.1634582, 0.1631758
3: -0.0138359, 0.0514910, -0.0148788, 0.0554075, -0.0692435, 0.0663698
4: -0.0447125, 0.0519745, -0.0486630, 0.0574092, -0.1021216, 0.1006375
5: -0.0292351, 0.0407417, -0.0338588, 0.0449455, -0.0741807, 0.0746005
6: -0.0962425, 0.0580950, -0.1012731, 0.0628187, -0.1590612, 0.1593681
7: 0.8595375, 1.0059826, 0.8526630, 1.0085737, -0.1490362, 0.1533196
8: -0.0514586, 0.0895426, -0.0550770, 0.0949969, -0.1441992, 0.1430632
9: -0.0742418, 0.0449915, -0.0779322, 0.0526288, -0.1268706, 0.1229237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0643372, 0.0188382, -0.0425346, 0.0157907, -0.0801279, 0.0613729
1: -0.0579876, 0.0431690, -0.0383179, 0.0256880, -0.0836756, 0.0814869
2: -0.0631548, 0.1317425, -0.0403522, 0.1082275, -0.1713823, 0.1720947
3: -0.0136197, 0.0668611, -0.0139233, 0.0476497, -0.0612695, 0.0807845
4: -0.0564969, 0.0712292, -0.0416041, 0.0468599, -0.1033568, 0.1128333
5: -0.0463156, 0.0546484, -0.0247817, 0.0370477, -0.0833633, 0.0794301
6: -0.1139990, 0.0750375, -0.0915258, 0.0536503, -0.1676494, 0.1665633
7: 0.8348267, 1.0153522, 0.8661014, 1.0035493, -0.1687226, 0.1492508
8: -0.0507085, 0.1073061, -0.0517619, 0.0848241, -0.1355326, 0.1553574
9: -0.0883066, 0.0670248, -0.0705455, 0.0391388, -0.1274454, 0.1375704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0643372, 0.0188382, -0.0516358, 0.0170662, -0.0814035, 0.0704741
1: -0.0579876, 0.0431690, -0.0467677, 0.0328764, -0.0908640, 0.0899367
2: -0.0631548, 0.1317425, -0.0500130, 0.1183230, -0.1814778, 0.1817555
3: -0.0136197, 0.0668611, -0.0148788, 0.0554075, -0.0690273, 0.0817399
4: -0.0564969, 0.0712292, -0.0486630, 0.0574092, -0.1139061, 0.1198922
5: -0.0463156, 0.0546484, -0.0338588, 0.0449455, -0.0912611, 0.0885072
6: -0.1139990, 0.0750375, -0.1012731, 0.0628187, -0.1768177, 0.1763105
7: 0.8348267, 1.0153522, 0.8526630, 1.0085737, -0.1737469, 0.1626892
8: -0.0507085, 0.1073061, -0.0550770, 0.0949969, -0.1457054, 0.1608797
9: -0.0883066, 0.0670248, -0.0779322, 0.0526288, -0.1409354, 0.1449570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0356602, 0.0150436, -0.0589047, 0.0186496, -0.0543098, 0.0739483
1: -0.0336167, 0.0188196, -0.0540292, 0.0383246, -0.0719413, 0.0728489
2: -0.0314463, 0.0999831, -0.0579262, 0.1266552, -0.1581015, 0.1579092
3: -0.0133638, 0.0437791, -0.0136850, 0.0614703, -0.0748341, 0.0574641
4: -0.0360880, 0.0397449, -0.0555446, 0.0664913, -0.1025793, 0.0952895
5: -0.0197772, 0.0301322, -0.0412355, 0.0522892, -0.0720664, 0.0713677
6: -0.0837465, 0.0460974, -0.1097010, 0.0704598, -0.1542063, 0.1557984
7: 0.8783208, 0.9997565, 0.8415542, 1.0128422, -0.1345214, 0.1582023
8: -0.0498204, 0.0770647, -0.0509348, 0.1046155, -0.1488398, 0.1279995
9: -0.0641747, 0.0302929, -0.0837801, 0.0660863, -0.1302610, 0.1140729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0356602, 0.0150436, -0.0684047, 0.0223800, -0.0580401, 0.0834483
1: -0.0336167, 0.0188196, -0.0635196, 0.0454453, -0.0790619, 0.0823393
2: -0.0314463, 0.0999831, -0.0682687, 0.1375451, -0.1689914, 0.1682518
3: -0.0133638, 0.0437791, -0.0212437, 0.0693943, -0.0827581, 0.0650228
4: -0.0360880, 0.0397449, -0.0645387, 0.0783611, -0.1144492, 0.1042836
5: -0.0197772, 0.0301322, -0.0508766, 0.0618871, -0.0816643, 0.0810088
6: -0.0837465, 0.0460974, -0.1207161, 0.0804465, -0.1641930, 0.1668136
7: 0.8783208, 0.9997565, 0.8270352, 1.0184215, -0.1401008, 0.1727212
8: -0.0498204, 0.0770647, -0.0542353, 0.1171868, -0.1612455, 0.1313000
9: -0.0641747, 0.0302929, -0.0914230, 0.0836750, -0.1478496, 0.1217159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0470289, 0.0167566, -0.0589047, 0.0186496, -0.0656785, 0.0756613
1: -0.0423048, 0.0293211, -0.0540292, 0.0383246, -0.0806295, 0.0833503
2: -0.0450884, 0.1131146, -0.0579262, 0.1266552, -0.1717436, 0.1710407
3: -0.0146469, 0.0514512, -0.0136850, 0.0614703, -0.0761172, 0.0651361
4: -0.0446818, 0.0519245, -0.0555446, 0.0664913, -0.1111731, 0.1074691
5: -0.0291908, 0.0407057, -0.0412355, 0.0522892, -0.0814800, 0.0819412
6: -0.0961964, 0.0580510, -0.1097010, 0.0704598, -0.1666562, 0.1677520
7: 0.8596016, 1.0059586, 0.8415542, 1.0128422, -0.1532406, 0.1644044
8: -0.0542724, 0.0894965, -0.0509348, 0.1046155, -0.1552777, 0.1404313
9: -0.0742052, 0.0449343, -0.0837801, 0.0660863, -0.1402915, 0.1287144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0470289, 0.0167566, -0.0684047, 0.0223800, -0.0694089, 0.0851614
1: -0.0423048, 0.0293211, -0.0635196, 0.0454453, -0.0877501, 0.0928407
2: -0.0450884, 0.1131146, -0.0682687, 0.1375451, -0.1826335, 0.1813833
3: -0.0146469, 0.0514512, -0.0212437, 0.0693943, -0.0840411, 0.0726949
4: -0.0446818, 0.0519245, -0.0645387, 0.0783611, -0.1230430, 0.1164632
5: -0.0291908, 0.0407057, -0.0508766, 0.0618871, -0.0910779, 0.0915823
6: -0.0961964, 0.0580510, -0.1207161, 0.0804465, -0.1766429, 0.1787671
7: 0.8596016, 1.0059586, 0.8270352, 1.0184215, -0.1588199, 0.1789233
8: -0.0542724, 0.0894965, -0.0542353, 0.1171868, -0.1676834, 0.1437318
9: -0.0742052, 0.0449343, -0.0914230, 0.0836750, -0.1578802, 0.1363573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0560730, 0.0175377, -0.0431453, 0.0165977, -0.0726707, 0.0606830
1: -0.0512004, 0.0362022, -0.0387860, 0.0262140, -0.0774144, 0.0749882
2: -0.0548435, 0.1234092, -0.0410348, 0.1089349, -0.1637785, 0.1644440
3: -0.0151620, 0.0591084, -0.0145278, 0.0479935, -0.0631555, 0.0736362
4: -0.0528637, 0.0629532, -0.0420308, 0.0475930, -0.1004568, 0.1049840
5: -0.0383618, 0.0494284, -0.0253484, 0.0375772, -0.0759390, 0.0747768
6: -0.1064177, 0.0674830, -0.0922019, 0.0542396, -0.1606573, 0.1596850
7: 0.8458818, 1.0111794, 0.8651605, 1.0038503, -0.1579685, 0.1460190
8: -0.0560597, 0.1008684, -0.0538593, 0.0855004, -0.1404369, 0.1504695
9: -0.0815019, 0.0608437, -0.0710412, 0.0399777, -0.1214797, 0.1318849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0560730, 0.0175377, -0.0523474, 0.0178621, -0.0739350, 0.0698851
1: -0.0512004, 0.0362022, -0.0474785, 0.0334098, -0.0846102, 0.0836807
2: -0.0548435, 0.1234092, -0.0507876, 0.1191386, -0.1739821, 0.1741968
3: -0.0151620, 0.0591084, -0.0154749, 0.0560009, -0.0711629, 0.0745833
4: -0.0528637, 0.0629532, -0.0493366, 0.0582982, -0.1111619, 0.1122898
5: -0.0383618, 0.0494284, -0.0345808, 0.0456644, -0.0840262, 0.0840091
6: -0.1064177, 0.0674830, -0.1020980, 0.0635667, -0.1699844, 0.1695811
7: 0.8458818, 1.0111794, 0.8515756, 1.0089915, -0.1631097, 0.1596038
8: -0.0560597, 0.1008684, -0.0571453, 0.0959384, -0.1508758, 0.1562183
9: -0.0815019, 0.0608437, -0.0785046, 0.0539461, -0.1354480, 0.1393483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0734499, 0.0243610, -0.0431453, 0.0165977, -0.0900476, 0.0675063
1: -0.0685597, 0.0492267, -0.0387860, 0.0262140, -0.0947736, 0.0880127
2: -0.0737611, 0.1433283, -0.0410348, 0.1089349, -0.1826960, 0.1843631
3: -0.0254914, 0.0736023, -0.0145278, 0.0479935, -0.0734849, 0.0881301
4: -0.0693150, 0.0846649, -0.0420308, 0.0475930, -0.1169081, 0.1266956
5: -0.0559966, 0.0669842, -0.0253484, 0.0375772, -0.0935738, 0.0923326
6: -0.1265658, 0.0857500, -0.0922019, 0.0542396, -0.1808054, 0.1779519
7: 0.8193248, 1.0213845, 0.8651605, 1.0038503, -0.1845255, 0.1562240
8: -0.0551789, 0.1238629, -0.0538593, 0.0855004, -0.1406794, 0.1731683
9: -0.0954819, 0.0930156, -0.0710412, 0.0399777, -0.1354597, 0.1640568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0734499, 0.0243610, -0.0523474, 0.0178621, -0.0913120, 0.0767083
1: -0.0685597, 0.0492267, -0.0474785, 0.0334098, -0.1019694, 0.0967052
2: -0.0737611, 0.1433283, -0.0507876, 0.1191386, -0.1928997, 0.1941159
3: -0.0254914, 0.0736023, -0.0154749, 0.0560009, -0.0814923, 0.0890772
4: -0.0693150, 0.0846649, -0.0493366, 0.0582982, -0.1276132, 0.1340015
5: -0.0559966, 0.0669842, -0.0345808, 0.0456644, -0.1016610, 0.1015650
6: -0.1265658, 0.0857500, -0.1020980, 0.0635667, -0.1901325, 0.1878480
7: 0.8193248, 1.0213845, 0.8515756, 1.0089915, -0.1896667, 0.1698089
8: -0.0551789, 0.1238629, -0.0571453, 0.0959384, -0.1511174, 0.1789170
9: -0.0954819, 0.0930156, -0.0785046, 0.0539461, -0.1494280, 0.1715202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0427713, 0.0167634, -0.0596262, 0.0189330, -0.0617042, 0.0763896
1: -0.0384922, 0.0258950, -0.0547500, 0.0388654, -0.0773576, 0.0806450
2: -0.0406204, 0.1085058, -0.0587118, 0.1274822, -0.1681026, 0.1672177
3: -0.0146520, 0.0477663, -0.0142801, 0.0620721, -0.0767241, 0.0620465
4: -0.0417703, 0.0471483, -0.0562277, 0.0673928, -0.1091630, 0.1033760
5: -0.0249982, 0.0372560, -0.0419677, 0.0530182, -0.0780164, 0.0792237
6: -0.0917918, 0.0538779, -0.1105376, 0.0712183, -0.1630101, 0.1644155
7: 0.8657314, 1.0036635, 0.8404516, 1.0132661, -0.1475347, 0.1632119
8: -0.0542900, 0.0850901, -0.0529999, 0.1055703, -0.1537762, 0.1380900
9: -0.0707375, 0.0394688, -0.0843605, 0.0674222, -0.1381596, 0.1238293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0427713, 0.0167634, -0.0691534, 0.0226739, -0.0654452, 0.0859169
1: -0.0384922, 0.0258950, -0.0642676, 0.0460064, -0.0844986, 0.0901626
2: -0.0406204, 0.1085058, -0.0690837, 0.1384033, -0.1790237, 0.1775896
3: -0.0146520, 0.0477663, -0.0218740, 0.0700187, -0.0846706, 0.0696403
4: -0.0417703, 0.0471483, -0.0652474, 0.0792966, -0.1210669, 0.1123957
5: -0.0249982, 0.0372560, -0.0516364, 0.0626435, -0.0876417, 0.0888924
6: -0.0917918, 0.0538779, -0.1215842, 0.0812334, -0.1730252, 0.1754621
7: 0.8657314, 1.0036635, 0.8258910, 1.0188612, -0.1531298, 0.1777725
8: -0.0542900, 0.0850901, -0.0561778, 0.1181774, -0.1662166, 0.1412679
9: -0.0707375, 0.0394688, -0.0920254, 0.0850610, -0.1557984, 0.1314942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0559901, 0.0184617, -0.0596262, 0.0189330, -0.0749230, 0.0780878
1: -0.0511175, 0.0361401, -0.0547500, 0.0388654, -0.0899830, 0.0908901
2: -0.0547533, 0.1233142, -0.0587118, 0.1274822, -0.1822355, 0.1820261
3: -0.0159239, 0.0590393, -0.0142801, 0.0620721, -0.0779961, 0.0733195
4: -0.0527853, 0.0628496, -0.0562277, 0.0673928, -0.1201781, 0.1190773
5: -0.0382776, 0.0493446, -0.0419677, 0.0530182, -0.0912958, 0.0913123
6: -0.1063216, 0.0673960, -0.1105376, 0.0712183, -0.1775399, 0.1779336
7: 0.8460085, 1.0111308, 0.8404516, 1.0132661, -0.1672575, 0.1706792
8: -0.0587035, 0.1007587, -0.0529999, 0.1055703, -0.1615072, 0.1537586
9: -0.0814353, 0.0606903, -0.0843605, 0.0674222, -0.1488574, 0.1450508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0559901, 0.0184617, -0.0691534, 0.0226739, -0.0786640, 0.0876151
1: -0.0511175, 0.0361401, -0.0642676, 0.0460064, -0.0971240, 0.1004077
2: -0.0547533, 0.1233142, -0.0690837, 0.1384033, -0.1931565, 0.1923979
3: -0.0159239, 0.0590393, -0.0218740, 0.0700187, -0.0859426, 0.0809133
4: -0.0527853, 0.0628496, -0.0652474, 0.0792966, -0.1320819, 0.1280970
5: -0.0382776, 0.0493446, -0.0516364, 0.0626435, -0.1009211, 0.1009810
6: -0.1063216, 0.0673960, -0.1215842, 0.0812334, -0.1875550, 0.1889801
7: 0.8460085, 1.0111308, 0.8258910, 1.0188612, -0.1728526, 0.1852398
8: -0.0587035, 0.1007587, -0.0561778, 0.1181774, -0.1739476, 0.1569365
9: -0.0814353, 0.0606903, -0.0920254, 0.0850610, -0.1664962, 0.1527157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0475435, 0.0163706, -0.0557359, 0.0174956, -0.0650391, 0.0721066
1: -0.0427711, 0.0297328, -0.0508637, 0.0359496, -0.0787207, 0.0805965
2: -0.0456256, 0.1136685, -0.0544765, 0.1230229, -0.1686485, 0.1681450
3: -0.0143578, 0.0519093, -0.0152004, 0.0588273, -0.0731851, 0.0671097
4: -0.0450331, 0.0524986, -0.0525447, 0.0625320, -0.1075651, 0.1050433
5: -0.0297000, 0.0411202, -0.0380197, 0.0490878, -0.0787878, 0.0791399
6: -0.0967258, 0.0585561, -0.1060269, 0.0671288, -0.1638546, 0.1645830
7: 0.8588650, 1.0062377, 0.8463969, 1.0109816, -0.1521165, 0.1598408
8: -0.0532692, 0.0900260, -0.0561928, 0.1004224, -0.1508013, 0.1441678
9: -0.0746245, 0.0455912, -0.0812308, 0.0602198, -0.1348443, 0.1268220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0475435, 0.0163706, -0.0641420, 0.0207062, -0.0682497, 0.0805127
1: -0.0427711, 0.0297328, -0.0592612, 0.0422503, -0.0850214, 0.0889941
2: -0.0456256, 0.1136685, -0.0636280, 0.1326587, -0.1782844, 0.1772965
3: -0.0143578, 0.0519093, -0.0176547, 0.0658388, -0.0801965, 0.0695640
4: -0.0450331, 0.0524986, -0.0605030, 0.0730351, -0.1180682, 0.1130016
5: -0.0297000, 0.0411202, -0.0465506, 0.0575805, -0.0872805, 0.0876708
6: -0.0967258, 0.0585561, -0.1157736, 0.0759654, -0.1726912, 0.1743297
7: 0.8588650, 1.0062377, 0.8335499, 1.0159180, -0.1570530, 0.1726879
8: -0.0532692, 0.0900260, -0.0592139, 0.1115461, -0.1617942, 0.1492399
9: -0.0746245, 0.0455912, -0.0879936, 0.0757828, -0.1504073, 0.1335848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0647890, 0.0189602, -0.0557359, 0.0174956, -0.0822846, 0.0746961
1: -0.0583970, 0.0435304, -0.0508637, 0.0359496, -0.0943466, 0.0943941
2: -0.0636264, 0.1322288, -0.0544765, 0.1230229, -0.1866493, 0.1867053
3: -0.0141307, 0.0672634, -0.0152004, 0.0588273, -0.0729580, 0.0824637
4: -0.0568053, 0.0717332, -0.0525447, 0.0625320, -0.1193373, 0.1242779
5: -0.0467627, 0.0550124, -0.0380197, 0.0490878, -0.0958504, 0.0930321
6: -0.1144638, 0.0754808, -0.1060269, 0.0671288, -0.1815927, 0.1815077
7: 0.8341800, 1.0155976, 0.8463969, 1.0109816, -0.1768016, 0.1692007
8: -0.0524813, 0.1077710, -0.0561928, 0.1004224, -0.1529037, 0.1621301
9: -0.0886746, 0.0676016, -0.0812308, 0.0602198, -0.1488944, 0.1488324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0647890, 0.0189602, -0.0641420, 0.0207062, -0.0854952, 0.0831023
1: -0.0583970, 0.0435304, -0.0592612, 0.0422503, -0.1006472, 0.1027917
2: -0.0636264, 0.1322288, -0.0636280, 0.1326587, -0.1962852, 0.1958568
3: -0.0141307, 0.0672634, -0.0176547, 0.0658388, -0.0799694, 0.0849180
4: -0.0568053, 0.0717332, -0.0605030, 0.0730351, -0.1298404, 0.1322362
5: -0.0467627, 0.0550124, -0.0465506, 0.0575805, -0.1043431, 0.1015630
6: -0.1144638, 0.0754808, -0.1157736, 0.0759654, -0.1904293, 0.1912544
7: 0.8341800, 1.0155976, 0.8335499, 1.0159180, -0.1817380, 0.1820477
8: -0.0524813, 0.1077710, -0.0592139, 0.1115461, -0.1640273, 0.1669849
9: -0.0886746, 0.0676016, -0.0879936, 0.0757828, -0.1644575, 0.1555952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0356602, 0.0150436, -0.0731253, 0.0242335, -0.0598937, 0.0881690
1: -0.0336167, 0.0188196, -0.0682354, 0.0489835, -0.0826001, 0.0870550
2: -0.0314463, 0.0999831, -0.0734078, 0.1429563, -0.1744026, 0.1733909
3: -0.0133638, 0.0437791, -0.0252181, 0.0733315, -0.0866953, 0.0689972
4: -0.0360880, 0.0397449, -0.0690077, 0.0842593, -0.1203473, 0.1087527
5: -0.0197772, 0.0301322, -0.0556672, 0.0666563, -0.0864335, 0.0857995
6: -0.0837465, 0.0460974, -0.1261894, 0.0854087, -0.1691552, 0.1722869
7: 0.8783208, 0.9997565, 0.8198209, 1.0211935, -0.1428728, 0.1799356
8: -0.0498204, 0.0770647, -0.0553077, 0.1234333, -0.1673672, 0.1323724
9: -0.0641747, 0.0302929, -0.0952208, 0.0924146, -0.1565893, 0.1255137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0356602, 0.0150436, -0.0813327, 0.0274561, -0.0631163, 0.0963763
1: -0.0336167, 0.0188196, -0.0764343, 0.0551350, -0.0887517, 0.0952540
2: -0.0314463, 0.0999831, -0.0823427, 0.1523641, -0.1838105, 0.1823258
3: -0.0133638, 0.0437791, -0.0321282, 0.0801770, -0.0935408, 0.0759073
4: -0.0360880, 0.0397449, -0.0767778, 0.0945139, -0.1306020, 0.1165227
5: -0.0197772, 0.0301322, -0.0639963, 0.0749481, -0.0947253, 0.0941285
6: -0.0837465, 0.0460974, -0.1357055, 0.0940363, -0.1777828, 0.1818030
7: 0.8783208, 0.9997565, 0.8072776, 1.0260134, -0.1476926, 0.1924788
8: -0.0498204, 0.0770647, -0.0584877, 0.1342938, -0.1779863, 0.1355525
9: -0.0641747, 0.0302929, -0.1018237, 0.1076096, -0.1717843, 0.1321166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0470289, 0.0167566, -0.0731253, 0.0242335, -0.0712624, 0.0898820
1: -0.0423048, 0.0293211, -0.0682354, 0.0489835, -0.0912883, 0.0975565
2: -0.0450884, 0.1131146, -0.0734078, 0.1429563, -0.1880447, 0.1865223
3: -0.0146469, 0.0514512, -0.0252181, 0.0733315, -0.0879784, 0.0766693
4: -0.0446818, 0.0519245, -0.0690077, 0.0842593, -0.1289411, 0.1209323
5: -0.0291908, 0.0407057, -0.0556672, 0.0666563, -0.0958471, 0.0963729
6: -0.0961964, 0.0580510, -0.1261894, 0.0854087, -0.1816051, 0.1842404
7: 0.8596016, 1.0059586, 0.8198209, 1.0211935, -0.1615919, 0.1861377
8: -0.0542724, 0.0894965, -0.0553077, 0.1234333, -0.1719364, 0.1448042
9: -0.0742052, 0.0449343, -0.0952208, 0.0924146, -0.1666198, 0.1401551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0470289, 0.0167566, -0.0813327, 0.0274561, -0.0744850, 0.0980893
1: -0.0423048, 0.0293211, -0.0764343, 0.0551350, -0.0974399, 0.1057554
2: -0.0450884, 0.1131146, -0.0823427, 0.1523641, -0.1974526, 0.1954572
3: -0.0146469, 0.0514512, -0.0321282, 0.0801770, -0.0948239, 0.0835794
4: -0.0446818, 0.0519245, -0.0767778, 0.0945139, -0.1391957, 0.1287023
5: -0.0291908, 0.0407057, -0.0639963, 0.0749481, -0.1041389, 0.1047020
6: -0.0961964, 0.0580510, -0.1357055, 0.0940363, -0.1902328, 0.1937565
7: 0.8596016, 1.0059586, 0.8072776, 1.0260134, -0.1664118, 0.1986809
8: -0.0542724, 0.0894965, -0.0584877, 0.1342938, -0.1826727, 0.1479843
9: -0.0742052, 0.0449343, -0.1018237, 0.1076096, -0.1818148, 0.1467580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0565825, 0.0181467, -0.0564640, 0.0183070, -0.0748895, 0.0746107
1: -0.0517095, 0.0365842, -0.0515910, 0.0364953, -0.0882047, 0.0881752
2: -0.0553984, 0.1239934, -0.0552693, 0.1238575, -0.1792559, 0.1792627
3: -0.0156881, 0.0595335, -0.0158081, 0.0594347, -0.0751228, 0.0753416
4: -0.0533463, 0.0635899, -0.0532340, 0.0634418, -0.1167880, 0.1168239
5: -0.0388789, 0.0499432, -0.0387587, 0.0498234, -0.0887023, 0.0887019
6: -0.1070086, 0.0680188, -0.1068711, 0.0678941, -0.1749028, 0.1748900
7: 0.8451029, 1.0114785, 0.8452842, 1.0114089, -0.1663060, 0.1661944
8: -0.0578851, 0.1015428, -0.0583015, 0.1013859, -0.1575178, 0.1573018
9: -0.0819119, 0.0617872, -0.0818166, 0.0615676, -0.1434795, 0.1436038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0565825, 0.0181467, -0.0648960, 0.0210022, -0.0775847, 0.0830427
1: -0.0517095, 0.0365842, -0.0600144, 0.0428153, -0.0945247, 0.0965986
2: -0.0553984, 0.1239934, -0.0644488, 0.1335230, -0.1889213, 0.1884422
3: -0.0156881, 0.0595335, -0.0182894, 0.0664675, -0.0821556, 0.0778229
4: -0.0533463, 0.0635899, -0.0612167, 0.0739771, -0.1273234, 0.1248066
5: -0.0388789, 0.0499432, -0.0473157, 0.0583422, -0.0972211, 0.0972589
6: -0.1070086, 0.0680188, -0.1166477, 0.0767579, -0.1837665, 0.1846665
7: 0.8451029, 1.0114785, 0.8323978, 1.0163609, -0.1712580, 0.1790808
8: -0.0578851, 0.1015428, -0.0613592, 0.1125436, -0.1684765, 0.1629021
9: -0.0819119, 0.0617872, -0.0886001, 0.0771787, -0.1590905, 0.1503874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0739569, 0.0245600, -0.0564640, 0.0183070, -0.0922639, 0.0810240
1: -0.0690661, 0.0496067, -0.0515910, 0.0364953, -0.1055614, 0.1011978
2: -0.0743131, 0.1439094, -0.0552693, 0.1238575, -0.1981706, 0.1991787
3: -0.0259182, 0.0740251, -0.0158081, 0.0594347, -0.0853529, 0.0898332
4: -0.0697950, 0.0852984, -0.0532340, 0.0634418, -0.1332368, 0.1385323
5: -0.0565112, 0.0674964, -0.0387587, 0.0498234, -0.1063346, 0.1062550
6: -0.1271536, 0.0862829, -0.1068711, 0.0678941, -0.1950477, 0.1931540
7: 0.8185500, 1.0216819, 0.8452842, 1.0114089, -0.1928589, 0.1763977
8: -0.0569428, 0.1245337, -0.0583015, 0.1013859, -0.1583287, 0.1801102
9: -0.0958898, 0.0939542, -0.0818166, 0.0615676, -0.1574574, 0.1757707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0739569, 0.0245600, -0.0648960, 0.0210022, -0.0949591, 0.0894560
1: -0.0690661, 0.0496067, -0.0600144, 0.0428153, -0.1118814, 0.1096211
2: -0.0743131, 0.1439094, -0.0644488, 0.1335230, -0.2078360, 0.2083582
3: -0.0259182, 0.0740251, -0.0182894, 0.0664675, -0.0923857, 0.0923145
4: -0.0697950, 0.0852984, -0.0612167, 0.0739771, -0.1437721, 0.1465151
5: -0.0565112, 0.0674964, -0.0473157, 0.0583422, -0.1148534, 0.1148120
6: -0.1271536, 0.0862829, -0.1166477, 0.0767579, -0.2039115, 0.2029306
7: 0.8185500, 1.0216819, 0.8323978, 1.0163609, -0.1978109, 0.1892841
8: -0.0569428, 0.1245337, -0.0613592, 0.1125436, -0.1694864, 0.1858930
9: -0.0958898, 0.0939542, -0.0886001, 0.0771787, -0.1730685, 0.1825543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0427713, 0.0167634, -0.0738770, 0.0245287, -0.0672999, 0.0906404
1: -0.0384922, 0.0258950, -0.0689863, 0.0495469, -0.0880391, 0.0948813
2: -0.0406204, 0.1085058, -0.0742261, 0.1438179, -0.1844383, 0.1827319
3: -0.0146520, 0.0477663, -0.0258510, 0.0739585, -0.0886105, 0.0736174
4: -0.0417703, 0.0471483, -0.0697194, 0.0851986, -0.1269688, 0.1168677
5: -0.0249982, 0.0372560, -0.0564301, 0.0674157, -0.0924139, 0.0936861
6: -0.0917918, 0.0538779, -0.1270611, 0.0861989, -0.1779907, 0.1809389
7: 0.8657314, 1.0036635, 0.8186721, 1.0216352, -0.1559038, 0.1849915
8: -0.0542900, 0.0850901, -0.0573318, 0.1244280, -0.1724458, 0.1424219
9: -0.0707375, 0.0394688, -0.0958256, 0.0938064, -0.1645438, 0.1352944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0427713, 0.0167634, -0.0821236, 0.0277667, -0.0705380, 0.0988870
1: -0.0384922, 0.0258950, -0.0772245, 0.0557279, -0.0942201, 0.1031195
2: -0.0406204, 0.1085058, -0.0832039, 0.1532708, -0.1938912, 0.1917097
3: -0.0146520, 0.0477663, -0.0327941, 0.0808368, -0.0954888, 0.0805604
4: -0.0417703, 0.0471483, -0.0775266, 0.0955021, -0.1372724, 0.1246749
5: -0.0249982, 0.0372560, -0.0647990, 0.0757471, -0.1007453, 0.1020550
6: -0.0917918, 0.0538779, -0.1366226, 0.0948678, -0.1866596, 0.1905005
7: 0.8657314, 1.0036635, 0.8060688, 1.0264779, -0.1607466, 0.1975948
8: -0.0542900, 0.0850901, -0.0604206, 0.1353404, -0.1831489, 0.1455107
9: -0.0707375, 0.0394688, -0.1024601, 0.1090740, -0.1798114, 0.1419289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0559901, 0.0184617, -0.0738770, 0.0245287, -0.0805187, 0.0923387
1: -0.0511175, 0.0361401, -0.0689863, 0.0495469, -0.1006644, 0.1051265
2: -0.0547533, 0.1233142, -0.0742261, 0.1438179, -0.1985712, 0.1975403
3: -0.0159239, 0.0590393, -0.0258510, 0.0739585, -0.0898825, 0.0848904
4: -0.0527853, 0.0628496, -0.0697194, 0.0851986, -0.1379839, 0.1325690
5: -0.0382776, 0.0493446, -0.0564301, 0.0674157, -0.1056934, 0.1057747
6: -0.1063216, 0.0673960, -0.1270611, 0.0861989, -0.1925206, 0.1944570
7: 0.8460085, 1.0111308, 0.8186721, 1.0216352, -0.1756266, 0.1924587
8: -0.0587035, 0.1007587, -0.0573318, 0.1244280, -0.1785691, 0.1580905
9: -0.0814353, 0.0606903, -0.0958256, 0.0938064, -0.1752416, 0.1565159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0559901, 0.0184617, -0.0821236, 0.0277667, -0.0837568, 0.1005852
1: -0.0511175, 0.0361401, -0.0772245, 0.0557279, -0.1068454, 0.1133646
2: -0.0547533, 0.1233142, -0.0832039, 0.1532708, -0.2080241, 0.2065181
3: -0.0159239, 0.0590393, -0.0327941, 0.0808368, -0.0967608, 0.0918334
4: -0.0527853, 0.0628496, -0.0775266, 0.0955021, -0.1482875, 0.1403762
5: -0.0382776, 0.0493446, -0.0647990, 0.0757471, -0.1140248, 0.1141436
6: -0.1063216, 0.0673960, -0.1366226, 0.0948678, -0.2011894, 0.2040186
7: 0.8460085, 1.0111308, 0.8060688, 1.0264779, -0.1804694, 0.2050620
8: -0.0587035, 0.1007587, -0.0604206, 0.1353404, -0.1894113, 0.1611792
9: -0.0814353, 0.0606903, -0.1024601, 0.1090740, -0.1905093, 0.1631504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0555254, 0.0173227, -0.0425346, 0.0157907, -0.0713160, 0.0598574
1: -0.0506534, 0.0357918, -0.0383179, 0.0256880, -0.0763414, 0.0741097
2: -0.0542474, 0.1227816, -0.0403522, 0.1082275, -0.1624749, 0.1631338
3: -0.0147985, 0.0586518, -0.0139233, 0.0476497, -0.0624483, 0.0725752
4: -0.0523453, 0.0622690, -0.0416041, 0.0468599, -0.0992052, 0.1038731
5: -0.0378061, 0.0488751, -0.0247817, 0.0370477, -0.0748538, 0.0736568
6: -0.1057829, 0.0669075, -0.0915258, 0.0536503, -0.1594332, 0.1584333
7: 0.8467187, 1.0108578, 0.8661014, 1.0035493, -0.1568307, 0.1447564
8: -0.0547986, 0.1001438, -0.0517619, 0.0848241, -0.1396227, 0.1477848
9: -0.0810614, 0.0598299, -0.0705455, 0.0391388, -0.1202002, 0.1303754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0555254, 0.0173227, -0.0516358, 0.0170662, -0.0725916, 0.0689586
1: -0.0506534, 0.0357918, -0.0467677, 0.0328764, -0.0835298, 0.0825595
2: -0.0542474, 0.1227816, -0.0500130, 0.1183230, -0.1725704, 0.1727945
3: -0.0147985, 0.0586518, -0.0148788, 0.0554075, -0.0702061, 0.0735306
4: -0.0523453, 0.0622690, -0.0486630, 0.0574092, -0.1097545, 0.1109320
5: -0.0378061, 0.0488751, -0.0338588, 0.0449455, -0.0827516, 0.0827339
6: -0.1057829, 0.0669075, -0.1012731, 0.0628187, -0.1686016, 0.1681806
7: 0.8467187, 1.0108578, 0.8526630, 1.0085737, -0.1618550, 0.1581948
8: -0.0547986, 0.1001438, -0.0550770, 0.0949969, -0.1491711, 0.1518927
9: -0.0810614, 0.0598299, -0.0779322, 0.0526288, -0.1336902, 0.1377621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0741686, 0.0246432, -0.0425346, 0.0157907, -0.0899593, 0.0671779
1: -0.0692777, 0.0497655, -0.0383179, 0.0256880, -0.0949658, 0.0880834
2: -0.0745436, 0.1441523, -0.0403522, 0.1082275, -0.1827711, 0.1845045
3: -0.0260966, 0.0742018, -0.0139233, 0.0476497, -0.0737463, 0.0881252
4: -0.0699956, 0.0855629, -0.0416041, 0.0468599, -0.1168555, 0.1271671
5: -0.0567261, 0.0677103, -0.0247817, 0.0370477, -0.0937738, 0.0924920
6: -0.1273992, 0.0865056, -0.0915258, 0.0536503, -0.1810495, 0.1780314
7: 0.8182260, 1.0218064, 0.8661014, 1.0035493, -0.1853233, 0.1557049
8: -0.0539877, 0.1248140, -0.0517619, 0.0848241, -0.1388118, 0.1723276
9: -0.0960602, 0.0943464, -0.0705455, 0.0391388, -0.1351990, 0.1648919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0741686, 0.0246432, -0.0516358, 0.0170662, -0.0912348, 0.0762791
1: -0.0692777, 0.0497655, -0.0467677, 0.0328764, -0.1021541, 0.0965332
2: -0.0745436, 0.1441523, -0.0500130, 0.1183230, -0.1928667, 0.1941652
3: -0.0260966, 0.0742018, -0.0148788, 0.0554075, -0.0815041, 0.0890806
4: -0.0699956, 0.0855629, -0.0486630, 0.0574092, -0.1274047, 0.1342259
5: -0.0567261, 0.0677103, -0.0338588, 0.0449455, -0.1016716, 0.1015691
6: -0.1273992, 0.0865056, -0.1012731, 0.0628187, -0.1902179, 0.1877787
7: 0.8182260, 1.0218064, 0.8526630, 1.0085737, -0.1903476, 0.1691433
8: -0.0539877, 0.1248140, -0.0550770, 0.0949969, -0.1489846, 0.1764168
9: -0.0960602, 0.0943464, -0.0779322, 0.0526288, -0.1486890, 0.1722785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0430042, 0.0163419, -0.0589047, 0.0186496, -0.0616538, 0.0752465
1: -0.0386638, 0.0260986, -0.0540292, 0.0383246, -0.0769884, 0.0801278
2: -0.0408845, 0.1087798, -0.0579262, 0.1266552, -0.1675397, 0.1667060
3: -0.0143362, 0.0478811, -0.0136850, 0.0614703, -0.0758065, 0.0615661
4: -0.0419338, 0.0474322, -0.0555446, 0.0664913, -0.1084251, 0.1029768
5: -0.0252113, 0.0374611, -0.0412355, 0.0522892, -0.0775005, 0.0786966
6: -0.0920536, 0.0541018, -0.1097010, 0.0704598, -0.1625134, 0.1638028
7: 0.8653668, 1.0037758, 0.8415542, 1.0128422, -0.1474754, 0.1622217
8: -0.0531944, 0.0853521, -0.0509348, 0.1046155, -0.1537745, 0.1362869
9: -0.0709264, 0.0397937, -0.0837801, 0.0660863, -0.1370127, 0.1235738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0430042, 0.0163419, -0.0684047, 0.0223800, -0.0653842, 0.0847466
1: -0.0386638, 0.0260986, -0.0635196, 0.0454453, -0.0841091, 0.0896183
2: -0.0408845, 0.1087798, -0.0682687, 0.1375451, -0.1784296, 0.1770485
3: -0.0143362, 0.0478811, -0.0212437, 0.0693943, -0.0837304, 0.0691248
4: -0.0419338, 0.0474322, -0.0645387, 0.0783611, -0.1202950, 0.1119709
5: -0.0252113, 0.0374611, -0.0508766, 0.0618871, -0.0870984, 0.0883377
6: -0.0920536, 0.0541018, -0.1207161, 0.0804465, -0.1725001, 0.1748179
7: 0.8653668, 1.0037758, 0.8270352, 1.0184215, -0.1530547, 0.1767406
8: -0.0531944, 0.0853521, -0.0542353, 0.1171868, -0.1644669, 0.1395874
9: -0.0709264, 0.0397937, -0.0914230, 0.0836750, -0.1546013, 0.1312168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0555025, 0.0179202, -0.0589047, 0.0186496, -0.0741521, 0.0768249
1: -0.0506305, 0.0357746, -0.0540292, 0.0383246, -0.0889551, 0.0898038
2: -0.0542224, 0.1227553, -0.0579262, 0.1266552, -0.1808776, 0.1806815
3: -0.0155184, 0.0586326, -0.0136850, 0.0614703, -0.0769887, 0.0723176
4: -0.0523237, 0.0622403, -0.0555446, 0.0664913, -0.1188149, 0.1177849
5: -0.0377828, 0.0488520, -0.0412355, 0.0522892, -0.0900720, 0.0900875
6: -0.1057563, 0.0668834, -0.1097010, 0.0704598, -0.1762161, 0.1765844
7: 0.8467536, 1.0108443, 0.8415542, 1.0128422, -0.1660886, 0.1692902
8: -0.0572963, 0.1001135, -0.0509348, 0.1046155, -0.1609055, 0.1510483
9: -0.0810429, 0.0597876, -0.0837801, 0.0660863, -0.1471292, 0.1435676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0555025, 0.0179202, -0.0684047, 0.0223800, -0.0778824, 0.0863249
1: -0.0506305, 0.0357746, -0.0635196, 0.0454453, -0.0960758, 0.0992942
2: -0.0542224, 0.1227553, -0.0682687, 0.1375451, -0.1917676, 0.1910240
3: -0.0155184, 0.0586326, -0.0212437, 0.0693943, -0.0849126, 0.0798763
4: -0.0523237, 0.0622403, -0.0645387, 0.0783611, -0.1306848, 0.1267790
5: -0.0377828, 0.0488520, -0.0508766, 0.0618871, -0.0996699, 0.0997286
6: -0.1057563, 0.0668834, -0.1207161, 0.0804465, -0.1862028, 0.1875996
7: 0.8467536, 1.0108443, 0.8270352, 1.0184215, -0.1716679, 0.1838091
8: -0.0572963, 0.1001135, -0.0542353, 0.1171868, -0.1719134, 0.1543488
9: -0.0810429, 0.0597876, -0.0914230, 0.0836750, -0.1647179, 0.1512106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0645275, 0.0208575, -0.0431453, 0.0165977, -0.0811252, 0.0640028
1: -0.0596463, 0.0425391, -0.0387860, 0.0262140, -0.0858603, 0.0813251
2: -0.0640476, 0.1331007, -0.0410348, 0.1089349, -0.1729825, 0.1741355
3: -0.0179792, 0.0661603, -0.0145278, 0.0479935, -0.0659727, 0.0806881
4: -0.0608679, 0.0735168, -0.0420308, 0.0475930, -0.1084609, 0.1155476
5: -0.0469418, 0.0579699, -0.0253484, 0.0375772, -0.0845191, 0.0833183
6: -0.1162205, 0.0763706, -0.0922019, 0.0542396, -0.1704602, 0.1685725
7: 0.8329608, 1.0161443, 0.8651605, 1.0038503, -0.1708896, 0.1509838
8: -0.0593606, 0.1120560, -0.0538593, 0.0855004, -0.1448610, 0.1615312
9: -0.0883038, 0.0764965, -0.0710412, 0.0399777, -0.1282815, 0.1475377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0645275, 0.0208575, -0.0523474, 0.0178621, -0.0823896, 0.0732048
1: -0.0596463, 0.0425391, -0.0474785, 0.0334098, -0.0930561, 0.0900176
2: -0.0640476, 0.1331007, -0.0507876, 0.1191386, -0.1831862, 0.1838883
3: -0.0179792, 0.0661603, -0.0154749, 0.0560009, -0.0739801, 0.0816351
4: -0.0608679, 0.0735168, -0.0493366, 0.0582982, -0.1191661, 0.1228534
5: -0.0469418, 0.0579699, -0.0345808, 0.0456644, -0.0926062, 0.0925507
6: -0.1162205, 0.0763706, -0.1020980, 0.0635667, -0.1797872, 0.1784686
7: 0.8329608, 1.0161443, 0.8515756, 1.0089915, -0.1760307, 0.1645687
8: -0.0593606, 0.1120560, -0.0571453, 0.0959384, -0.1552990, 0.1656592
9: -0.0883038, 0.0764965, -0.0785046, 0.0539461, -0.1422498, 0.1550011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0817171, 0.0276071, -0.0431453, 0.0165977, -0.0983148, 0.0707524
1: -0.0768185, 0.0554232, -0.0387860, 0.0262140, -0.1030324, 0.0942092
2: -0.0827612, 0.1528048, -0.0410348, 0.1089349, -0.1916961, 0.1938396
3: -0.0324519, 0.0804978, -0.0145278, 0.0479935, -0.0804454, 0.0950256
4: -0.0771418, 0.0949943, -0.0420308, 0.0475930, -0.1247348, 0.1370251
5: -0.0643865, 0.0753365, -0.0253484, 0.0375772, -0.1019637, 0.1006849
6: -0.1361514, 0.0944405, -0.0922019, 0.0542396, -0.1903910, 0.1866424
7: 0.8066901, 1.0262394, 0.8651605, 1.0038503, -0.1971602, 0.1610789
8: -0.0582217, 0.1348025, -0.0538593, 0.0855004, -0.1437221, 0.1838685
9: -0.1021330, 0.1083214, -0.0710412, 0.0399777, -0.1421107, 0.1793626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0817171, 0.0276071, -0.0523474, 0.0178621, -0.0995791, 0.0799545
1: -0.0768185, 0.0554232, -0.0474785, 0.0334098, -0.1102282, 0.1029017
2: -0.0827612, 0.1528048, -0.0507876, 0.1191386, -0.2018998, 0.2035924
3: -0.0324519, 0.0804978, -0.0154749, 0.0560009, -0.0884528, 0.0959726
4: -0.0771418, 0.0949943, -0.0493366, 0.0582982, -0.1354399, 0.1443309
5: -0.0643865, 0.0753365, -0.0345808, 0.0456644, -0.1100509, 0.1099173
6: -0.1361514, 0.0944405, -0.1020980, 0.0635667, -0.1997180, 0.1965385
7: 0.8066901, 1.0262394, 0.8515756, 1.0089915, -0.2023014, 0.1746638
8: -0.0582217, 0.1348025, -0.0571453, 0.0959384, -0.1541601, 0.1880003
9: -0.1021330, 0.1083214, -0.0785046, 0.0539461, -0.1560790, 0.1868261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0519553, 0.0179822, -0.0596262, 0.0189330, -0.0708882, 0.0776083
1: -0.0470869, 0.0331159, -0.0547500, 0.0388654, -0.0859523, 0.0878659
2: -0.0503608, 0.1186892, -0.0587118, 0.1274822, -0.1778431, 0.1774011
3: -0.0155648, 0.0556740, -0.0142801, 0.0620721, -0.0776369, 0.0699541
4: -0.0489655, 0.0578084, -0.0562277, 0.0673928, -0.1163582, 0.1140361
5: -0.0341830, 0.0452683, -0.0419677, 0.0530182, -0.0872012, 0.0872360
6: -0.1016435, 0.0631546, -0.1105376, 0.0712183, -0.1728617, 0.1736921
7: 0.8521746, 1.0087614, 0.8404516, 1.0132661, -0.1610914, 0.1683098
8: -0.0574574, 0.0954197, -0.0529999, 0.1055703, -0.1593181, 0.1484196
9: -0.0781892, 0.0532203, -0.0843605, 0.0674222, -0.1456113, 0.1375808

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0519553, 0.0179822, -0.0691534, 0.0226739, -0.0746292, 0.0871356
1: -0.0470869, 0.0331159, -0.0642676, 0.0460064, -0.0930933, 0.0973835
2: -0.0503608, 0.1186892, -0.0690837, 0.1384033, -0.1887641, 0.1877729
3: -0.0155648, 0.0556740, -0.0218740, 0.0700187, -0.0855835, 0.0775480
4: -0.0489655, 0.0578084, -0.0652474, 0.0792966, -0.1282621, 0.1230558
5: -0.0341830, 0.0452683, -0.0516364, 0.0626435, -0.0968265, 0.0969047
6: -0.1016435, 0.0631546, -0.1215842, 0.0812334, -0.1828769, 0.1847387
7: 0.8521746, 1.0087614, 0.8258910, 1.0188612, -0.1666865, 0.1828704
8: -0.0574574, 0.0954197, -0.0561778, 0.1181774, -0.1701382, 0.1515975
9: -0.0781892, 0.0532203, -0.0920254, 0.0850610, -0.1632501, 0.1452457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0644373, 0.0208221, -0.0596262, 0.0189330, -0.0833703, 0.0804482
1: -0.0595562, 0.0424715, -0.0547500, 0.0388654, -0.0984217, 0.0972215
2: -0.0639495, 0.1329973, -0.0587118, 0.1274822, -0.1914317, 0.1917091
3: -0.0179033, 0.0660851, -0.0142801, 0.0620721, -0.0799754, 0.0803652
4: -0.0607825, 0.0734040, -0.0562277, 0.0673928, -0.1281753, 0.1296317
5: -0.0468503, 0.0578788, -0.0419677, 0.0530182, -0.0998684, 0.0998465
6: -0.1161160, 0.0762758, -0.1105376, 0.0712183, -0.1873342, 0.1868134
7: 0.8330986, 1.0160918, 0.8404516, 1.0132661, -0.1801674, 0.1756402
8: -0.0617287, 0.1119367, -0.0529999, 0.1055703, -0.1672990, 0.1649367
9: -0.0882312, 0.0763295, -0.0843605, 0.0674222, -0.1556533, 0.1606900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0644373, 0.0208221, -0.0691534, 0.0226739, -0.0871112, 0.0899755
1: -0.0595562, 0.0424715, -0.0642676, 0.0460064, -0.1055627, 0.1067391
2: -0.0639495, 0.1329973, -0.0690837, 0.1384033, -0.2023527, 0.2020810
3: -0.0179033, 0.0660851, -0.0218740, 0.0700187, -0.0879219, 0.0879591
4: -0.0607825, 0.0734040, -0.0652474, 0.0792966, -0.1400791, 0.1386514
5: -0.0468503, 0.0578788, -0.0516364, 0.0626435, -0.1094937, 0.1095152
6: -0.1161160, 0.0762758, -0.1215842, 0.0812334, -0.1973494, 0.1978600
7: 0.8330986, 1.0160918, 0.8258910, 1.0188612, -0.1857625, 0.1902008
8: -0.0617287, 0.1119367, -0.0561778, 0.1181774, -0.1788690, 0.1681145
9: -0.0882312, 0.0763295, -0.0920254, 0.0850610, -0.1732922, 0.1683549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0560204, 0.0175849, -0.0557359, 0.0174956, -0.0735160, 0.0733208
1: -0.0511479, 0.0361627, -0.0508637, 0.0359496, -0.0870975, 0.0870264
2: -0.0547863, 0.1233490, -0.0544765, 0.1230229, -0.1778091, 0.1778255
3: -0.0152672, 0.0590646, -0.0152004, 0.0588273, -0.0740945, 0.0742650
4: -0.0528140, 0.0628875, -0.0525447, 0.0625320, -0.1153460, 0.1154322
5: -0.0383084, 0.0493752, -0.0380197, 0.0490878, -0.0873962, 0.0873949
6: -0.1063568, 0.0674278, -0.1060269, 0.0671288, -0.1734856, 0.1734547
7: 0.8459622, 1.0111486, 0.8463969, 1.0109816, -0.1650193, 0.1647516
8: -0.0564248, 0.1007988, -0.0561928, 0.1004224, -0.1567695, 0.1547774
9: -0.0814596, 0.0607464, -0.0812308, 0.0602198, -0.1416794, 0.1419772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0560204, 0.0175849, -0.0641420, 0.0207062, -0.0767265, 0.0817269
1: -0.0511479, 0.0361627, -0.0592612, 0.0422503, -0.0933981, 0.0954240
2: -0.0547863, 0.1233490, -0.0636280, 0.1326587, -0.1874450, 0.1869771
3: -0.0152672, 0.0590646, -0.0176547, 0.0658388, -0.0811060, 0.0767193
4: -0.0528140, 0.0628875, -0.0605030, 0.0730351, -0.1258491, 0.1233905
5: -0.0383084, 0.0493752, -0.0465506, 0.0575805, -0.0958889, 0.0959258
6: -0.1063568, 0.0674278, -0.1157736, 0.0759654, -0.1823222, 0.1832014
7: 0.8459622, 1.0111486, 0.8335499, 1.0159180, -0.1699558, 0.1775987
8: -0.0564248, 0.1007988, -0.0592139, 0.1115461, -0.1665885, 0.1600127
9: -0.0814596, 0.0607464, -0.0879936, 0.0757828, -0.1572425, 0.1487400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0746582, 0.0248354, -0.0557359, 0.0174956, -0.0921538, 0.0805713
1: -0.0697668, 0.0501324, -0.0508637, 0.0359496, -0.1057163, 0.1009961
2: -0.0750765, 0.1447134, -0.0544765, 0.1230229, -0.1980994, 0.1991900
3: -0.0265087, 0.0746102, -0.0152004, 0.0588273, -0.0853360, 0.0898105
4: -0.0704590, 0.0861745, -0.0525447, 0.0625320, -0.1329910, 0.1387193
5: -0.0572229, 0.0682049, -0.0380197, 0.0490878, -0.1063107, 0.1062247
6: -0.1279668, 0.0870201, -0.1060269, 0.0671288, -0.1950956, 0.1930470
7: 0.8174782, 1.0220940, 0.8463969, 1.0109816, -0.1935033, 0.1756971
8: -0.0556930, 0.1254617, -0.0561928, 0.1004224, -0.1561154, 0.1793431
9: -0.0964540, 0.0952526, -0.0812308, 0.0602198, -0.1566738, 0.1764835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0746582, 0.0248354, -0.0641420, 0.0207062, -0.0953644, 0.0889774
1: -0.0697668, 0.0501324, -0.0592612, 0.0422503, -0.1120170, 0.1093937
2: -0.0750765, 0.1447134, -0.0636280, 0.1326587, -0.2077353, 0.2083415
3: -0.0265087, 0.0746102, -0.0176547, 0.0658388, -0.0923475, 0.0922648
4: -0.0704590, 0.0861745, -0.0605030, 0.0730351, -0.1434941, 0.1466776
5: -0.0572229, 0.0682049, -0.0465506, 0.0575805, -0.1148034, 0.1147555
6: -0.1279668, 0.0870201, -0.1157736, 0.0759654, -0.2039322, 0.2027937
7: 0.8174782, 1.0220940, 0.8335499, 1.0159180, -0.1984398, 0.1885442
8: -0.0556930, 0.1254617, -0.0592139, 0.1115461, -0.1672391, 0.1846756
9: -0.0964540, 0.0952526, -0.0879936, 0.0757828, -0.1722369, 0.1832463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0430042, 0.0163419, -0.0731253, 0.0242335, -0.0672377, 0.0894672
1: -0.0386638, 0.0260986, -0.0682354, 0.0489835, -0.0876472, 0.0943340
2: -0.0408845, 0.1087798, -0.0734078, 0.1429563, -0.1838408, 0.1821876
3: -0.0143362, 0.0478811, -0.0252181, 0.0733315, -0.0876677, 0.0730992
4: -0.0419338, 0.0474322, -0.0690077, 0.0842593, -0.1261931, 0.1164400
5: -0.0252113, 0.0374611, -0.0556672, 0.0666563, -0.0918676, 0.0931283
6: -0.0920536, 0.0541018, -0.1261894, 0.0854087, -0.1774623, 0.1802912
7: 0.8653668, 1.0037758, 0.8198209, 1.0211935, -0.1558267, 0.1839550
8: -0.0531944, 0.0853521, -0.0553077, 0.1234333, -0.1723019, 0.1406598
9: -0.0709264, 0.0397937, -0.0952208, 0.0924146, -0.1633410, 0.1350145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0430042, 0.0163419, -0.0813327, 0.0274561, -0.0704603, 0.0976745
1: -0.0386638, 0.0260986, -0.0764343, 0.0551350, -0.0937988, 0.1025329
2: -0.0408845, 0.1087798, -0.0823427, 0.1523641, -0.1932487, 0.1911225
3: -0.0143362, 0.0478811, -0.0321282, 0.0801770, -0.0945132, 0.0800093
4: -0.0419338, 0.0474322, -0.0767778, 0.0945139, -0.1364478, 0.1242100
5: -0.0252113, 0.0374611, -0.0639963, 0.0749481, -0.1001594, 0.1014574
6: -0.0920536, 0.0541018, -0.1357055, 0.0940363, -0.1860899, 0.1898073
7: 0.8653668, 1.0037758, 0.8072776, 1.0260134, -0.1606466, 0.1964982
8: -0.0531944, 0.0853521, -0.0584877, 0.1342938, -0.1813025, 0.1438398
9: -0.0709264, 0.0397937, -0.1018237, 0.1076096, -0.1785360, 0.1416175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0555025, 0.0179202, -0.0731253, 0.0242335, -0.0797360, 0.0910455
1: -0.0506305, 0.0357746, -0.0682354, 0.0489835, -0.0996140, 0.1040100
2: -0.0542224, 0.1227553, -0.0734078, 0.1429563, -0.1971787, 0.1961631
3: -0.0155184, 0.0586326, -0.0252181, 0.0733315, -0.0888499, 0.0838507
4: -0.0523237, 0.0622403, -0.0690077, 0.0842593, -0.1365830, 0.1312481
5: -0.0377828, 0.0488520, -0.0556672, 0.0666563, -0.1044391, 0.1045192
6: -0.1057563, 0.0668834, -0.1261894, 0.0854087, -0.1911650, 0.1930728
7: 0.8467536, 1.0108443, 0.8198209, 1.0211935, -0.1744399, 0.1910235
8: -0.0572963, 0.1001135, -0.0553077, 0.1234333, -0.1777591, 0.1554212
9: -0.0810429, 0.0597876, -0.0952208, 0.0924146, -0.1734576, 0.1550083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0555025, 0.0179202, -0.0813327, 0.0274561, -0.0829586, 0.0992528
1: -0.0506305, 0.0357746, -0.0764343, 0.0551350, -0.1057655, 0.1122089
2: -0.0542224, 0.1227553, -0.0823427, 0.1523641, -0.2065866, 0.2050980
3: -0.0155184, 0.0586326, -0.0321282, 0.0801770, -0.0956954, 0.0907608
4: -0.0523237, 0.0622403, -0.0767778, 0.0945139, -0.1468376, 0.1390181
5: -0.0377828, 0.0488520, -0.0639963, 0.0749481, -0.1127309, 0.1128483
6: -0.1057563, 0.0668834, -0.1357055, 0.0940363, -0.1997926, 0.2025889
7: 0.8467536, 1.0108443, 0.8072776, 1.0260134, -0.1792598, 0.2035667
8: -0.0572963, 0.1001135, -0.0584877, 0.1342938, -0.1873522, 0.1586012
9: -0.0810429, 0.0597876, -0.1018237, 0.1076096, -0.1886526, 0.1616113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0650108, 0.0210473, -0.0564640, 0.0183070, -0.0833177, 0.0775113
1: -0.0601291, 0.0429014, -0.0515910, 0.0364953, -0.0966244, 0.0944924
2: -0.0645737, 0.1336546, -0.0552693, 0.1238575, -0.1884312, 0.1889239
3: -0.0183861, 0.0665633, -0.0158081, 0.0594347, -0.0778208, 0.0823714
4: -0.0613255, 0.0741206, -0.0532340, 0.0634418, -0.1247673, 0.1273545
5: -0.0474323, 0.0584581, -0.0387587, 0.0498234, -0.0972557, 0.0972168
6: -0.1167808, 0.0768786, -0.1068711, 0.0678941, -0.1846750, 0.1837498
7: 0.8322222, 1.0164282, 0.8452842, 1.0114089, -0.1791867, 0.1711441
8: -0.0610462, 0.1126955, -0.0583015, 0.1013859, -0.1624321, 0.1684831
9: -0.0886925, 0.0773913, -0.0818166, 0.0615676, -0.1502601, 0.1592079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0650108, 0.0210473, -0.0648960, 0.0210022, -0.0860130, 0.0859432
1: -0.0601291, 0.0429014, -0.0600144, 0.0428153, -0.1029444, 0.1029157
2: -0.0645737, 0.1336546, -0.0644488, 0.1335230, -0.1980967, 0.1981034
3: -0.0183861, 0.0665633, -0.0182894, 0.0664675, -0.0848536, 0.0848527
4: -0.0613255, 0.0741206, -0.0612167, 0.0739771, -0.1353026, 0.1353373
5: -0.0474323, 0.0584581, -0.0473157, 0.0583422, -0.1057745, 0.1057738
6: -0.1167808, 0.0768786, -0.1166477, 0.0767579, -0.1935387, 0.1935264
7: 0.8322222, 1.0164282, 0.8323978, 1.0163609, -0.1841387, 0.1840305
8: -0.0610462, 0.1126955, -0.0613592, 0.1125436, -0.1735898, 0.1739444
9: -0.0886925, 0.0773913, -0.0886001, 0.0771787, -0.1658712, 0.1659914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0822103, 0.0278008, -0.0564640, 0.0183070, -0.1005173, 0.0842648
1: -0.0773112, 0.0557929, -0.0515910, 0.0364953, -0.1138065, 0.1073840
2: -0.0832983, 0.1533702, -0.0552693, 0.1238575, -0.2071557, 0.2086395
3: -0.0328672, 0.0809092, -0.0158081, 0.0594347, -0.0923018, 0.0967173
4: -0.0776087, 0.0956105, -0.0532340, 0.0634418, -0.1410505, 0.1488445
5: -0.0648870, 0.0758348, -0.0387587, 0.0498234, -0.1147104, 0.1145934
6: -0.1367232, 0.0949589, -0.1068711, 0.0678941, -0.2046173, 0.2018301
7: 0.8059363, 1.0265290, 0.8452842, 1.0114089, -0.2054726, 0.1812448
8: -0.0599421, 0.1354552, -0.0583015, 0.1013859, -0.1613280, 0.1909146
9: -0.1025298, 0.1092346, -0.0818166, 0.0615676, -0.1640974, 0.1910511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0822103, 0.0278008, -0.0648960, 0.0210022, -0.1032125, 0.0926968
1: -0.0773112, 0.0557929, -0.0600144, 0.0428153, -0.1201265, 0.1158073
2: -0.0832983, 0.1533702, -0.0644488, 0.1335230, -0.2168212, 0.2178190
3: -0.0328672, 0.0809092, -0.0182894, 0.0664675, -0.0993347, 0.0991986
4: -0.0776087, 0.0956105, -0.0612167, 0.0739771, -0.1515858, 0.1568273
5: -0.0648870, 0.0758348, -0.0473157, 0.0583422, -0.1232292, 0.1231505
6: -0.1367232, 0.0949589, -0.1166477, 0.0767579, -0.2134811, 0.2116067
7: 0.8059363, 1.0265290, 0.8323978, 1.0163609, -0.2104245, 0.1941312
8: -0.0599421, 0.1354552, -0.0613592, 0.1125436, -0.1724857, 0.1964172
9: -0.1025298, 0.1092346, -0.0886001, 0.0771787, -0.1797085, 0.1978347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0519553, 0.0179822, -0.0738770, 0.0245287, -0.0764839, 0.0918592
1: -0.0470869, 0.0331159, -0.0689863, 0.0495469, -0.0966338, 0.1021022
2: -0.0503608, 0.1186892, -0.0742261, 0.1438179, -0.1941788, 0.1929153
3: -0.0155648, 0.0556740, -0.0258510, 0.0739585, -0.0895233, 0.0815250
4: -0.0489655, 0.0578084, -0.0697194, 0.0851986, -0.1341640, 0.1275278
5: -0.0341830, 0.0452683, -0.0564301, 0.0674157, -0.1015987, 0.1016984
6: -0.1016435, 0.0631546, -0.1270611, 0.0861989, -0.1878424, 0.1902156
7: 0.8521746, 1.0087614, 0.8186721, 1.0216352, -0.1694605, 0.1900893
8: -0.0574574, 0.0954197, -0.0573318, 0.1244280, -0.1779878, 0.1527515
9: -0.0781892, 0.0532203, -0.0958256, 0.0938064, -0.1719955, 0.1490459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0519553, 0.0179822, -0.0821236, 0.0277667, -0.0797220, 0.1001057
1: -0.0470869, 0.0331159, -0.0772245, 0.0557279, -0.1028148, 0.1103404
2: -0.0503608, 0.1186892, -0.0832039, 0.1532708, -0.2036317, 0.2018931
3: -0.0155648, 0.0556740, -0.0327941, 0.0808368, -0.0964016, 0.0884681
4: -0.0489655, 0.0578084, -0.0775266, 0.0955021, -0.1444676, 0.1353350
5: -0.0341830, 0.0452683, -0.0647990, 0.0757471, -0.1099301, 0.1100673
6: -0.1016435, 0.0631546, -0.1366226, 0.0948678, -0.1965113, 0.1997772
7: 0.8521746, 1.0087614, 0.8060688, 1.0264779, -0.1743033, 0.2026926
8: -0.0574574, 0.0954197, -0.0604206, 0.1353404, -0.1870860, 0.1558402
9: -0.0781892, 0.0532203, -0.1024601, 0.1090740, -0.1872632, 0.1556804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0644373, 0.0208221, -0.0738770, 0.0245287, -0.0889660, 0.0946991
1: -0.0595562, 0.0424715, -0.0689863, 0.0495469, -0.1091031, 0.1114578
2: -0.0639495, 0.1329973, -0.0742261, 0.1438179, -0.2077674, 0.2072234
3: -0.0179033, 0.0660851, -0.0258510, 0.0739585, -0.0918618, 0.0919361
4: -0.0607825, 0.0734040, -0.0697194, 0.0851986, -0.1459811, 0.1431234
5: -0.0468503, 0.0578788, -0.0564301, 0.0674157, -0.1142660, 0.1143089
6: -0.1161160, 0.0762758, -0.1270611, 0.0861989, -0.2023149, 0.2033368
7: 0.8330986, 1.0160918, 0.8186721, 1.0216352, -0.1885365, 0.1974198
8: -0.0617287, 0.1119367, -0.0573318, 0.1244280, -0.1848294, 0.1692685
9: -0.0882312, 0.0763295, -0.0958256, 0.0938064, -0.1820375, 0.1721551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0644373, 0.0208221, -0.0821236, 0.0277667, -0.0922041, 0.1029457
1: -0.0595562, 0.0424715, -0.0772245, 0.0557279, -0.1152841, 0.1196960
2: -0.0639495, 0.1329973, -0.0832039, 0.1532708, -0.2172203, 0.2162011
3: -0.0179033, 0.0660851, -0.0327941, 0.0808368, -0.0987401, 0.0988792
4: -0.0607825, 0.0734040, -0.0775266, 0.0955021, -0.1562847, 0.1509306
5: -0.0468503, 0.0578788, -0.0647990, 0.0757471, -0.1225974, 0.1226778
6: -0.1161160, 0.0762758, -0.1366226, 0.0948678, -0.2109838, 0.2128984
7: 0.8330986, 1.0160918, 0.8060688, 1.0264779, -0.1933793, 0.2100230
8: -0.0617287, 0.1119367, -0.0604206, 0.1353404, -0.1949081, 0.1723573
9: -0.0882312, 0.0763295, -0.1024601, 0.1090740, -0.1973052, 0.1787896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.95 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.95
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0354598, 0.0143041, -0.0425346, 0.0157907, -0.0512505, 0.0568388
1: -0.0334864, 0.0186035, -0.0383179, 0.0256880, -0.0591744, 0.0569214
2: -0.0311662, 0.0997398, -0.0403522, 0.1082275, -0.1393936, 0.1400920
3: -0.0128099, 0.0436573, -0.0139233, 0.0476497, -0.0604596, 0.0575806
4: -0.0359145, 0.0395610, -0.0416041, 0.0468599, -0.0827743, 0.0811652
5: -0.0196553, 0.0299146, -0.0247817, 0.0370477, -0.0567030, 0.0546962
6: -0.0835187, 0.0458598, -0.0915258, 0.0536503, -0.1371691, 0.1373856
7: 0.8787041, 0.9996369, 0.8661014, 1.0035493, -0.1248452, 0.1335355
8: -0.0478985, 0.0768380, -0.0517619, 0.0848241, -0.1276526, 0.1245800
9: -0.0639742, 0.0300489, -0.0705455, 0.0391388, -0.1031130, 0.1005944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0467779, 0.0159775, -0.0425346, 0.0157907, -0.0625685, 0.0585122
1: -0.0420774, 0.0291203, -0.0383179, 0.0256880, -0.0677654, 0.0674382
2: -0.0448264, 0.1128445, -0.0403522, 0.1082275, -0.1530539, 0.1531967
3: -0.0140633, 0.0512277, -0.0139233, 0.0476497, -0.0617130, 0.0651510
4: -0.0445105, 0.0516446, -0.0416041, 0.0468599, -0.0913704, 0.0932487
5: -0.0289425, 0.0405034, -0.0247817, 0.0370477, -0.0659902, 0.0652851
6: -0.0959383, 0.0578047, -0.0915258, 0.0536503, -0.1495886, 0.1493305
7: 0.8599609, 1.0058222, 0.8661014, 1.0035493, -0.1435884, 0.1397207
8: -0.0522475, 0.0892382, -0.0517619, 0.0848241, -0.1340474, 0.1372538
9: -0.0740007, 0.0446140, -0.0705455, 0.0391388, -0.1131395, 0.1151595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0354598, 0.0143041, -0.0516358, 0.0170662, -0.0525261, 0.0659399
1: -0.0334864, 0.0186035, -0.0467677, 0.0328764, -0.0663628, 0.0653713
2: -0.0311662, 0.0997398, -0.0500130, 0.1183230, -0.1494892, 0.1497528
3: -0.0128099, 0.0436573, -0.0148788, 0.0554075, -0.0682174, 0.0585361
4: -0.0359145, 0.0395610, -0.0486630, 0.0574092, -0.0933236, 0.0882241
5: -0.0196553, 0.0299146, -0.0338588, 0.0449455, -0.0646008, 0.0637733
6: -0.0835187, 0.0458598, -0.1012731, 0.0628187, -0.1463374, 0.1471329
7: 0.8787041, 0.9996369, 0.8526630, 1.0085737, -0.1298695, 0.1469739
8: -0.0478985, 0.0768380, -0.0550770, 0.0949969, -0.1377598, 0.1301023
9: -0.0639742, 0.0300489, -0.0779322, 0.0526288, -0.1166030, 0.1079810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0467779, 0.0159775, -0.0516358, 0.0170662, -0.0638441, 0.0676134
1: -0.0420774, 0.0291203, -0.0467677, 0.0328764, -0.0749538, 0.0758880
2: -0.0448264, 0.1128445, -0.0500130, 0.1183230, -0.1631494, 0.1628574
3: -0.0140633, 0.0512277, -0.0148788, 0.0554075, -0.0694708, 0.0661064
4: -0.0445105, 0.0516446, -0.0486630, 0.0574092, -0.1019196, 0.1003076
5: -0.0289425, 0.0405034, -0.0338588, 0.0449455, -0.0738880, 0.0743622
6: -0.0959383, 0.0578047, -0.1012731, 0.0628187, -0.1587570, 0.1590778
7: 0.8599609, 1.0058222, 0.8526630, 1.0085737, -0.1486127, 0.1531591
8: -0.0522475, 0.0892382, -0.0550770, 0.0949969, -0.1441546, 0.1427761
9: -0.0740007, 0.0446140, -0.0779322, 0.0526288, -0.1266295, 0.1225461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0509658, 0.0152277, -0.0425346, 0.0157907, -0.0667565, 0.0577623
1: -0.0458720, 0.0324709, -0.0383179, 0.0256880, -0.0715600, 0.0707887
2: -0.0491977, 0.1173516, -0.0403522, 0.1082275, -0.1574252, 0.1577038
3: -0.0125760, 0.0549562, -0.0139233, 0.0476497, -0.0602257, 0.0688796
4: -0.0473692, 0.0563156, -0.0416041, 0.0468599, -0.0942291, 0.0979197
5: -0.0330860, 0.0438770, -0.0247817, 0.0370477, -0.0701337, 0.0686587
6: -0.1002458, 0.0619147, -0.0915258, 0.0536503, -0.1538961, 0.1534405
7: 0.8539665, 1.0080949, 0.8661014, 1.0035493, -0.1495829, 0.1419935
8: -0.0470869, 0.0935475, -0.0517619, 0.0848241, -0.1319110, 0.1416181
9: -0.0774126, 0.0499590, -0.0705455, 0.0391388, -0.1165514, 0.1205045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0640337, 0.0187563, -0.0425346, 0.0157907, -0.0798244, 0.0612909
1: -0.0577126, 0.0429261, -0.0383179, 0.0256880, -0.0834006, 0.0812440
2: -0.0628379, 0.1314158, -0.0403522, 0.1082275, -0.1710654, 0.1717680
3: -0.0138287, 0.0665909, -0.0139233, 0.0476497, -0.0614784, 0.0805142
4: -0.0562897, 0.0708906, -0.0416041, 0.0468599, -0.1031496, 0.1124948
5: -0.0460152, 0.0544039, -0.0247817, 0.0370477, -0.0830629, 0.0791856
6: -0.1136869, 0.0747396, -0.0915258, 0.0536503, -0.1673372, 0.1662654
7: 0.8352613, 1.0151875, 0.8661014, 1.0035493, -0.1682880, 0.1490861
8: -0.0514336, 0.1069937, -0.0517619, 0.0848241, -0.1362576, 0.1550589
9: -0.0880593, 0.0666374, -0.0705455, 0.0391388, -0.1271981, 0.1371830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0509658, 0.0152277, -0.0516358, 0.0170662, -0.0680320, 0.0668635
1: -0.0458720, 0.0324709, -0.0467677, 0.0328764, -0.0787484, 0.0792386
2: -0.0491977, 0.1173516, -0.0500130, 0.1183230, -0.1675208, 0.1673646
3: -0.0125760, 0.0549562, -0.0148788, 0.0554075, -0.0679835, 0.0698350
4: -0.0473692, 0.0563156, -0.0486630, 0.0574092, -0.1047784, 0.1049786
5: -0.0330860, 0.0438770, -0.0338588, 0.0449455, -0.0780315, 0.0777358
6: -0.1002458, 0.0619147, -0.1012731, 0.0628187, -0.1630645, 0.1631877
7: 0.8539665, 1.0080949, 0.8526630, 1.0085737, -0.1546072, 0.1554319
8: -0.0470869, 0.0935475, -0.0550770, 0.0949969, -0.1420838, 0.1471403
9: -0.0774126, 0.0499590, -0.0779322, 0.0526288, -0.1300415, 0.1278912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0640337, 0.0187563, -0.0516358, 0.0170662, -0.0810999, 0.0703921
1: -0.0577126, 0.0429261, -0.0467677, 0.0328764, -0.0905890, 0.0896938
2: -0.0628379, 0.1314158, -0.0500130, 0.1183230, -0.1811610, 0.1814288
3: -0.0138287, 0.0665909, -0.0148788, 0.0554075, -0.0692362, 0.0814696
4: -0.0562897, 0.0708906, -0.0486630, 0.0574092, -0.1136988, 0.1195537
5: -0.0460152, 0.0544039, -0.0338588, 0.0449455, -0.0909608, 0.0882627
6: -0.1136869, 0.0747396, -0.1012731, 0.0628187, -0.1765056, 0.1760127
7: 0.8352613, 1.0151875, 0.8526630, 1.0085737, -0.1733123, 0.1625245
8: -0.0514336, 0.1069937, -0.0550770, 0.0949969, -0.1464304, 0.1605811
9: -0.0880593, 0.0666374, -0.0779322, 0.0526288, -0.1406881, 0.1445696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0354598, 0.0143041, -0.0589047, 0.0186496, -0.0541094, 0.0732088
1: -0.0334864, 0.0186035, -0.0540292, 0.0383246, -0.0718110, 0.0726327
2: -0.0311662, 0.0997398, -0.0579262, 0.1266552, -0.1578213, 0.1576660
3: -0.0128099, 0.0436573, -0.0136850, 0.0614703, -0.0742802, 0.0573422
4: -0.0359145, 0.0395610, -0.0555446, 0.0664913, -0.1024057, 0.0951056
5: -0.0196553, 0.0299146, -0.0412355, 0.0522892, -0.0719445, 0.0711501
6: -0.0835187, 0.0458598, -0.1097010, 0.0704598, -0.1539785, 0.1555608
7: 0.8787041, 0.9996369, 0.8415542, 1.0128422, -0.1341380, 0.1580828
8: -0.0478985, 0.0768380, -0.0509348, 0.1046155, -0.1475223, 0.1277729
9: -0.0639742, 0.0300489, -0.0837801, 0.0660863, -0.1300605, 0.1138289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0483143, 0.0152277, -0.0589047, 0.0186496, -0.0669639, 0.0741323
1: -0.0418445, 0.0324709, -0.0540292, 0.0383246, -0.0801691, 0.0865001
2: -0.0491470, 0.1153460, -0.0579262, 0.1266552, -0.1758022, 0.1732722
3: -0.0125760, 0.0514721, -0.0136850, 0.0614703, -0.0740463, 0.0651571
4: -0.0470514, 0.0513562, -0.0555446, 0.0664913, -0.1135427, 0.1069008
5: -0.0274782, 0.0438770, -0.0412355, 0.0522892, -0.0797674, 0.0851125
6: -0.0981278, 0.0611091, -0.1097010, 0.0704598, -0.1685876, 0.1708101
7: 0.8541052, 1.0072950, 0.8415542, 1.0128422, -0.1587369, 0.1657408
8: -0.0470869, 0.0913820, -0.0509348, 0.1046155, -0.1517024, 0.1423168
9: -0.0768370, 0.0457035, -0.0837801, 0.0660863, -0.1429233, 0.1294835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0354598, 0.0143041, -0.0684047, 0.0223800, -0.0578398, 0.0827089
1: -0.0334864, 0.0186035, -0.0635196, 0.0454453, -0.0789316, 0.0821232
2: -0.0311662, 0.0997398, -0.0682687, 0.1375451, -0.1687113, 0.1680085
3: -0.0128099, 0.0436573, -0.0212437, 0.0693943, -0.0822041, 0.0649010
4: -0.0359145, 0.0395610, -0.0645387, 0.0783611, -0.1142756, 0.1040997
5: -0.0196553, 0.0299146, -0.0508766, 0.0618871, -0.0815424, 0.0807911
6: -0.0835187, 0.0458598, -0.1207161, 0.0804465, -0.1639653, 0.1665759
7: 0.8787041, 0.9996369, 0.8270352, 1.0184215, -0.1397174, 0.1726017
8: -0.0478985, 0.0768380, -0.0542353, 0.1171868, -0.1599281, 0.1310733
9: -0.0639742, 0.0300489, -0.0914230, 0.0836750, -0.1476492, 0.1214719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0483143, 0.0152277, -0.0684047, 0.0223800, -0.0706943, 0.0836324
1: -0.0418445, 0.0324709, -0.0635196, 0.0454453, -0.0872897, 0.0959905
2: -0.0491470, 0.1153460, -0.0682687, 0.1375451, -0.1866921, 0.1836147
3: -0.0125760, 0.0514721, -0.0212437, 0.0693943, -0.0819702, 0.0727158
4: -0.0470514, 0.0513562, -0.0645387, 0.0783611, -0.1254126, 0.1158949
5: -0.0274782, 0.0438770, -0.0508766, 0.0618871, -0.0893653, 0.0947536
6: -0.0981278, 0.0611091, -0.1207161, 0.0804465, -0.1785743, 0.1818253
7: 0.8541052, 1.0072950, 0.8270352, 1.0184215, -0.1643163, 0.1802598
8: -0.0470869, 0.0913820, -0.0542353, 0.1171868, -0.1642737, 0.1456173
9: -0.0768370, 0.0457035, -0.0914230, 0.0836750, -0.1605120, 0.1371265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0467779, 0.0159775, -0.0589047, 0.0186496, -0.0654275, 0.0748822
1: -0.0420774, 0.0291203, -0.0540292, 0.0383246, -0.0804020, 0.0831495
2: -0.0448264, 0.1128445, -0.0579262, 0.1266552, -0.1714816, 0.1707706
3: -0.0140633, 0.0512277, -0.0136850, 0.0614703, -0.0755336, 0.0649126
4: -0.0445105, 0.0516446, -0.0555446, 0.0664913, -0.1110017, 0.1071891
5: -0.0289425, 0.0405034, -0.0412355, 0.0522892, -0.0812316, 0.0817389
6: -0.0959383, 0.0578047, -0.1097010, 0.0704598, -0.1663980, 0.1675057
7: 0.8599609, 1.0058222, 0.8415542, 1.0128422, -0.1528813, 0.1642680
8: -0.0522475, 0.0892382, -0.0509348, 0.1046155, -0.1539171, 0.1401730
9: -0.0740007, 0.0446140, -0.0837801, 0.0660863, -0.1400870, 0.1283940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0640337, 0.0187563, -0.0589047, 0.0186496, -0.0826833, 0.0776609
1: -0.0577126, 0.0429261, -0.0540292, 0.0383246, -0.0960372, 0.0969553
2: -0.0628379, 0.1314158, -0.0579262, 0.1266552, -0.1894931, 0.1893420
3: -0.0138287, 0.0665909, -0.0136850, 0.0614703, -0.0752990, 0.0802758
4: -0.0562897, 0.0708906, -0.0555446, 0.0664913, -0.1227809, 0.1264352
5: -0.0460152, 0.0544039, -0.0412355, 0.0522892, -0.0983044, 0.0956394
6: -0.1136869, 0.0747396, -0.1097010, 0.0704598, -0.1841466, 0.1844406
7: 0.8352613, 1.0151875, 0.8415542, 1.0128422, -0.1775808, 0.1736333
8: -0.0514336, 0.1069937, -0.0509348, 0.1046155, -0.1560490, 0.1579286
9: -0.0880593, 0.0666374, -0.0837801, 0.0660863, -0.1541456, 0.1504175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0467779, 0.0159775, -0.0684047, 0.0223800, -0.0691578, 0.0843823
1: -0.0420774, 0.0291203, -0.0635196, 0.0454453, -0.0875226, 0.0926399
2: -0.0448264, 0.1128445, -0.0682687, 0.1375451, -0.1823715, 0.1811132
3: -0.0140633, 0.0512277, -0.0212437, 0.0693943, -0.0834576, 0.0724713
4: -0.0445105, 0.0516446, -0.0645387, 0.0783611, -0.1228716, 0.1161833
5: -0.0289425, 0.0405034, -0.0508766, 0.0618871, -0.0908296, 0.0913800
6: -0.0959383, 0.0578047, -0.1207161, 0.0804465, -0.1763848, 0.1785208
7: 0.8599609, 1.0058222, 0.8270352, 1.0184215, -0.1584606, 0.1787869
8: -0.0522475, 0.0892382, -0.0542353, 0.1171868, -0.1663229, 0.1434735
9: -0.0740007, 0.0446140, -0.0914230, 0.0836750, -0.1576756, 0.1360370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0640337, 0.0187563, -0.0684047, 0.0223800, -0.0864137, 0.0871610
1: -0.0577126, 0.0429261, -0.0635196, 0.0454453, -0.1031578, 0.1064457
2: -0.0628379, 0.1314158, -0.0682687, 0.1375451, -0.2003831, 0.1996845
3: -0.0138287, 0.0665909, -0.0212437, 0.0693943, -0.0832230, 0.0878345
4: -0.0562897, 0.0708906, -0.0645387, 0.0783611, -0.1346508, 0.1354293
5: -0.0460152, 0.0544039, -0.0508766, 0.0618871, -0.1079023, 0.1052805
6: -0.1136869, 0.0747396, -0.1207161, 0.0804465, -0.1941334, 0.1954557
7: 0.8352613, 1.0151875, 0.8270352, 1.0184215, -0.1831602, 0.1881523
8: -0.0514336, 0.1069937, -0.0542353, 0.1171868, -0.1686203, 0.1612290
9: -0.0880593, 0.0666374, -0.0914230, 0.0836750, -0.1717342, 0.1580605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0425417, 0.0160595, -0.0431453, 0.0165977, -0.0591394, 0.0592048
1: -0.0383231, 0.0256942, -0.0387860, 0.0262140, -0.0645370, 0.0644802
2: -0.0403601, 0.1082357, -0.0410348, 0.1089349, -0.1492950, 0.1492705
3: -0.0141247, 0.0476532, -0.0145278, 0.0479935, -0.0621182, 0.0621810
4: -0.0416090, 0.0468685, -0.0420308, 0.0475930, -0.0892021, 0.0888993
5: -0.0247881, 0.0370539, -0.0253484, 0.0375772, -0.0623653, 0.0624023
6: -0.0915337, 0.0536571, -0.0922019, 0.0542396, -0.1457733, 0.1458590
7: 0.8660905, 1.0035528, 0.8651605, 1.0038503, -0.1377599, 0.1383923
8: -0.0524605, 0.0848320, -0.0538593, 0.0855004, -0.1322848, 0.1343597
9: -0.0705512, 0.0391486, -0.0710412, 0.0399777, -0.1105290, 0.1101898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0557481, 0.0177518, -0.0431453, 0.0165977, -0.0723458, 0.0608971
1: -0.0508759, 0.0359587, -0.0387860, 0.0262140, -0.0770898, 0.0747447
2: -0.0544898, 0.1230369, -0.0410348, 0.1089349, -0.1634247, 0.1640717
3: -0.0153923, 0.0588375, -0.0145278, 0.0479935, -0.0633858, 0.0733653
4: -0.0525562, 0.0625473, -0.0420308, 0.0475930, -0.1001493, 0.1045781
5: -0.0380321, 0.0491001, -0.0253484, 0.0375772, -0.0756093, 0.0744486
6: -0.1060411, 0.0671416, -0.0922019, 0.0542396, -0.1602807, 0.1593435
7: 0.8463782, 1.0109885, 0.8651605, 1.0038503, -0.1574721, 0.1458280
8: -0.0568588, 0.1004386, -0.0538593, 0.0855004, -0.1400537, 0.1500549
9: -0.0812405, 0.0602423, -0.0710412, 0.0399777, -0.1212183, 0.1312835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0425417, 0.0160595, -0.0523474, 0.0178621, -0.0604038, 0.0684068
1: -0.0383231, 0.0256942, -0.0474785, 0.0334098, -0.0717328, 0.0731727
2: -0.0403601, 0.1082357, -0.0507876, 0.1191386, -0.1594987, 0.1590233
3: -0.0141247, 0.0476532, -0.0154749, 0.0560009, -0.0701256, 0.0631280
4: -0.0416090, 0.0468685, -0.0493366, 0.0582982, -0.0999072, 0.0962051
5: -0.0247881, 0.0370539, -0.0345808, 0.0456644, -0.0704525, 0.0716347
6: -0.0915337, 0.0536571, -0.1020980, 0.0635667, -0.1551004, 0.1557552
7: 0.8660905, 1.0035528, 0.8515756, 1.0089915, -0.1429010, 0.1519772
8: -0.0524605, 0.0848320, -0.0571453, 0.0959384, -0.1427237, 0.1401085
9: -0.0705512, 0.0391486, -0.0785046, 0.0539461, -0.1244973, 0.1176532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0557481, 0.0177518, -0.0523474, 0.0178621, -0.0736102, 0.0700992
1: -0.0508759, 0.0359587, -0.0474785, 0.0334098, -0.0842856, 0.0834372
2: -0.0544898, 0.1230369, -0.0507876, 0.1191386, -0.1736284, 0.1738245
3: -0.0153923, 0.0588375, -0.0154749, 0.0560009, -0.0713932, 0.0743123
4: -0.0525562, 0.0625473, -0.0493366, 0.0582982, -0.1108544, 0.1118840
5: -0.0380321, 0.0491001, -0.0345808, 0.0456644, -0.0836965, 0.0836809
6: -0.1060411, 0.0671416, -0.1020980, 0.0635667, -0.1696078, 0.1692397
7: 0.8463782, 1.0109885, 0.8515756, 1.0089915, -0.1626133, 0.1594129
8: -0.0568588, 0.1004386, -0.0571453, 0.0959384, -0.1504926, 0.1558037
9: -0.0812405, 0.0602423, -0.0785046, 0.0539461, -0.1351866, 0.1387469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0589021, 0.0186486, -0.0431453, 0.0165977, -0.0754998, 0.0617939
1: -0.0540266, 0.0383227, -0.0387860, 0.0262140, -0.0802406, 0.0771087
2: -0.0579235, 0.1266523, -0.0410348, 0.1089349, -0.1668584, 0.1676871
3: -0.0138563, 0.0614683, -0.0145278, 0.0479935, -0.0618498, 0.0759961
4: -0.0555422, 0.0664881, -0.0420308, 0.0475930, -0.1031352, 0.1085189
5: -0.0412329, 0.0522867, -0.0253484, 0.0375772, -0.0788101, 0.0776351
6: -0.1096981, 0.0704571, -0.0922019, 0.0542396, -0.1639377, 0.1626591
7: 0.8415580, 1.0128410, 0.8651605, 1.0038503, -0.1622924, 0.1476805
8: -0.0515294, 0.1046121, -0.0538593, 0.0855004, -0.1370298, 0.1542139
9: -0.0837780, 0.0660817, -0.0710412, 0.0399777, -0.1237558, 0.1371229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0731237, 0.0242328, -0.0431453, 0.0165977, -0.0897214, 0.0673781
1: -0.0682337, 0.0489822, -0.0387860, 0.0262140, -0.0944477, 0.0877682
2: -0.0734059, 0.1429543, -0.0410348, 0.1089349, -0.1823408, 0.1839890
3: -0.0252167, 0.0733301, -0.0145278, 0.0479935, -0.0732102, 0.0878579
4: -0.0690062, 0.0842572, -0.0420308, 0.0475930, -0.1165992, 0.1262880
5: -0.0556655, 0.0666546, -0.0253484, 0.0375772, -0.0932427, 0.0920030
6: -0.1261875, 0.0854070, -0.0922019, 0.0542396, -0.1804272, 0.1776089
7: 0.8198234, 1.0211928, 0.8651605, 1.0038503, -0.1840270, 0.1560323
8: -0.0558497, 0.1234311, -0.0538593, 0.0855004, -0.1413502, 0.1727438
9: -0.0952195, 0.0924115, -0.0710412, 0.0399777, -0.1351973, 0.1634527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0589021, 0.0186486, -0.0523474, 0.0178621, -0.0767642, 0.0709960
1: -0.0540266, 0.0383227, -0.0474785, 0.0334098, -0.0874364, 0.0858012
2: -0.0579235, 0.1266523, -0.0507876, 0.1191386, -0.1770621, 0.1774399
3: -0.0138563, 0.0614683, -0.0154749, 0.0560009, -0.0698573, 0.0769431
4: -0.0555422, 0.0664881, -0.0493366, 0.0582982, -0.1138404, 0.1158247
5: -0.0412329, 0.0522867, -0.0345808, 0.0456644, -0.0868973, 0.0868675
6: -0.1096981, 0.0704571, -0.1020980, 0.0635667, -0.1732648, 0.1725552
7: 0.8415580, 1.0128410, 0.8515756, 1.0089915, -0.1674335, 0.1612654
8: -0.0515294, 0.1046121, -0.0571453, 0.0959384, -0.1474678, 0.1599627
9: -0.0837780, 0.0660817, -0.0785046, 0.0539461, -0.1377241, 0.1445863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0731237, 0.0242328, -0.0523474, 0.0178621, -0.0909858, 0.0765802
1: -0.0682337, 0.0489822, -0.0474785, 0.0334098, -0.1016435, 0.0964607
2: -0.0734059, 0.1429543, -0.0507876, 0.1191386, -0.1925445, 0.1937418
3: -0.0252167, 0.0733301, -0.0154749, 0.0560009, -0.0812176, 0.0888050
4: -0.0690062, 0.0842572, -0.0493366, 0.0582982, -0.1273043, 0.1335938
5: -0.0556655, 0.0666546, -0.0345808, 0.0456644, -0.1013299, 0.1012354
6: -0.1261875, 0.0854070, -0.1020980, 0.0635667, -0.1897542, 0.1875050
7: 0.8198234, 1.0211928, 0.8515756, 1.0089915, -0.1891681, 0.1696172
8: -0.0558497, 0.1234311, -0.0571453, 0.0959384, -0.1517881, 0.1784925
9: -0.0952195, 0.0924115, -0.0785046, 0.0539461, -0.1491656, 0.1709161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0425417, 0.0160595, -0.0596262, 0.0189330, -0.0614747, 0.0756856
1: -0.0383231, 0.0256942, -0.0547500, 0.0388654, -0.0771885, 0.0804442
2: -0.0403601, 0.1082357, -0.0587118, 0.1274822, -0.1678423, 0.1669476
3: -0.0141247, 0.0476532, -0.0142801, 0.0620721, -0.0761968, 0.0619333
4: -0.0416090, 0.0468685, -0.0562277, 0.0673928, -0.1090018, 0.1030962
5: -0.0247881, 0.0370539, -0.0419677, 0.0530182, -0.0778063, 0.0790216
6: -0.0915337, 0.0536571, -0.1105376, 0.0712183, -0.1627520, 0.1641947
7: 0.8660905, 1.0035528, 0.8404516, 1.0132661, -0.1471756, 0.1631012
8: -0.0524605, 0.0848320, -0.0529999, 0.1055703, -0.1525827, 0.1378319
9: -0.0705512, 0.0391486, -0.0843605, 0.0674222, -0.1379734, 0.1235091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0569855, 0.0172026, -0.0596262, 0.0189330, -0.0759184, 0.0768288
1: -0.0489622, 0.0383227, -0.0547500, 0.0388654, -0.0878277, 0.0930727
2: -0.0567347, 0.1252235, -0.0587118, 0.1274822, -0.1842169, 0.1839354
3: -0.0138563, 0.0547700, -0.0142801, 0.0620721, -0.0759284, 0.0690501
4: -0.0517511, 0.0644734, -0.0562277, 0.0673928, -0.1191439, 0.1207011
5: -0.0380039, 0.0497691, -0.0419677, 0.0530182, -0.0910220, 0.0917368
6: -0.1077690, 0.0675441, -0.1105376, 0.0712183, -0.1789872, 0.1780817
7: 0.8434970, 1.0105265, 0.8404516, 1.0132661, -0.1697690, 0.1700749
8: -0.0515294, 0.1010734, -0.0529999, 0.1055703, -0.1562738, 0.1540733
9: -0.0822650, 0.0592941, -0.0843605, 0.0674222, -0.1496872, 0.1436546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0425417, 0.0160595, -0.0691534, 0.0226739, -0.0652156, 0.0852129
1: -0.0383231, 0.0256942, -0.0642676, 0.0460064, -0.0843295, 0.0899618
2: -0.0403601, 0.1082357, -0.0690837, 0.1384033, -0.1787634, 0.1773195
3: -0.0141247, 0.0476532, -0.0218740, 0.0700187, -0.0841433, 0.0695272
4: -0.0416090, 0.0468685, -0.0652474, 0.0792966, -0.1209057, 0.1121159
5: -0.0247881, 0.0370539, -0.0516364, 0.0626435, -0.0874316, 0.0886903
6: -0.0915337, 0.0536571, -0.1215842, 0.0812334, -0.1727671, 0.1752413
7: 0.8660905, 1.0035528, 0.8258910, 1.0188612, -0.1527707, 0.1776618
8: -0.0524605, 0.0848320, -0.0561778, 0.1181774, -0.1650231, 0.1410098
9: -0.0705512, 0.0391486, -0.0920254, 0.0850610, -0.1556122, 0.1311740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0569855, 0.0172026, -0.0691534, 0.0226739, -0.0796594, 0.0863561
1: -0.0489622, 0.0383227, -0.0642676, 0.0460064, -0.0949686, 0.1025903
2: -0.0567347, 0.1252235, -0.0690837, 0.1384033, -0.1951380, 0.1943072
3: -0.0138563, 0.0547700, -0.0218740, 0.0700187, -0.0838750, 0.0766440
4: -0.0517511, 0.0644734, -0.0652474, 0.0792966, -0.1310477, 0.1297208
5: -0.0380039, 0.0497691, -0.0516364, 0.0626435, -0.1006474, 0.1014055
6: -0.1077690, 0.0675441, -0.1215842, 0.0812334, -0.1890024, 0.1891283
7: 0.8434970, 1.0105265, 0.8258910, 1.0188612, -0.1753641, 0.1846355
8: -0.0515294, 0.1010734, -0.0561778, 0.1181774, -0.1688161, 0.1572512
9: -0.0822650, 0.0592941, -0.0920254, 0.0850610, -0.1673260, 0.1513195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0557481, 0.0177518, -0.0596262, 0.0189330, -0.0746810, 0.0773780
1: -0.0508759, 0.0359587, -0.0547500, 0.0388654, -0.0897413, 0.0907087
2: -0.0544898, 0.1230369, -0.0587118, 0.1274822, -0.1819720, 0.1817488
3: -0.0153923, 0.0588375, -0.0142801, 0.0620721, -0.0774644, 0.0731176
4: -0.0525562, 0.0625473, -0.0562277, 0.0673928, -0.1199490, 0.1187750
5: -0.0380321, 0.0491001, -0.0419677, 0.0530182, -0.0910503, 0.0910679
6: -0.1060411, 0.0671416, -0.1105376, 0.0712183, -0.1772594, 0.1776792
7: 0.8463782, 1.0109885, 0.8404516, 1.0132661, -0.1668879, 0.1705369
8: -0.0568588, 0.1004386, -0.0529999, 0.1055703, -0.1603516, 0.1534385
9: -0.0812405, 0.0602423, -0.0843605, 0.0674222, -0.1486627, 0.1446029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0731237, 0.0242328, -0.0596262, 0.0189330, -0.0920566, 0.0838590
1: -0.0682337, 0.0489822, -0.0547500, 0.0388654, -0.1070992, 0.1037322
2: -0.0734059, 0.1429543, -0.0587118, 0.1274822, -0.2008881, 0.2016661
3: -0.0252167, 0.0733301, -0.0142801, 0.0620721, -0.0872888, 0.0876103
4: -0.0690062, 0.0842572, -0.0562277, 0.0673928, -0.1363989, 0.1404849
5: -0.0556655, 0.0666546, -0.0419677, 0.0530182, -0.1086837, 0.1086223
6: -0.1261875, 0.0854070, -0.1105376, 0.0712183, -0.1974058, 0.1959446
7: 0.8198234, 1.0211928, 0.8404516, 1.0132661, -0.1934427, 0.1807412
8: -0.0558497, 0.1234311, -0.0529999, 0.1055703, -0.1614200, 0.1764310
9: -0.0952195, 0.0924115, -0.0843605, 0.0674222, -0.1626417, 0.1767720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0557481, 0.0177518, -0.0691534, 0.0226739, -0.0784220, 0.0869053
1: -0.0508759, 0.0359587, -0.0642676, 0.0460064, -0.0968823, 0.1002263
2: -0.0544898, 0.1230369, -0.0690837, 0.1384033, -0.1928931, 0.1921206
3: -0.0153923, 0.0588375, -0.0218740, 0.0700187, -0.0854110, 0.0807115
4: -0.0525562, 0.0625473, -0.0652474, 0.0792966, -0.1318528, 0.1277947
5: -0.0380321, 0.0491001, -0.0516364, 0.0626435, -0.1006756, 0.1007365
6: -0.1060411, 0.0671416, -0.1215842, 0.0812334, -0.1872746, 0.1887258
7: 0.8463782, 1.0109885, 0.8258910, 1.0188612, -0.1724830, 0.1850975
8: -0.0568588, 0.1004386, -0.0561778, 0.1181774, -0.1727919, 0.1566163
9: -0.0812405, 0.0602423, -0.0920254, 0.0850610, -0.1663015, 0.1522677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0731237, 0.0242328, -0.0691534, 0.0226739, -0.0957976, 0.0933863
1: -0.0682337, 0.0489822, -0.0642676, 0.0460064, -0.1142402, 0.1132498
2: -0.0734059, 0.1429543, -0.0690837, 0.1384033, -0.2118092, 0.2120380
3: -0.0252167, 0.0733301, -0.0218740, 0.0700187, -0.0952353, 0.0952041
4: -0.0690062, 0.0842572, -0.0652474, 0.0792966, -0.1483028, 0.1495046
5: -0.0556655, 0.0666546, -0.0516364, 0.0626435, -0.1183090, 0.1182910
6: -0.1261875, 0.0854070, -0.1215842, 0.0812334, -0.2074210, 0.2069912
7: 0.8198234, 1.0211928, 0.8258910, 1.0188612, -0.1990378, 0.1953018
8: -0.0558497, 0.1234311, -0.0561778, 0.1181774, -0.1740271, 0.1796089
9: -0.0952195, 0.0924115, -0.0920254, 0.0850610, -0.1802805, 0.1844369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0354598, 0.0143041, -0.0557359, 0.0174956, -0.0529554, 0.0700400
1: -0.0334864, 0.0186035, -0.0508637, 0.0359496, -0.0694360, 0.0694672
2: -0.0311662, 0.0997398, -0.0544765, 0.1230229, -0.1541890, 0.1542163
3: -0.0128099, 0.0436573, -0.0152004, 0.0588273, -0.0716372, 0.0588576
4: -0.0359145, 0.0395610, -0.0525447, 0.0625320, -0.0984465, 0.0921058
5: -0.0196553, 0.0299146, -0.0380197, 0.0490878, -0.0687431, 0.0679343
6: -0.0835187, 0.0458598, -0.1060269, 0.0671288, -0.1506476, 0.1518867
7: 0.8787041, 0.9996369, 0.8463969, 1.0109816, -0.1322774, 0.1532400
8: -0.0478985, 0.0768380, -0.0561928, 0.1004224, -0.1433448, 0.1322058
9: -0.0639742, 0.0300489, -0.0812308, 0.0602198, -0.1241940, 0.1112797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0467779, 0.0159775, -0.0557359, 0.0174956, -0.0642734, 0.0717134
1: -0.0420774, 0.0291203, -0.0508637, 0.0359496, -0.0780270, 0.0799840
2: -0.0448264, 0.1128445, -0.0544765, 0.1230229, -0.1678493, 0.1673210
3: -0.0140633, 0.0512277, -0.0152004, 0.0588273, -0.0728906, 0.0664280
4: -0.0445105, 0.0516446, -0.0525447, 0.0625320, -0.1070425, 0.1041893
5: -0.0289425, 0.0405034, -0.0380197, 0.0490878, -0.0780302, 0.0785232
6: -0.0959383, 0.0578047, -0.1060269, 0.0671288, -0.1630671, 0.1638316
7: 0.8599609, 1.0058222, 0.8463969, 1.0109816, -0.1510206, 0.1594253
8: -0.0522475, 0.0892382, -0.0561928, 0.1004224, -0.1480134, 0.1433951
9: -0.0740007, 0.0446140, -0.0812308, 0.0602198, -0.1342205, 0.1258448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0354598, 0.0143041, -0.0641420, 0.0207062, -0.0561660, 0.0784461
1: -0.0334864, 0.0186035, -0.0592612, 0.0422503, -0.0757366, 0.0778648
2: -0.0311662, 0.0997398, -0.0636280, 0.1326587, -0.1638249, 0.1633679
3: -0.0128099, 0.0436573, -0.0176547, 0.0658388, -0.0786486, 0.0613120
4: -0.0359145, 0.0395610, -0.0605030, 0.0730351, -0.1089496, 0.1000641
5: -0.0196553, 0.0299146, -0.0465506, 0.0575805, -0.0772358, 0.0764651
6: -0.0835187, 0.0458598, -0.1157736, 0.0759654, -0.1594842, 0.1616334
7: 0.8787041, 0.9996369, 0.8335499, 1.0159180, -0.1372139, 0.1660871
8: -0.0478985, 0.0768380, -0.0592139, 0.1115461, -0.1543378, 0.1360519
9: -0.0639742, 0.0300489, -0.0879936, 0.0757828, -0.1397570, 0.1180425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.1474390, upper bound: 0.1474390
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0467779, 0.0159775, -0.0641420, 0.0207062, -0.0674840, 0.0801196
1: -0.0420774, 0.0291203, -0.0592612, 0.0422503, -0.0843276, 0.0883815
2: -0.0448264, 0.1128445, -0.0636280, 0.1326587, -0.1774851, 0.1764725
3: -0.0140633, 0.0512277, -0.0176547, 0.0658388, -0.0799021, 0.0688823
4: -0.0445105, 0.0516446, -0.0605030, 0.0730351, -0.1175456, 0.1121476
5: -0.0289425, 0.0405034, -0.0465506, 0.0575805, -0.0865229, 0.0870540
6: -0.0959383, 0.0578047, -0.1157736, 0.0759654, -0.1719037, 0.1735783
7: 0.8599609, 1.0058222, 0.8335499, 1.0159180, -0.1559571, 0.1722723
8: -0.0522475, 0.0892382, -0.0592139, 0.1115461, -0.1591592, 0.1484521
9: -0.0740007, 0.0446140, -0.0879936, 0.0757828, -0.1497835, 0.1326076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.35 + 596.66 = 600.01 seconds
