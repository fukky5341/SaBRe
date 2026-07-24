## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.25792804


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=37, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1242590, 0.1150069, -0.1242590, 0.1150069, -0.2392659, 0.2392659)
1: (-0.1340886, 0.1769537, -0.1340886, 0.1769537, -0.3110423, 0.3110423)
2: (-0.1403643, 0.2423417, -0.1403643, 0.2423417, -0.3827060, 0.3827060)
3: (-0.1022411, 0.1108306, -0.1022411, 0.1108306, -0.2130718, 0.2130718)
4: (-0.1814714, 0.1541879, -0.1814714, 0.1541879, -0.3356593, 0.3356593)
5: (-0.1517652, 0.1826258, -0.1517652, 0.1826258, -0.3343910, 0.3343910)
6: (0.6989413, 1.0502478, 0.6989413, 1.0502478, -0.3513064, 0.3513064)
7: (-0.1971259, 0.1781816, -0.1971259, 0.1781816, -0.3753075, 0.3753075)
8: (-0.1327113, 0.2121396, -0.1327113, 0.2121396, -0.3448508, 0.3448508)
9: (-0.1669101, 0.1474919, -0.1669101, 0.1474919, -0.3144020, 0.3144020)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 2.48 = 4.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.44 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.44
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.44
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0637783, 0.0659338, -0.0971037, 0.0933120, -0.1570902, 0.1630375
1: -0.0773089, 0.1240145, -0.1087421, 0.1535023, -0.2308113, 0.2327566
2: -0.0954509, 0.1686474, -0.1201215, 0.2095875, -0.3050385, 0.2887688
3: -0.0661582, 0.0542805, -0.0861888, 0.0858573, -0.1520156, 0.1404693
4: -0.1131554, 0.1061503, -0.1508966, 0.1324911, -0.2456464, 0.2570469
5: -0.1052207, 0.1077879, -0.1312080, 0.1488848, -0.2541055, 0.2389960
6: 0.7864713, 1.0312374, 0.7378104, 1.0416334, -0.2551621, 0.2934270
7: -0.1344888, 0.1088593, -0.1690915, 0.1472564, -0.2817452, 0.2779508
8: -0.0851941, 0.1408501, -0.1115437, 0.1807353, -0.2659294, 0.2523938
9: -0.0972313, 0.0852147, -0.1360570, 0.1190022, -0.2162335, 0.2212717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
time: 1.05 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
time: 1.11 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0914818, 0.0888203, -0.1166100, 0.1088960, -0.2003778, 0.2054304
1: -0.1034946, 0.1486473, -0.1269491, 0.1703479, -0.2738425, 0.2755964
2: -0.1159306, 0.2028070, -0.1346624, 0.2331155, -0.3490461, 0.3374695
3: -0.0828654, 0.0806871, -0.0977196, 0.1037962, -0.1866615, 0.1784067
4: -0.1445676, 0.1279991, -0.1728592, 0.1480765, -0.2926441, 0.3008584
5: -0.1269520, 0.1418995, -0.1459747, 0.1731219, -0.3000739, 0.2878742
6: 0.7458577, 1.0398504, 0.7098899, 1.0478213, -0.3019636, 0.3299605
7: -0.1632873, 0.1408554, -0.1892293, 0.1694707, -0.3327579, 0.3300847
8: -0.1071613, 0.1742334, -0.1267489, 0.2032937, -0.3104549, 0.3009823
9: -0.1296698, 0.1131040, -0.1582194, 0.1394672, -0.2691370, 0.2713234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=37, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.02 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.72 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0548440, 0.0584827, -0.0588742, 0.0611562, -0.1160002, 0.1173569
1: -0.0683818, 0.1157825, -0.0721605, 0.1194633, -0.1878451, 0.1879431
2: -0.0883959, 0.1584250, -0.0916033, 0.1627542, -0.2511501, 0.2500283
3: -0.0620825, 0.0429692, -0.0635679, 0.0481114, -0.1101939, 0.1065371
4: -0.1043141, 0.0988804, -0.1082717, 0.1021420, -0.2064560, 0.2071521
5: -0.0965688, 0.0979839, -0.1005021, 0.1018595, -0.1984283, 0.1984860
6: 0.8009572, 1.0285320, 0.7947664, 1.0297619, -0.2288047, 0.2337656
7: -0.1253488, 0.0986705, -0.1292153, 0.1032091, -0.2285579, 0.2278858
8: -0.0797574, 0.1265506, -0.0819458, 0.1330513, -0.2128087, 0.2084964
9: -0.0859598, 0.0807615, -0.0908869, 0.0827859, -0.1687457, 0.1716484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
time: 1.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0483570, 0.0546695, -0.0625864, 0.0648246, -0.1131815, 0.1172559
1: -0.0624221, 0.1094886, -0.0761136, 0.1228534, -0.1852755, 0.1856021
2: -0.0831924, 0.1512772, -0.0945574, 0.1671471, -0.2503396, 0.2458346
3: -0.0599036, 0.0345217, -0.0655101, 0.0528480, -0.1127516, 0.1000318
4: -0.0975462, 0.0935123, -0.1119173, 0.1052195, -0.2027657, 0.2054296
5: -0.0902207, 0.0919494, -0.1041250, 0.1063719, -0.1965927, 0.1960744
6: 0.8109908, 1.0264288, 0.7884457, 1.0308948, -0.2199039, 0.2379831
7: -0.1190437, 0.0911068, -0.1332643, 0.1073895, -0.2264333, 0.2243711
8: -0.0762916, 0.1164896, -0.0844003, 0.1390393, -0.2153309, 0.2008899
9: -0.0782922, 0.0772996, -0.0956513, 0.0846508, -0.1629430, 0.1729508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
time: 1.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0789339, 0.0787589, -0.0753898, 0.0759171, -0.1548509, 0.1541487
1: -0.0917397, 0.1379006, -0.0884195, 0.1348653, -0.2266049, 0.2263201
2: -0.1065426, 0.1878283, -0.1038910, 0.1835977, -0.2901402, 0.2917193
3: -0.0754207, 0.0691666, -0.0733180, 0.0659126, -0.1413333, 0.1424847
4: -0.1306946, 0.1179367, -0.1267762, 0.1150946, -0.2457893, 0.2447129
5: -0.1174229, 0.1262514, -0.1147314, 0.1218317, -0.2392546, 0.2409828
6: 0.7638839, 1.0360202, 0.7689752, 1.0349386, -0.2710546, 0.2670450
7: -0.1502858, 0.1268964, -0.1466136, 0.1229538, -0.2732396, 0.2735099
8: -0.0973723, 0.1596692, -0.0946074, 0.1555555, -0.2529278, 0.2542767
9: -0.1155177, 0.0998912, -0.1115205, 0.0961593, -0.2116770, 0.2114117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0695791, 0.0712575, -0.0799284, 0.0795562, -0.1491352, 0.1511859
1: -0.0829759, 0.1298885, -0.0926712, 0.1387524, -0.2217284, 0.2225597
2: -0.0995435, 0.1766610, -0.1072866, 0.1890153, -0.2885588, 0.2839476
3: -0.0698705, 0.0605775, -0.0760107, 0.0700796, -0.1399501, 0.1365882
4: -0.1203517, 0.1104349, -0.1317939, 0.1187342, -0.2390859, 0.2422289
5: -0.1103185, 0.1145853, -0.1181780, 0.1274914, -0.2378100, 0.2327633
6: 0.7773232, 1.0331650, 0.7624553, 1.0363238, -0.2590005, 0.2707096
7: -0.1405927, 0.1164894, -0.1513162, 0.1280025, -0.2685952, 0.2678056
8: -0.0900743, 0.1488108, -0.0981481, 0.1608233, -0.2508976, 0.2469589
9: -0.1049667, 0.0900409, -0.1166392, 0.1009383, -0.2059050, 0.2066801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=36, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=47, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.04 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0426818, 0.0505992, -0.0537749, 0.0578911, -0.1005729, 0.1043741
1: -0.0566866, 0.1028137, -0.0674255, 0.1148061, -0.1714927, 0.1702392
2: -0.0784335, 0.1437032, -0.0875451, 0.1573162, -0.2357497, 0.2312483
3: -0.0576174, 0.0264762, -0.0617445, 0.0416050, -0.0992224, 0.0882207
4: -0.0904130, 0.0882833, -0.1032641, 0.0980223, -0.1884353, 0.1915474
5: -0.0844562, 0.0856324, -0.0955253, 0.0970477, -0.1815039, 0.1811577
6: 0.8212293, 1.0242541, 0.8025392, 1.0282059, -0.2069766, 0.2217149
7: -0.1123136, 0.0837466, -0.1243707, 0.0974665, -0.2097801, 0.2081172
8: -0.0726540, 0.1086479, -0.0792198, 0.1248259, -0.1974800, 0.1878677
9: -0.0718163, 0.0736042, -0.0846747, 0.0802244, -0.1520407, 0.1582789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740668, upper bound: 0.2748985
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0472149, 0.0538737, -0.0559251, 0.0590809, -0.1062958, 0.1097988
1: -0.0612854, 0.1081751, -0.0693489, 0.1167699, -0.1780553, 0.1775239
2: -0.0822534, 0.1497854, -0.0892563, 0.1595464, -0.2417998, 0.2390417
3: -0.0594488, 0.0329397, -0.0624243, 0.0443486, -0.1037974, 0.0953640
4: -0.0961337, 0.0924773, -0.1053758, 0.0997481, -0.1958817, 0.1978531
5: -0.0890937, 0.0906899, -0.0976239, 0.0989305, -0.1880242, 0.1883138
6: 0.8129995, 1.0259901, 0.7993574, 1.0288620, -0.2158625, 0.2266327
7: -0.1177278, 0.0896315, -0.1263379, 0.0998880, -0.2176159, 0.2159693
8: -0.0755682, 0.1149423, -0.0803011, 0.1282944, -0.2038627, 0.1952434
9: -0.0770141, 0.0765770, -0.0872591, 0.0813046, -0.1583187, 0.1638361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740609, upper bound: 0.2748491
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0387613, 0.0469773, -0.0576091, 0.0600126, -0.0987739, 0.1045864
1: -0.0524787, 0.0969851, -0.0708551, 0.1183079, -0.1707866, 0.1678403
2: -0.0747005, 0.1371064, -0.0905965, 0.1612929, -0.2359934, 0.2277029
3: -0.0560402, 0.0194346, -0.0629567, 0.0464973, -0.1025375, 0.0823913
4: -0.0842987, 0.0843731, -0.1070294, 0.1010996, -0.1853983, 0.1914025
5: -0.0793264, 0.0813723, -0.0992674, 0.1004050, -0.1797314, 0.1806398
6: 0.8299373, 1.0224985, 0.7968661, 1.0293760, -0.1994387, 0.2256324
7: -0.1063245, 0.0791341, -0.1278785, 0.1017844, -0.2081089, 0.2070126
8: -0.0701318, 0.1018552, -0.0811481, 0.1310108, -0.2011426, 0.1830033
9: -0.0665116, 0.0703157, -0.0892832, 0.0821505, -0.1486621, 0.1595989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740295, upper bound: 0.2747373
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0421282, 0.0500878, -0.0597398, 0.0620117, -0.1041399, 0.1098276
1: -0.0560923, 0.1019905, -0.0730824, 0.1202538, -0.1763461, 0.1750729
2: -0.0779063, 0.1427717, -0.0922921, 0.1637786, -0.2416849, 0.2350638
3: -0.0573947, 0.0254817, -0.0640208, 0.0492159, -0.1066106, 0.0895026
4: -0.0895496, 0.0877311, -0.1091218, 0.1028596, -0.1924092, 0.1968529
5: -0.0837317, 0.0850309, -0.1013469, 0.1029118, -0.1866435, 0.1863777
6: 0.8224589, 1.0240062, 0.7932926, 1.0300261, -0.2075672, 0.2307137
7: -0.1114678, 0.0830952, -0.1301595, 0.1041840, -0.2156518, 0.2132547
8: -0.0722978, 0.1076887, -0.0825182, 0.1344476, -0.2067454, 0.1902069
9: -0.0710672, 0.0731398, -0.0919979, 0.0832208, -0.1542880, 0.1651377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740193, upper bound: 0.2746771
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0657141, 0.0676484, -0.0681750, 0.0700510, -0.1357650, 0.1358234
1: -0.0791416, 0.1260333, -0.0816268, 0.1285996, -0.2077413, 0.2076602
2: -0.0968007, 0.1712877, -0.0985167, 0.1748646, -0.2716653, 0.2698044
3: -0.0672983, 0.0564446, -0.0689933, 0.0591959, -0.1264942, 0.1254379
4: -0.1153749, 0.1075564, -0.1186879, 0.1093441, -0.2247190, 0.2262443
5: -0.1069001, 0.1100282, -0.1091757, 0.1128761, -0.2197762, 0.2192039
6: 0.7833655, 1.0317907, 0.7794176, 1.0327057, -0.2493402, 0.2523731
7: -0.1363900, 0.1114817, -0.1391067, 0.1148152, -0.2512052, 0.2505883
8: -0.0865874, 0.1435861, -0.0889043, 0.1470643, -0.2336516, 0.2324904
9: -0.0998898, 0.0862460, -0.1032695, 0.0886060, -0.1884958, 0.1895155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0680671, 0.0699456, -0.0712034, 0.0725601, -0.1406273, 0.1411490
1: -0.0815178, 0.1284872, -0.0844977, 0.1312798, -0.2127976, 0.2129849
2: -0.0984414, 0.1747078, -0.1007589, 0.1786001, -0.2770416, 0.2754667
3: -0.0689189, 0.0590753, -0.0708343, 0.0620689, -0.1309879, 0.1299096
4: -0.1185426, 0.1092657, -0.1221475, 0.1117376, -0.2302802, 0.2314132
5: -0.1090760, 0.1127511, -0.1115521, 0.1166110, -0.2256870, 0.2243033
6: 0.7795908, 1.0326655, 0.7749894, 1.0336607, -0.2540698, 0.2576761
7: -0.1389874, 0.1146691, -0.1422758, 0.1182966, -0.2572840, 0.2569449
8: -0.0888026, 0.1469118, -0.0913415, 0.1506963, -0.2394989, 0.2382533
9: -0.1031214, 0.0885024, -0.1067989, 0.0917511, -0.1948725, 0.1953012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0589142, 0.0611957, -0.0727995, 0.0738399, -0.1327541, 0.1339951
1: -0.0722031, 0.1194997, -0.0859928, 0.1326465, -0.2048497, 0.2054926
2: -0.0916350, 0.1628015, -0.1019529, 0.1805054, -0.2721404, 0.2647544
3: -0.0635888, 0.0481624, -0.0717812, 0.0635341, -0.1271229, 0.1199436
4: -0.1083111, 0.1021751, -0.1239122, 0.1130174, -0.2213285, 0.2260873
5: -0.1005411, 0.1019081, -0.1127640, 0.1186015, -0.2191426, 0.2146721
6: 0.7946984, 1.0297740, 0.7726967, 1.0341480, -0.2394496, 0.2570773
7: -0.1292589, 0.1032541, -0.1439295, 0.1200720, -0.2493308, 0.2471836
8: -0.0819722, 0.1331159, -0.0925865, 0.1525488, -0.2345210, 0.2257024
9: -0.0909382, 0.0828061, -0.1085988, 0.0934318, -0.1843700, 0.1914049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
time: 1.39 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0608892, 0.0631474, -0.0758809, 0.0763108, -0.1372000, 0.1390283
1: -0.0743062, 0.1213035, -0.0888795, 0.1352859, -0.2095922, 0.2101830
2: -0.0932068, 0.1651386, -0.1042584, 0.1841838, -0.2773906, 0.2693970
3: -0.0646221, 0.0506825, -0.0736094, 0.0663635, -0.1309856, 0.1242919
4: -0.1102505, 0.1038124, -0.1273191, 0.1154885, -0.2257391, 0.2311316
5: -0.1024686, 0.1043088, -0.1151043, 0.1224441, -0.2249127, 0.2194132
6: 0.7913356, 1.0303769, 0.7682698, 1.0350884, -0.2437528, 0.2621071
7: -0.1314132, 0.1054782, -0.1471224, 0.1235000, -0.2549132, 0.2526006
8: -0.0832781, 0.1363016, -0.0949906, 0.1561255, -0.2394036, 0.2312922
9: -0.0934730, 0.0837982, -0.1120744, 0.0966765, -0.1901495, 0.1958726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=36, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2740668, upper bound: 0.2748985
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2740609, upper bound: 0.2748491
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2740295, upper bound: 0.2747373
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2750912, upper bound: 0.2753898
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2740193, upper bound: 0.2746771
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2750860, upper bound: 0.2753898
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0405806, 0.0486579, -0.0962424, 0.0813882, -0.1219688, 0.1449003
1: -0.0544313, 0.0996896, -0.1054125, 0.1535907, -0.2080219, 0.2051021
2: -0.0764327, 0.1401676, -0.1213414, 0.2013621, -0.2777947, 0.2615089
3: -0.0567721, 0.0227021, -0.0751713, 0.0957910, -0.1525631, 0.0978734
4: -0.0871359, 0.0861875, -0.1449685, 0.1321057, -0.2192415, 0.2311559
5: -0.0817067, 0.0833491, -0.1369718, 0.1342328, -0.2159395, 0.2203210
6: 0.8258967, 1.0233129, 0.7397041, 1.0411654, -0.2152686, 0.2836088
7: -0.1091035, 0.0812744, -0.1632233, 0.1452913, -0.2543948, 0.2444978
8: -0.0713021, 0.1050072, -0.1005770, 0.1933278, -0.2646300, 0.2055841
9: -0.0689730, 0.0718417, -0.1357160, 0.1015574, -0.1705304, 0.2075577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740668, upper bound: 0.2748985
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740668, upper bound: 0.2748985
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0426818, 0.0505992, -0.0522271, 0.0570348, -0.0997166, 0.1028263
1: -0.0566866, 0.1028137, -0.0660409, 0.1133926, -0.1700791, 0.1688546
2: -0.0784335, 0.1437032, -0.0863134, 0.1557109, -0.2341443, 0.2300166
3: -0.0576174, 0.0264762, -0.0612551, 0.0396301, -0.0972475, 0.0877313
4: -0.0904130, 0.0882833, -0.1017441, 0.0967801, -0.1871931, 0.1900274
5: -0.0844562, 0.0856324, -0.0940148, 0.0956924, -0.1801486, 0.1796472
6: 0.8212293, 1.0242541, 0.8048292, 1.0277333, -0.2065040, 0.2194248
7: -0.1123136, 0.0837466, -0.1229546, 0.0957234, -0.2080369, 0.2067011
8: -0.0726540, 0.1086479, -0.0784414, 0.1223294, -0.1949834, 0.1870893
9: -0.0718163, 0.0736042, -0.0828145, 0.0794469, -0.1512631, 0.1564187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0444424, 0.0519418, -0.0984566, 0.0826133, -0.1270557, 0.1503984
1: -0.0585262, 0.1049860, -0.1073931, 0.1556129, -0.2141391, 0.2123792
2: -0.0799739, 0.1461640, -0.1231035, 0.2036586, -0.2836325, 0.2692676
3: -0.0583449, 0.0290990, -0.0758713, 0.0986163, -0.1569612, 0.1049703
4: -0.0927047, 0.0899647, -0.1471429, 0.1338828, -0.2265875, 0.2371077
5: -0.0863575, 0.0876326, -0.1391329, 0.1361716, -0.2225291, 0.2267655
6: 0.8178757, 1.0249244, 0.7364278, 1.0418411, -0.2239654, 0.2884966
7: -0.1145333, 0.0860502, -0.1652492, 0.1477849, -0.2623183, 0.2512994
8: -0.0738123, 0.1111859, -0.1016905, 0.1968995, -0.2707118, 0.2128764
9: -0.0739112, 0.0748230, -0.1383774, 0.1026697, -0.1765809, 0.2132004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740609, upper bound: 0.2748491
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2740609, upper bound: 0.2748491
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0472149, 0.0538737, -0.0543951, 0.0582342, -0.1054491, 0.1082688
1: -0.0612854, 0.1081751, -0.0679802, 0.1153724, -0.1766579, 0.1761552
2: -0.0822534, 0.1497854, -0.0880386, 0.1579594, -0.2402128, 0.2378240
3: -0.0594488, 0.0329397, -0.0619405, 0.0423963, -0.1018451, 0.0948802
4: -0.0961337, 0.0924773, -0.1038731, 0.0985200, -0.1946537, 0.1963504
5: -0.0890937, 0.0906899, -0.0961306, 0.0975907, -0.1866845, 0.1868204
6: 0.8129995, 1.0259901, 0.8016215, 1.0283951, -0.2153956, 0.2243686
7: -0.1177278, 0.0896315, -0.1249380, 0.0981648, -0.2158926, 0.2145694
8: -0.0755682, 0.1149423, -0.0795316, 0.1258262, -0.2013945, 0.1944739
9: -0.0770141, 0.0765770, -0.0854201, 0.0805359, -0.1575500, 0.1619971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0370338, 0.0450145, -0.0951557, 0.0807869, -0.1178208, 0.1401702
1: -0.0505428, 0.0938267, -0.1044405, 0.1525985, -0.2031412, 0.1982672
2: -0.0726776, 0.1338145, -0.1204767, 0.2002350, -0.2729126, 0.2542912
3: -0.0551856, 0.0162168, -0.0748278, 0.0944047, -0.1495903, 0.0910446
4: -0.0814278, 0.0822542, -0.1439014, 0.1312336, -0.2126615, 0.2261556
5: -0.0765465, 0.0796476, -0.1359114, 0.1332814, -0.2098279, 0.2155589
6: 0.8342292, 1.0215472, 0.7413119, 1.0408340, -0.2066047, 0.2802353
7: -0.1036152, 0.0766345, -0.1622293, 0.1440676, -0.2476828, 0.2388638
8: -0.0687649, 0.0986235, -0.1000305, 0.1915750, -0.2603399, 0.1986540
9: -0.0636370, 0.0689867, -0.1344101, 0.1010116, -0.1646487, 0.2033968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2689986, upper bound: 0.2684985
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2684420, upper bound: 0.2683389
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0387613, 0.0469773, -0.0559904, 0.0591170, -0.0978782, 0.1029677
1: -0.0524787, 0.0969851, -0.0694072, 0.1168295, -0.1693082, 0.1663924
2: -0.0747005, 0.1371064, -0.0893082, 0.1596141, -0.2343146, 0.2264146
3: -0.0560402, 0.0194346, -0.0624449, 0.0444319, -0.1004721, 0.0818795
4: -0.0842987, 0.0843731, -0.1054398, 0.0998004, -0.1840991, 0.1898129
5: -0.0793264, 0.0813723, -0.0976876, 0.0989877, -0.1783141, 0.1790599
6: 0.8299373, 1.0224985, 0.7992611, 1.0288818, -0.1989444, 0.2232374
7: -0.1063245, 0.0791341, -0.1263976, 0.0999615, -0.2062860, 0.2055317
8: -0.0701318, 0.1018552, -0.0803340, 0.1283996, -0.1985314, 0.1821892
9: -0.0665116, 0.0703157, -0.0873375, 0.0813373, -0.1478489, 0.1576533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746293
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0400322, 0.0481515, -0.0974092, 0.0992360, -0.1392682, 0.1455608
1: -0.0538427, 0.0988745, -0.1131947, 0.1546565, -0.2084992, 0.2120692
2: -0.0759106, 0.1392449, -0.1222701, 0.2083557, -0.2842663, 0.2615149
3: -0.0565515, 0.0217172, -0.0837290, 0.0972800, -0.1538315, 0.1054462
4: -0.0862808, 0.0856406, -0.1461144, 0.1340893, -0.2203701, 0.2317550
5: -0.0809894, 0.0827532, -0.1381108, 0.1487002, -0.2296896, 0.2208640
6: 0.8271147, 1.0230676, 0.7291526, 1.0415215, -0.2144068, 0.2939150
7: -0.1082659, 0.0806293, -0.1712462, 0.1466054, -0.2548713, 0.2518755
8: -0.0709494, 0.1040573, -0.1074254, 0.1952101, -0.2661595, 0.2114827
9: -0.0682313, 0.0713817, -0.1403459, 0.1021436, -0.1703749, 0.2117276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2689752, upper bound: 0.2684243
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2684331, upper bound: 0.2682534
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0421282, 0.0500878, -0.0581472, 0.0604379, -0.1025661, 0.1082349
1: -0.0560923, 0.1019905, -0.0713864, 0.1187993, -0.1748916, 0.1733769
2: -0.0779063, 0.1427717, -0.0910246, 0.1618939, -0.2398002, 0.2337964
3: -0.0573947, 0.0254817, -0.0631876, 0.0471838, -0.1045785, 0.0886693
4: -0.0895496, 0.0877311, -0.1075578, 0.1015392, -0.1910888, 0.1952889
5: -0.0837317, 0.0850309, -0.0997926, 0.1009759, -0.1847077, 0.1848234
6: 0.8224589, 1.0240062, 0.7960042, 1.0295402, -0.2070813, 0.2280021
7: -0.1114678, 0.0830952, -0.1284224, 0.1023904, -0.2138581, 0.2115176
8: -0.0722978, 0.1076887, -0.0814651, 0.1318786, -0.2041764, 0.1891539
9: -0.0710672, 0.0731398, -0.0899537, 0.0824208, -0.1534880, 0.1630935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2745901
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0657141, 0.0676484, -0.0362156, 0.0437600, -0.1094741, 0.1038639
1: -0.0791416, 0.1260333, -0.0495533, 0.0918080, -0.1709497, 0.1755866
2: -0.0968007, 0.1712877, -0.0713847, 0.1319142, -0.2287150, 0.2426724
3: -0.0672983, 0.0564446, -0.0546394, 0.0145907, -0.0818891, 0.1110840
4: -0.1153749, 0.1075564, -0.0799114, 0.0809000, -0.1962749, 0.1874678
5: -0.1069001, 0.1100282, -0.0747699, 0.0789653, -0.1858654, 0.1847981
6: 0.7833655, 1.0317907, 0.8366649, 1.0209391, -0.2375736, 0.1951258
7: -0.1363900, 0.1114817, -0.1022696, 0.0750370, -0.2114271, 0.2137513
8: -0.0865874, 0.1435861, -0.0678913, 0.0968813, -0.1834687, 0.2114774
9: -0.0998898, 0.0862460, -0.0617998, 0.0684633, -0.1683531, 0.1480458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747119, upper bound: 0.2741956
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0657141, 0.0676484, -0.0495427, 0.0554957, -0.1212097, 0.1171911
1: -0.0791416, 0.1260333, -0.0636021, 0.1108523, -0.1899939, 0.1896354
2: -0.0968007, 0.1712877, -0.0841672, 0.1528259, -0.2496267, 0.2554549
3: -0.0672983, 0.0564446, -0.0603756, 0.0361641, -0.1034624, 0.1168203
4: -0.1153749, 0.1075564, -0.0990125, 0.0945869, -0.2099618, 0.2065689
5: -0.1069001, 0.1100282, -0.0913909, 0.0932569, -0.2001570, 0.2014191
6: 0.7833655, 1.0317907, 0.8089057, 1.0268847, -0.2435192, 0.2228851
7: -0.1363900, 0.1114817, -0.1204099, 0.0926384, -0.2290284, 0.2318915
8: -0.0865874, 0.1435861, -0.0770425, 0.1180960, -0.2046833, 0.2206286
9: -0.0998898, 0.0862460, -0.0796192, 0.0780497, -0.1779395, 0.1658652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747119, upper bound: 0.2753898
time: 1.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0680671, 0.0699456, -0.0369804, 0.0449325, -0.1129996, 0.1069259
1: -0.0815178, 0.1284872, -0.0504781, 0.0936945, -0.1752124, 0.1789653
2: -0.0984414, 0.1747078, -0.0725931, 0.1336904, -0.2321318, 0.2473009
3: -0.0689189, 0.0590753, -0.0551499, 0.0161104, -0.0850293, 0.1142252
4: -0.1185426, 0.1092657, -0.0813287, 0.0821656, -0.2007082, 0.1905944
5: -0.1090760, 0.1127511, -0.0764304, 0.0796030, -0.1886790, 0.1891815
6: 0.7795908, 1.0326655, 0.8343886, 1.0215074, -0.2419165, 0.1982769
7: -0.1389874, 0.1146691, -0.1035272, 0.0765302, -0.2155176, 0.2181963
8: -0.0888026, 0.1469118, -0.0687078, 0.0985096, -0.1873122, 0.2156196
9: -0.1031214, 0.0885024, -0.0635169, 0.0689525, -0.1720739, 0.1520193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747123, upper bound: 0.2742028
time: 1.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0680671, 0.0699456, -0.0517112, 0.0567493, -0.1248164, 0.1216567
1: -0.0815178, 0.1284872, -0.0655795, 0.1129214, -0.1944392, 0.1940667
2: -0.0984414, 0.1747078, -0.0859028, 0.1551757, -0.2536172, 0.2606106
3: -0.0689189, 0.0590753, -0.0610920, 0.0389718, -0.1078907, 0.1201672
4: -0.1185426, 0.1092657, -0.1012374, 0.0963661, -0.2149087, 0.2105031
5: -0.1090760, 0.1127511, -0.0935113, 0.0952406, -0.2043166, 0.2062624
6: 0.7795908, 1.0326655, 0.8055925, 1.0275761, -0.2479852, 0.2270730
7: -0.1389874, 0.1146691, -0.1224826, 0.0951424, -0.2341298, 0.2371517
8: -0.0888026, 0.1469118, -0.0781819, 0.1214971, -0.2102997, 0.2250937
9: -0.1031214, 0.0885024, -0.0821944, 0.0791877, -0.1823091, 0.1706968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747123, upper bound: 0.2753898
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0589142, 0.0611957, -0.0376815, 0.0459797, -0.1048939, 0.0988772
1: -0.0722031, 0.1194997, -0.0513198, 0.0953800, -0.1675831, 0.1708195
2: -0.0916350, 0.1628015, -0.0736724, 0.1352896, -0.2269247, 0.2364739
3: -0.0635888, 0.0481624, -0.0556059, 0.0174952, -0.0810840, 0.1037683
4: -0.1083111, 0.1021751, -0.0826147, 0.0832963, -0.1916073, 0.1847898
5: -0.1005411, 0.1019081, -0.0779137, 0.0801991, -0.1807402, 0.1798217
6: 0.7946984, 1.0297740, 0.8323356, 1.0220150, -0.2273166, 0.1974384
7: -0.1292589, 0.1032541, -0.1046751, 0.0778638, -0.2071227, 0.2079292
8: -0.0819722, 0.1331159, -0.0694371, 0.0999844, -0.1819566, 0.2025530
9: -0.0909382, 0.0828061, -0.0650507, 0.0694101, -0.1603483, 0.1478568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746765, upper bound: 0.2740126
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0589142, 0.0611957, -0.0529782, 0.0574504, -0.1163646, 0.1141739
1: -0.0722031, 0.1194997, -0.0667129, 0.1140786, -0.1862817, 0.1862126
2: -0.0916350, 0.1628015, -0.0869112, 0.1564898, -0.2481249, 0.2497127
3: -0.0635888, 0.0481624, -0.0614926, 0.0405885, -0.1041773, 0.1096550
4: -0.1083111, 0.1021751, -0.1024817, 0.0973830, -0.2056941, 0.2046568
5: -0.1005411, 0.1019081, -0.0947479, 0.0963502, -0.1968913, 0.1966559
6: 0.7946984, 1.0297740, 0.8037177, 1.0279628, -0.2332644, 0.2260562
7: -0.1292589, 0.1032541, -0.1236418, 0.0965694, -0.2258283, 0.2268959
8: -0.0819722, 0.1331159, -0.0788191, 0.1235408, -0.2055130, 0.2119350
9: -0.0909382, 0.0828061, -0.0837173, 0.0798242, -0.1707624, 0.1665233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746765, upper bound: 0.2753898
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0608892, 0.0631474, -0.0389459, 0.0471479, -0.1080371, 0.1020934
1: -0.0743062, 0.1213035, -0.0526769, 0.0972597, -0.1715659, 0.1739804
2: -0.0932068, 0.1651386, -0.0748764, 0.1374172, -0.2306240, 0.2400150
3: -0.0646221, 0.0506825, -0.0561146, 0.0197663, -0.0843884, 0.1067971
4: -0.1102505, 0.1038124, -0.0845867, 0.0845573, -0.1948079, 0.1883991
5: -0.1024686, 0.1043088, -0.0795680, 0.0815730, -0.1840416, 0.1838769
6: 0.7913356, 1.0303769, 0.8295270, 1.0225811, -0.2312455, 0.2008499
7: -0.1314132, 0.1054782, -0.1066066, 0.0793514, -0.2107646, 0.2120848
8: -0.0832781, 0.1363016, -0.0702506, 0.1021753, -0.1854534, 0.2065522
9: -0.0934730, 0.0837982, -0.0667615, 0.0704707, -0.1639437, 0.1505597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2740193
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0608892, 0.0631474, -0.0552457, 0.0587049, -0.1195941, 0.1183931
1: -0.0743062, 0.1213035, -0.0687411, 0.1161494, -0.1904556, 0.1900446
2: -0.0932068, 0.1651386, -0.0887156, 0.1588417, -0.2520485, 0.2538542
3: -0.0646221, 0.0506825, -0.0622095, 0.0434817, -0.1081039, 0.1128920
4: -0.1102505, 0.1038124, -0.1047085, 0.0992028, -0.2094533, 0.2085209
5: -0.1024686, 0.1043088, -0.0969608, 0.0983356, -0.2008042, 0.2012696
6: 0.7913356, 1.0303769, 0.8003628, 1.0286545, -0.2373188, 0.2300141
7: -0.1314132, 0.1054782, -0.1257163, 0.0991228, -0.2305360, 0.2311945
8: -0.0832781, 0.1363016, -0.0799595, 0.1271983, -0.2104764, 0.2162611
9: -0.0934730, 0.0837982, -0.0864424, 0.0809633, -0.1744363, 0.1702406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=35, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2753898
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2740668, upper bound: 0.2748985
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2740668, upper bound: 0.2748985
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2740609, upper bound: 0.2748491
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2740609, upper bound: 0.2748491
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2689986, upper bound: 0.2684985
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2684420, upper bound: 0.2683389
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746293
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2689752, upper bound: 0.2684243
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2684331, upper bound: 0.2682534
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2745901
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2747119, upper bound: 0.2741956
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2747119, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2747123, upper bound: 0.2742028
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753131
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2747123, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2746765, upper bound: 0.2740126
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2746765, upper bound: 0.2753898
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2740193
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2750860
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2753898
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0327411, 0.0373249, -0.0962424, 0.0813882, -0.1141293, 0.1335672
1: -0.0464533, 0.0814526, -0.1054125, 0.1535907, -0.2000440, 0.1868651
2: -0.0648070, 0.1221664, -0.1213414, 0.2013621, -0.2661691, 0.2435078
3: -0.0518373, 0.0099908, -0.0751713, 0.0957910, -0.1476282, 0.0851621
4: -0.0724741, 0.0739529, -0.1449685, 0.1321057, -0.2045798, 0.2189213
5: -0.0669511, 0.0754658, -0.1369718, 0.1342328, -0.2011839, 0.2124376
6: 0.8491593, 1.0186851, 0.7397041, 1.0411654, -0.1920061, 0.2789810
7: -0.0953667, 0.0677014, -0.1632233, 0.1452913, -0.2406579, 0.2309248
8: -0.0635905, 0.0879442, -0.1005770, 0.1933278, -0.2569184, 0.1885212
9: -0.0529933, 0.0657782, -0.1357160, 0.1015574, -0.1545506, 0.2014942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2720479
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2748985
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0339039, 0.0394882, -0.0962424, 0.0813882, -0.1152921, 0.1357306
1: -0.0474813, 0.0849339, -0.1054125, 0.1535907, -0.2010719, 0.1903463
2: -0.0670179, 0.1254434, -0.1213414, 0.2013621, -0.2683800, 0.2467847
3: -0.0527793, 0.0115102, -0.0751713, 0.0957910, -0.1485703, 0.0866815
4: -0.0749720, 0.0762883, -0.1449685, 0.1321057, -0.2070777, 0.2212568
5: -0.0695703, 0.0766423, -0.1369718, 0.1342328, -0.2038030, 0.2136141
6: 0.8449589, 1.0194364, 0.7397041, 1.0411654, -0.1962065, 0.2797322
7: -0.0976873, 0.0701613, -0.1632233, 0.1452913, -0.2429786, 0.2333847
8: -0.0650351, 0.0909487, -0.1005770, 0.1933278, -0.2583629, 0.1915257
9: -0.0559494, 0.0666809, -0.1357160, 0.1015574, -0.1575067, 0.2023969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2720479
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2748985
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0683381, 0.0743022, -0.0522271, 0.0570348, -0.1253729, 0.1265293
1: -0.0842233, 0.1409560, -0.0660409, 0.1133926, -0.1976158, 0.2069969
2: -0.1028623, 0.1868740, -0.0863134, 0.1557109, -0.2585732, 0.2731874
3: -0.0679384, 0.0725577, -0.0612551, 0.0396301, -0.1075685, 0.1338128
4: -0.1304262, 0.1138716, -0.1017441, 0.0967801, -0.2272063, 0.2156158
5: -0.1180261, 0.1135109, -0.0940148, 0.0956924, -0.2137185, 0.2075257
6: 0.7642424, 1.0357430, 0.8048292, 1.0277333, -0.2634910, 0.2309138
7: -0.1515066, 0.1139310, -0.1229546, 0.0957234, -0.2472300, 0.2368856
8: -0.0891601, 0.1531008, -0.0784414, 0.1223294, -0.2114894, 0.2315421
9: -0.1065305, 0.0951241, -0.0828145, 0.0794469, -0.1859774, 0.1779386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0417515, 0.0497397, -0.0522271, 0.0570348, -0.0987862, 0.1019668
1: -0.0556880, 0.1014305, -0.0660409, 0.1133926, -0.1690805, 0.1674714
2: -0.0775476, 0.1421378, -0.0863134, 0.1557109, -0.2332585, 0.2284512
3: -0.0572431, 0.0248052, -0.0612551, 0.0396301, -0.0968732, 0.0860603
4: -0.0889620, 0.0873553, -0.1017441, 0.0967801, -0.1857421, 0.1890995
5: -0.0832388, 0.0846215, -0.0940148, 0.0956924, -0.1789312, 0.1786363
6: 0.8232958, 1.0238374, 0.8048292, 1.0277333, -0.2044375, 0.2190082
7: -0.1108923, 0.0826519, -0.1229546, 0.0957234, -0.2066157, 0.2056065
8: -0.0720555, 0.1070360, -0.0784414, 0.1223294, -0.1943849, 0.1854773
9: -0.0705574, 0.0728238, -0.0828145, 0.0794469, -0.1500043, 0.1556383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753131
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0344318, 0.0404702, -0.0984566, 0.0826133, -0.1170451, 0.1389269
1: -0.0479478, 0.0865144, -0.1073931, 0.1556129, -0.2035608, 0.1939075
2: -0.0680216, 0.1269310, -0.1231035, 0.2036586, -0.2716801, 0.2500345
3: -0.0532069, 0.0122001, -0.0758713, 0.0986163, -0.1518232, 0.0880714
4: -0.0761058, 0.0773486, -0.1471429, 0.1338828, -0.2099886, 0.2244916
5: -0.0707593, 0.0771763, -0.1391329, 0.1361716, -0.2069309, 0.2163092
6: 0.8430520, 1.0197777, 0.7364278, 1.0418411, -0.1987891, 0.2833498
7: -0.0987408, 0.0712780, -0.1652492, 0.1477849, -0.2465257, 0.2365272
8: -0.0656908, 0.0923125, -0.1016905, 0.1968995, -0.2625904, 0.1940030
9: -0.0572914, 0.0670906, -0.1383774, 0.1026697, -0.1599611, 0.2054680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2720479
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2748491
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0355394, 0.0425310, -0.0984566, 0.0826133, -0.1181528, 0.1409876
1: -0.0489270, 0.0898303, -0.1073931, 0.1556129, -0.2045399, 0.1972235
2: -0.0701276, 0.1300525, -0.1231035, 0.2036586, -0.2737862, 0.2531561
3: -0.0541042, 0.0136473, -0.0758713, 0.0986163, -0.1527205, 0.0895186
4: -0.0784851, 0.0795732, -0.1471429, 0.1338828, -0.2123679, 0.2267161
5: -0.0732541, 0.0782970, -0.1391329, 0.1361716, -0.2094258, 0.2174299
6: 0.8390511, 1.0204935, 0.7364278, 1.0418411, -0.2027900, 0.2840657
7: -0.1009513, 0.0736213, -0.1652492, 0.1477849, -0.2487362, 0.2388704
8: -0.0670668, 0.0951745, -0.1016905, 0.1968995, -0.2639664, 0.1968650
9: -0.0601072, 0.0679505, -0.1383774, 0.1026697, -0.1627769, 0.2063279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2720479
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2748491
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0799113, 0.0766567, -0.0543951, 0.0582342, -0.1381455, 0.1310518
1: -0.0938252, 0.1457810, -0.0679802, 0.1153724, -0.2091976, 0.2137612
2: -0.1091351, 0.1924928, -0.0880386, 0.1579594, -0.2670945, 0.2805313
3: -0.0724676, 0.0782313, -0.0619405, 0.0423963, -0.1148639, 0.1401718
4: -0.1365708, 0.1221080, -0.1038731, 0.0985200, -0.2350908, 0.2259811
5: -0.1213607, 0.1267451, -0.0961306, 0.0975907, -0.2189515, 0.2228757
6: 0.7554948, 1.0385557, 0.8016215, 1.0283951, -0.2729002, 0.2369342
7: -0.1553999, 0.1318663, -0.1249380, 0.0981648, -0.2535647, 0.2568043
8: -0.0962765, 0.1592403, -0.0795316, 0.1258262, -0.2221027, 0.2387719
9: -0.1136055, 0.0972618, -0.0854201, 0.0805359, -0.1941414, 0.1826819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0459928, 0.0530220, -0.0543951, 0.0582342, -0.1042270, 0.1074171
1: -0.0600691, 0.1067692, -0.0679802, 0.1153724, -0.1754415, 0.1747494
2: -0.0812485, 0.1481890, -0.0880386, 0.1579594, -0.2392079, 0.2362276
3: -0.0589622, 0.0312465, -0.0619405, 0.0423963, -0.1013584, 0.0931870
4: -0.0946221, 0.0913697, -0.1038731, 0.0985200, -0.1931421, 0.1952428
5: -0.0878875, 0.0893423, -0.0961306, 0.0975907, -0.1854782, 0.1854728
6: 0.8151492, 1.0255202, 0.8016215, 1.0283951, -0.2132459, 0.2238987
7: -0.1163196, 0.0880528, -0.1249380, 0.0981648, -0.2144844, 0.2129907
8: -0.0747941, 0.1132864, -0.0795316, 0.1258262, -0.2006204, 0.1928181
9: -0.0756463, 0.0758038, -0.0854201, 0.0805359, -0.1561822, 0.1612239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753131
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0312568, 0.0344295, -0.0918760, 0.0789724, -0.1102291, 0.1263056
1: -0.0450777, 0.0786921, -0.1015069, 0.1496031, -0.1946808, 0.1801989
2: -0.0624158, 0.1177806, -0.1178666, 0.1968334, -0.2592492, 0.2356472
3: -0.0505766, 0.0091108, -0.0737908, 0.0902199, -0.1407965, 0.0829016
4: -0.0691312, 0.0717474, -0.1406806, 0.1286014, -0.1977327, 0.2124279
5: -0.0640484, 0.0738913, -0.1327105, 0.1304096, -0.1944580, 0.2066018
6: 0.8541625, 1.0176791, 0.7461645, 1.0398331, -0.1856706, 0.2715146
7: -0.0932268, 0.0644093, -0.1592286, 0.1403742, -0.2336009, 0.2236380
8: -0.0620973, 0.0839233, -0.0983812, 0.1862846, -0.2483819, 0.1823045
9: -0.0496844, 0.0645702, -0.1304682, 0.0993640, -0.1490485, 0.1950384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2646901
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2684985
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0298840, 0.0314536, -0.0829602, 0.0740393, -0.1039233, 0.1144138
1: -0.0436637, 0.0769105, -0.0935317, 0.1414606, -0.1851242, 0.1704422
2: -0.0602743, 0.1132726, -0.1107713, 0.1875862, -0.2478606, 0.2240439
3: -0.0492808, 0.0087489, -0.0709719, 0.0788439, -0.1281246, 0.0797208
4: -0.0658121, 0.0699016, -0.1319251, 0.1214458, -0.1872579, 0.2018267
5: -0.0613684, 0.0722729, -0.1240091, 0.1226028, -0.1839712, 0.1962820
6: 0.8590221, 1.0168327, 0.7593564, 1.0371121, -0.1780900, 0.2574763
7: -0.0914694, 0.0612404, -0.1510718, 0.1303337, -0.2218031, 0.2123122
8: -0.0607639, 0.0799528, -0.0938973, 0.1719032, -0.2326671, 0.1738501
9: -0.0468134, 0.0633284, -0.1197525, 0.0948853, -0.1416987, 0.1830809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2646901
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2683389
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0644407, 0.0707015, -0.0559904, 0.0591170, -0.1235577, 0.1266919
1: -0.0800402, 0.1351618, -0.0694072, 0.1168295, -0.1968697, 0.2045690
2: -0.0991513, 0.1803161, -0.0893082, 0.1596141, -0.2587654, 0.2696243
3: -0.0663706, 0.0655575, -0.0624449, 0.0444319, -0.1108025, 0.1280024
4: -0.1243479, 0.1099846, -0.1054398, 0.0998004, -0.2241483, 0.2154244
5: -0.1129265, 0.1092759, -0.0976876, 0.0989877, -0.2119142, 0.2069634
6: 0.7728992, 1.0339979, 0.7992611, 1.0288818, -0.2559826, 0.2347368
7: -0.1455528, 0.1093458, -0.1263976, 0.0999615, -0.2455142, 0.2357434
8: -0.0866527, 0.1463480, -0.0803340, 0.1283996, -0.2150523, 0.2266820
9: -0.1012571, 0.0918550, -0.0873375, 0.0813373, -0.1825944, 0.1791925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2737201
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746293
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0378255, 0.0461127, -0.0559904, 0.0591170, -0.0969425, 0.1021031
1: -0.0514743, 0.0955939, -0.0694072, 0.1168295, -0.1683038, 0.1650012
2: -0.0738095, 0.1355318, -0.0893082, 0.1596141, -0.2334237, 0.2248400
3: -0.0556638, 0.0177538, -0.0624449, 0.0444319, -0.1000957, 0.0801987
4: -0.0828392, 0.0834398, -0.1054398, 0.0998004, -0.1826396, 0.1888796
5: -0.0781020, 0.0803555, -0.0976876, 0.0989877, -0.1770897, 0.1780430
6: 0.8320159, 1.0220795, 0.7992611, 1.0288818, -0.1968659, 0.2228184
7: -0.1048949, 0.0780332, -0.1263976, 0.0999615, -0.2048564, 0.2044308
8: -0.0695297, 0.1002338, -0.0803340, 0.1283996, -0.1979293, 0.1805678
9: -0.0652454, 0.0695308, -0.0873375, 0.0813373, -0.1465828, 0.1568684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2750860
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0327794, 0.0373962, -0.0941238, 0.0959895, -0.1287689, 0.1315200
1: -0.0464872, 0.0815675, -0.1096962, 0.1516559, -0.1981431, 0.1912636
2: -0.0648799, 0.1222744, -0.1196554, 0.2044678, -0.2693478, 0.2419297
3: -0.0518684, 0.0100409, -0.0820102, 0.0930878, -0.1449562, 0.0920511
4: -0.0725565, 0.0740299, -0.1428880, 0.1313654, -0.2039220, 0.2169179
5: -0.0670375, 0.0755046, -0.1349043, 0.1447066, -0.2117440, 0.2104089
6: 0.8490208, 1.0187097, 0.7347466, 1.0405189, -0.1914981, 0.2839631
7: -0.0954432, 0.0677825, -0.1676627, 0.1429055, -0.2383487, 0.2354452
8: -0.0636381, 0.0880433, -0.1052531, 0.1899105, -0.2535486, 0.1932964
9: -0.0530908, 0.0658080, -0.1361291, 0.1004932, -0.1535840, 0.2019370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2646901
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2684243
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0312693, 0.0344545, -0.0852213, 0.0871920, -0.1184613, 0.1196758
1: -0.0450895, 0.0787085, -0.1002163, 0.1435253, -0.1886148, 0.1789249
2: -0.0624342, 0.1178183, -0.1125706, 0.1939327, -0.2563669, 0.2303888
3: -0.0505875, 0.0091140, -0.0773524, 0.0817287, -0.1323162, 0.0864664
4: -0.0691600, 0.0717628, -0.1341455, 0.1239848, -0.1931448, 0.2059083
5: -0.0640711, 0.0739048, -0.1262157, 0.1338853, -0.1979564, 0.2001205
6: 0.8541219, 1.0176877, 0.7499051, 1.0378020, -0.1836801, 0.2677826
7: -0.0932415, 0.0644377, -0.1579525, 0.1328799, -0.2261214, 0.2223901
8: -0.0621085, 0.0839579, -0.0993666, 0.1755503, -0.2376588, 0.1833245
9: -0.0497104, 0.0645806, -0.1247028, 0.0960211, -0.1457316, 0.1892833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2646901
time: 2.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2682534
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0668166, 0.0728965, -0.0581472, 0.0604379, -0.1272545, 0.1310437
1: -0.0825902, 0.1386940, -0.0713864, 0.1187993, -0.2013895, 0.2100804
2: -0.1014135, 0.1843139, -0.0910246, 0.1618939, -0.2633075, 0.2753386
3: -0.0673263, 0.0698248, -0.0631876, 0.0471838, -0.1145101, 0.1330124
4: -0.1280533, 0.1123542, -0.1075578, 0.1015392, -0.2295925, 0.2199120
5: -0.1160352, 0.1118576, -0.0997926, 0.1009759, -0.2170111, 0.2116501
6: 0.7676219, 1.0350617, 0.7960042, 1.0295402, -0.2619182, 0.2390575
7: -0.1491823, 0.1121410, -0.1284224, 0.1023904, -0.2515727, 0.2405633
8: -0.0881812, 0.1504646, -0.0814651, 0.1318786, -0.2200598, 0.2319297
9: -0.1044718, 0.0938479, -0.0899537, 0.0824208, -0.1868926, 0.1838016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2737063
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2745901
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0412052, 0.0492351, -0.0581472, 0.0604379, -0.1016431, 0.1073823
1: -0.0551017, 0.1006184, -0.0713864, 0.1187993, -0.1739010, 0.1720048
2: -0.0770275, 0.1412186, -0.0910246, 0.1618939, -0.2389214, 0.2322433
3: -0.0570234, 0.0238240, -0.0631876, 0.0471838, -0.1042072, 0.0870116
4: -0.0881101, 0.0868105, -0.1075578, 0.1015392, -0.1896493, 0.1943684
5: -0.0825240, 0.0840279, -0.0997926, 0.1009759, -0.1835000, 0.1838204
6: 0.8245090, 1.0235928, 0.7960042, 1.0295402, -0.2050312, 0.2275887
7: -0.1100577, 0.0820093, -0.1284224, 0.1023904, -0.2124481, 0.2104317
8: -0.0717041, 0.1060895, -0.0814651, 0.1318786, -0.2035827, 0.1875547
9: -0.0698183, 0.0723656, -0.0899537, 0.0824208, -0.1522391, 0.1623193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2750860
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1198033, 0.1204593, -0.0351738, 0.0418507, -0.1616540, 0.1556331
1: -0.1337633, 0.1824415, -0.0486038, 0.0887355, -0.2224988, 0.2310452
2: -0.1345164, 0.2499089, -0.0694323, 0.1290221, -0.2635385, 0.3193411
3: -0.1045529, 0.1169145, -0.0538080, 0.0131695, -0.1177224, 0.1707225
4: -0.1881926, 0.1468469, -0.0776997, 0.0788388, -0.2670313, 0.2245466
5: -0.1569170, 0.1726216, -0.0724305, 0.0779270, -0.2348440, 0.2450521
6: 0.6965920, 1.0518944, 0.8403718, 1.0202572, -0.3236652, 0.2115226
7: -0.1960988, 0.1847504, -0.1002215, 0.0728476, -0.2689464, 0.2849719
8: -0.1375105, 0.2200322, -0.0666126, 0.0942297, -0.2317402, 0.2866448
9: -0.1741723, 0.1381169, -0.0591776, 0.0676666, -0.2418389, 0.1972945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2442122, upper bound: 0.2335502
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0638772, 0.0660198, -0.0362156, 0.0437600, -0.1076372, 0.1022353
1: -0.0774013, 0.1241177, -0.0495533, 0.0918080, -0.1692094, 0.1736710
2: -0.0955198, 0.1687807, -0.0713847, 0.1319142, -0.2274341, 0.2401654
3: -0.0662145, 0.0543911, -0.0546394, 0.0145907, -0.0808052, 0.1090305
4: -0.1132647, 0.1062221, -0.0799114, 0.0809000, -0.1941646, 0.1861335
5: -0.1053053, 0.1079026, -0.0747699, 0.0789653, -0.1842707, 0.1826725
6: 0.7863125, 1.0312638, 0.8366649, 1.0209391, -0.2346267, 0.1945989
7: -0.1345834, 0.1089936, -0.1022696, 0.0750370, -0.2096204, 0.2112632
8: -0.0852606, 0.1409900, -0.0678913, 0.0968813, -0.1821419, 0.2088813
9: -0.0973673, 0.0852583, -0.0617998, 0.0684633, -0.1658306, 0.1470581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1198033, 0.1204593, -0.0467408, 0.0535433, -0.1733466, 0.1672001
1: -0.1337633, 0.1824415, -0.0608136, 0.1076295, -0.2413928, 0.2432550
2: -0.1345164, 0.2499089, -0.0818635, 0.1491661, -0.2836825, 0.3317724
3: -0.1045529, 0.1169145, -0.0592600, 0.0322827, -0.1368357, 0.1761746
4: -0.1881926, 0.1468469, -0.0955473, 0.0920476, -0.2802402, 0.2423942
5: -0.1569170, 0.1726216, -0.0886257, 0.0901672, -0.2470841, 0.2612472
6: 0.6965920, 1.0518944, 0.8138335, 1.0258076, -0.3292156, 0.2380609
7: -0.1960988, 0.1847504, -0.1171815, 0.0890191, -0.2851178, 0.3019319
8: -0.1375105, 0.2200322, -0.0752679, 0.1142998, -0.2518103, 0.2953001
9: -0.1741723, 0.1381169, -0.0764834, 0.0762771, -0.2504494, 0.2146003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0638772, 0.0660198, -0.0495427, 0.0554957, -0.1193729, 0.1155625
1: -0.0774013, 0.1241177, -0.0636021, 0.1108523, -0.1882536, 0.1877198
2: -0.0955198, 0.1687807, -0.0841672, 0.1528259, -0.2483457, 0.2529479
3: -0.0662145, 0.0543911, -0.0603756, 0.0361641, -0.1023785, 0.1147667
4: -0.1132647, 0.1062221, -0.0990125, 0.0945869, -0.2078515, 0.2052346
5: -0.1053053, 0.1079026, -0.0913909, 0.0932569, -0.1985623, 0.1992934
6: 0.7863125, 1.0312638, 0.8089057, 1.0268847, -0.2405722, 0.2223582
7: -0.1345834, 0.1089936, -0.1204099, 0.0926384, -0.2272218, 0.2294034
8: -0.0852606, 0.1409900, -0.0770425, 0.1180960, -0.2033566, 0.2180325
9: -0.0973673, 0.0852583, -0.0796192, 0.0780497, -0.1754169, 0.1648774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1185531, 0.1192386, -0.0358105, 0.0430354, -0.1615885, 0.1550491
1: -0.1325008, 0.1811376, -0.0491666, 0.0906419, -0.2231427, 0.2303042
2: -0.1336446, 0.2480917, -0.0706430, 0.1308165, -0.2644611, 0.3187346
3: -0.1036917, 0.1155168, -0.0543238, 0.0140015, -0.1176932, 0.1698406
4: -0.1865094, 0.1459387, -0.0790674, 0.0801177, -0.2666271, 0.2250062
5: -0.1557608, 0.1711748, -0.0738647, 0.0785713, -0.2343321, 0.2450395
6: 0.6985979, 1.0514297, 0.8380717, 1.0206689, -0.3220710, 0.2133580
7: -0.1947186, 0.1830569, -0.1014923, 0.0741946, -0.2689132, 0.2845492
8: -0.1363333, 0.2182652, -0.0674036, 0.0958749, -0.2322083, 0.2856688
9: -0.1724553, 0.1369179, -0.0607964, 0.0681609, -0.2406163, 0.1977142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2446987, upper bound: 0.2344373
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0662946, 0.0682150, -0.0369804, 0.0449325, -0.1112271, 0.1051954
1: -0.0797279, 0.1266387, -0.0504781, 0.0936945, -0.1734224, 0.1771168
2: -0.0972055, 0.1721313, -0.0725931, 0.1336904, -0.2308959, 0.2447244
3: -0.0676981, 0.0570936, -0.0551499, 0.0161104, -0.0838085, 0.1122435
4: -0.1161564, 0.1079781, -0.0813287, 0.0821656, -0.1983220, 0.1893068
5: -0.1074369, 0.1106999, -0.0764304, 0.0796030, -0.1870399, 0.1871303
6: 0.7824345, 1.0320067, 0.8343886, 1.0215074, -0.2390729, 0.1976182
7: -0.1370308, 0.1122681, -0.1035272, 0.0765302, -0.2135610, 0.2157953
8: -0.0871339, 0.1444066, -0.0687078, 0.0985096, -0.1856435, 0.2131144
9: -0.1006871, 0.0868026, -0.0635169, 0.0689525, -0.1696396, 0.1503195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1185531, 0.1192386, -0.0485511, 0.0548048, -0.1733579, 0.1677897
1: -0.1325008, 0.1811376, -0.0626152, 0.1097119, -0.2422127, 0.2437528
2: -0.1336446, 0.2480917, -0.0833520, 0.1515308, -0.2851754, 0.3314436
3: -0.1036917, 0.1155168, -0.0599809, 0.0347906, -0.1384823, 0.1754977
4: -0.1865094, 0.1459387, -0.0977862, 0.0936883, -0.2801977, 0.2437250
5: -0.1557608, 0.1711748, -0.0904124, 0.0921634, -0.2479242, 0.2615871
6: 0.6985979, 1.0514297, 0.8106495, 1.0265034, -0.3279055, 0.2407802
7: -0.1947186, 0.1830569, -0.1192673, 0.0913575, -0.2860761, 0.3023242
8: -0.1363333, 0.2182652, -0.0764145, 0.1167526, -0.2530859, 0.2946797
9: -0.1724553, 0.1369179, -0.0785095, 0.0774223, -0.2498776, 0.2154274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
time: 1.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0662946, 0.0682150, -0.0517112, 0.0567493, -0.1230438, 0.1199261
1: -0.0797279, 0.1266387, -0.0655795, 0.1129214, -0.1926493, 0.1922182
2: -0.0972055, 0.1721313, -0.0859028, 0.1551757, -0.2523812, 0.2580342
3: -0.0676981, 0.0570936, -0.0610920, 0.0389718, -0.1066699, 0.1181856
4: -0.1161564, 0.1079781, -0.1012374, 0.0963661, -0.2125225, 0.2092155
5: -0.1074369, 0.1106999, -0.0935113, 0.0952406, -0.2026775, 0.2042112
6: 0.7824345, 1.0320067, 0.8055925, 1.0275761, -0.2451416, 0.2264142
7: -0.1370308, 0.1122681, -0.1224826, 0.0951424, -0.2321732, 0.2347507
8: -0.0871339, 0.1444066, -0.0781819, 0.1214971, -0.2086310, 0.2225885
9: -0.1006871, 0.0868026, -0.0821944, 0.0791877, -0.1798748, 0.1689971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1054612, 0.1071928, -0.0363463, 0.0439603, -0.1494215, 0.1435390
1: -0.1217688, 0.1620101, -0.0497113, 0.0921304, -0.2138992, 0.2117214
2: -0.1286779, 0.2178841, -0.0715912, 0.1322177, -0.2608956, 0.2894754
3: -0.0879417, 0.1075537, -0.0547266, 0.0148504, -0.1027922, 0.1622803
4: -0.1540216, 0.1407646, -0.0801536, 0.0811162, -0.2351378, 0.2209183
5: -0.1459690, 0.1584875, -0.0750536, 0.0790743, -0.2250433, 0.2335411
6: 0.7154424, 1.0439788, 0.8362759, 1.0210364, -0.3055940, 0.2077029
7: -0.1800286, 0.1556731, -0.1024845, 0.0752922, -0.2553208, 0.2581576
8: -0.1127494, 0.2081982, -0.0680309, 0.0971595, -0.2099089, 0.2762291
9: -0.1506803, 0.1061884, -0.0620932, 0.0685469, -0.2192272, 0.1682816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2684243, upper bound: 0.2689652
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684192
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0572997, 0.0598412, -0.0376815, 0.0459797, -0.1032795, 0.0975228
1: -0.0705784, 0.1180252, -0.0513198, 0.0953800, -0.1659584, 0.1693450
2: -0.0903502, 0.1609720, -0.0736724, 0.1352896, -0.2256398, 0.2346444
3: -0.0628589, 0.0461025, -0.0556059, 0.0174952, -0.0803541, 0.1017084
4: -0.1067254, 0.1008512, -0.0826147, 0.0832963, -0.1900217, 0.1834660
5: -0.0989653, 0.1001341, -0.0779137, 0.0801991, -0.1791644, 0.1780478
6: 0.7973238, 1.0292815, 0.8323356, 1.0220150, -0.2246912, 0.1969459
7: -0.1275954, 0.1014359, -0.1046751, 0.0778638, -0.2054592, 0.2061110
8: -0.0809924, 0.1305115, -0.0694371, 0.0999844, -0.1809769, 0.1999486
9: -0.0889111, 0.0819951, -0.0650507, 0.0694101, -0.1583212, 0.1470458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2712192
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2750860
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1054612, 0.1071928, -0.0494546, 0.0554344, -0.1608955, 0.1566474
1: -0.1217688, 0.1620101, -0.0635144, 0.1107510, -0.2325197, 0.2255245
2: -0.1286779, 0.2178841, -0.0840948, 0.1527108, -0.2813888, 0.3019789
3: -0.0879417, 0.1075537, -0.0603406, 0.0360421, -0.1239838, 0.1678943
4: -0.1540216, 0.1407646, -0.0989036, 0.0945070, -0.2485286, 0.2396682
5: -0.1459690, 0.1584875, -0.0913039, 0.0931597, -0.2391287, 0.2497914
6: 0.7154424, 1.0439788, 0.8090605, 1.0268508, -0.3114085, 0.2349184
7: -0.1800286, 0.1556731, -0.1203083, 0.0925246, -0.2725531, 0.2759814
8: -0.1127494, 0.2081982, -0.0769867, 0.1179767, -0.2307261, 0.2851849
9: -0.1506803, 0.1061884, -0.0795206, 0.0779939, -0.2286742, 0.1857090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2564253, upper bound: 0.2482876
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0572997, 0.0598412, -0.0529782, 0.0574504, -0.1147501, 0.1128194
1: -0.0705784, 0.1180252, -0.0667129, 0.1140786, -0.1846570, 0.1847381
2: -0.0903502, 0.1609720, -0.0869112, 0.1564898, -0.2468400, 0.2478831
3: -0.0628589, 0.0461025, -0.0614926, 0.0405885, -0.1034474, 0.1075951
4: -0.1067254, 0.1008512, -0.1024817, 0.0973830, -0.2041084, 0.2033330
5: -0.0989653, 0.1001341, -0.0947479, 0.0963502, -0.1953155, 0.1948820
6: 0.7973238, 1.0292815, 0.8037177, 1.0279628, -0.2306390, 0.2255638
7: -0.1275954, 0.1014359, -0.1236418, 0.0965694, -0.2241648, 0.2250777
8: -0.0809924, 0.1305115, -0.0788191, 0.1235408, -0.2045333, 0.2093306
9: -0.0889111, 0.0819951, -0.0837173, 0.0798242, -0.1687354, 0.1657123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2751140
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1046644, 0.1064055, -0.0371167, 0.0451415, -0.1498059, 0.1435222
1: -0.1209205, 0.1612825, -0.0506430, 0.0940311, -0.2149515, 0.2119255
2: -0.1280439, 0.2169415, -0.0728085, 0.1340070, -0.2620509, 0.2897500
3: -0.0875249, 0.1065372, -0.0552409, 0.0163815, -0.1039064, 0.1617781
4: -0.1532393, 0.1401041, -0.0815813, 0.0823913, -0.2356306, 0.2216855
5: -0.1451915, 0.1575190, -0.0767265, 0.0797167, -0.2249082, 0.2342455
6: 0.7167990, 1.0437357, 0.8339825, 1.0216087, -0.3048097, 0.2097533
7: -0.1791596, 0.1547759, -0.1037515, 0.0767964, -0.2559559, 0.2585274
8: -0.1122227, 0.2069131, -0.0688534, 0.0987999, -0.2110225, 0.2757665
9: -0.1496578, 0.1057882, -0.0638230, 0.0690397, -0.2186976, 0.1696112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 163

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2740193
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2740193
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0593296, 0.0616062, -0.0389459, 0.0471479, -0.1064775, 0.1005521
1: -0.0726455, 0.1198790, -0.0526769, 0.0972597, -0.1699052, 0.1725560
2: -0.0919656, 0.1632929, -0.0748764, 0.1374172, -0.2293827, 0.2381693
3: -0.0638062, 0.0486925, -0.0561146, 0.0197663, -0.0835725, 0.1048070
4: -0.1087190, 0.1025194, -0.0845867, 0.0845573, -0.1932763, 0.1871061
5: -0.1009464, 0.1024131, -0.0795680, 0.0815730, -0.1825194, 0.1819812
6: 0.7939912, 1.0299009, 0.8295270, 1.0225811, -0.2285899, 0.2003739
7: -0.1297120, 0.1037218, -0.1066066, 0.0793514, -0.2090634, 0.2103284
8: -0.0822468, 0.1337859, -0.0702506, 0.1021753, -0.1844221, 0.2040365
9: -0.0914712, 0.0830148, -0.0667615, 0.0704707, -0.1619419, 0.1497763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2712192
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2750860
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1046644, 0.1064055, -0.0516310, 0.0567049, -0.1613693, 0.1580366
1: -0.1209205, 0.1612825, -0.0655078, 0.1128483, -0.2337688, 0.2267903
2: -0.1280439, 0.2169415, -0.0858390, 0.1550926, -0.2831365, 0.3027804
3: -0.0875249, 0.1065372, -0.0610667, 0.0388697, -0.1263946, 0.1676039
4: -0.1532393, 0.1401041, -0.1011587, 0.0963017, -0.2495410, 0.2412629
5: -0.1451915, 0.1575190, -0.0934331, 0.0951705, -0.2403621, 0.2509521
6: 0.7167990, 1.0437357, 0.8057112, 1.0275514, -0.3107524, 0.2380246
7: -0.1791596, 0.1547759, -0.1224093, 0.0950521, -0.2742117, 0.2771853
8: -0.1122227, 0.2069131, -0.0781417, 0.1213678, -0.2335904, 0.2850547
9: -0.1496578, 0.1057882, -0.0820980, 0.0791475, -0.2288053, 0.1878862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2564300, upper bound: 0.2483571
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0593296, 0.0616062, -0.0552457, 0.0587049, -0.1180345, 0.1168518
1: -0.0726455, 0.1198790, -0.0687411, 0.1161494, -0.1887949, 0.1886201
2: -0.0919656, 0.1632929, -0.0887156, 0.1588417, -0.2508073, 0.2520086
3: -0.0638062, 0.0486925, -0.0622095, 0.0434817, -0.1072879, 0.1109020
4: -0.1087190, 0.1025194, -0.1047085, 0.0992028, -0.2079217, 0.2072279
5: -0.1009464, 0.1024131, -0.0969608, 0.0983356, -0.1992820, 0.1993739
6: 0.7939912, 1.0299009, 0.8003628, 1.0286545, -0.2346632, 0.2295381
7: -0.1297120, 0.1037218, -0.1257163, 0.0991228, -0.2288348, 0.2294381
8: -0.0822468, 0.1337859, -0.0799595, 0.1271983, -0.2094451, 0.2137454
9: -0.0914712, 0.0830148, -0.0864424, 0.0809633, -0.1724345, 0.1694572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=35, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2751140
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.29 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.38 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2720479
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2748985
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2720479
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737657, upper bound: 0.2748985
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753131
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2720479
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2748491
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2720479
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2737610, upper bound: 0.2748491
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753131
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2646901
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2684985
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2646901
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2683389
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2737201
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746293
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2750860
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2753898
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2646901
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2682653, upper bound: 0.2684243
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2646901
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2677799, upper bound: 0.2682534
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2737063
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2745901
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2750860
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2753898
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2442122, upper bound: 0.2335502
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2446987, upper bound: 0.2344373
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2684243, upper bound: 0.2689652
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684192
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2712192
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2750860
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2564253, upper bound: 0.2482876
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2751140
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2740193
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2746771, upper bound: 0.2740193
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2712192
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2745901, upper bound: 0.2750860
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2564300, upper bound: 0.2483571
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2751140
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0327411, 0.0373249, -0.0700760, 0.0669105, -0.0996516, 0.1074009
1: -0.0464533, 0.0814526, -0.0820068, 0.1296936, -0.1761469, 0.1634594
2: -0.0648070, 0.1221664, -0.1005179, 0.1742233, -0.2390303, 0.2226843
3: -0.0518373, 0.0099908, -0.0668984, 0.0624043, -0.1142415, 0.0768892
4: -0.0724741, 0.0739529, -0.1192723, 0.1111052, -0.1835794, 0.1932252
5: -0.0669511, 0.0754658, -0.1114347, 0.1113212, -0.1782723, 0.1869004
6: 0.8491593, 1.0186851, 0.7784198, 1.0331804, -0.1840211, 0.2402653
7: -0.0953667, 0.0677014, -0.1392843, 0.1158241, -0.2111908, 0.2069857
8: -0.0635905, 0.0879442, -0.0874178, 0.1511204, -0.2147109, 0.1753620
9: -0.0529933, 0.0657782, -0.1042670, 0.0884130, -0.1414063, 0.1700453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2475137, upper bound: 0.2553120
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2735462, upper bound: 0.2712177
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0327411, 0.0373249, -0.0935822, 0.0799165, -0.1126575, 0.1309071
1: -0.0464533, 0.0814526, -0.1030330, 0.1511615, -0.1976148, 0.1844856
2: -0.0648070, 0.1221664, -0.1192245, 0.1986031, -0.2634101, 0.2413909
3: -0.0518373, 0.0099908, -0.0743302, 0.0923970, -0.1442342, 0.0843210
4: -0.0724741, 0.0739529, -0.1423562, 0.1299709, -0.2024450, 0.2163091
5: -0.0669511, 0.0754658, -0.1343758, 0.1319036, -0.1988546, 0.2098416
6: 0.8491593, 1.0186851, 0.7436399, 1.0403537, -0.1911944, 0.2750452
7: -0.0953667, 0.0677014, -0.1607898, 0.1422956, -0.2376623, 0.2284912
8: -0.0635905, 0.0879442, -0.0992392, 0.1890369, -0.2526275, 0.1871834
9: -0.0529933, 0.0657782, -0.1325190, 0.1002212, -0.1532144, 0.1982973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2475137, upper bound: 0.2590214
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2735462, upper bound: 0.2747244
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0339039, 0.0394882, -0.0700760, 0.0669105, -0.1008144, 0.1095642
1: -0.0474813, 0.0849339, -0.0820068, 0.1296936, -0.1771749, 0.1669406
2: -0.0670179, 0.1254434, -0.1005179, 0.1742233, -0.2412412, 0.2259612
3: -0.0527793, 0.0115102, -0.0668984, 0.0624043, -0.1151836, 0.0784086
4: -0.0749720, 0.0762883, -0.1192723, 0.1111052, -0.1860772, 0.1955606
5: -0.0695703, 0.0766423, -0.1114347, 0.1113212, -0.1808914, 0.1880769
6: 0.8449589, 1.0194364, 0.7784198, 1.0331804, -0.1882215, 0.2410166
7: -0.0976873, 0.0701613, -0.1392843, 0.1158241, -0.2135114, 0.2094456
8: -0.0650351, 0.0909487, -0.0874178, 0.1511204, -0.2161554, 0.1783665
9: -0.0559494, 0.0666809, -0.1042670, 0.0884130, -0.1443624, 0.1709479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2648312
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2648312
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0339039, 0.0394882, -0.0935822, 0.0799165, -0.1138204, 0.1330704
1: -0.0474813, 0.0849339, -0.1030330, 0.1511615, -0.1986428, 0.1879669
2: -0.0670179, 0.1254434, -0.1192245, 0.1986031, -0.2656210, 0.2446679
3: -0.0527793, 0.0115102, -0.0743302, 0.0923970, -0.1451763, 0.0858404
4: -0.0749720, 0.0762883, -0.1423562, 0.1299709, -0.2049429, 0.2186445
5: -0.0695703, 0.0766423, -0.1343758, 0.1319036, -0.2014738, 0.2110181
6: 0.8449589, 1.0194364, 0.7436399, 1.0403537, -0.1953948, 0.2757964
7: -0.0976873, 0.0701613, -0.1607898, 0.1422956, -0.2399829, 0.2309511
8: -0.0650351, 0.0909487, -0.0992392, 0.1890369, -0.2540720, 0.1901879
9: -0.0559494, 0.0666809, -0.1325190, 0.1002212, -0.1561705, 0.1991999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2684923
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2683649
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0683381, 0.0743022, -0.0357408, 0.0429055, -0.1112436, 0.1100429
1: -0.0842233, 0.1409560, -0.0491050, 0.0904329, -0.1746562, 0.1900610
2: -0.1028623, 0.1868740, -0.0705103, 0.1306199, -0.2334822, 0.2573843
3: -0.0679384, 0.0725577, -0.0542673, 0.0139103, -0.0818487, 0.1268250
4: -0.1304262, 0.1138716, -0.0789175, 0.0799775, -0.2104037, 0.1927892
5: -0.1180261, 0.1135109, -0.0737076, 0.0785007, -0.1965268, 0.1872185
6: 0.7642424, 1.0357430, 0.8383240, 1.0206236, -0.2563812, 0.1974190
7: -0.1515066, 0.1139310, -0.1013530, 0.0740470, -0.2255536, 0.2152840
8: -0.0891601, 0.1531008, -0.0673169, 0.0956946, -0.1848547, 0.2204177
9: -0.1065305, 0.0951241, -0.0606189, 0.0681067, -0.1746372, 0.1557430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0683381, 0.0743022, -0.0482980, 0.0546283, -0.1229664, 0.1226002
1: -0.0842233, 0.1409560, -0.0623633, 0.1094206, -0.1936438, 0.2033193
2: -0.1028623, 0.1868740, -0.0831438, 0.1512001, -0.2540624, 0.2700178
3: -0.0679384, 0.0725577, -0.0598800, 0.0344400, -0.1023784, 0.1324378
4: -0.1304262, 0.1138716, -0.0974731, 0.0934588, -0.2238851, 0.2113447
5: -0.1180261, 0.1135109, -0.0901624, 0.0918843, -0.2099104, 0.2036733
6: 0.7642424, 1.0357430, 0.8110948, 1.0264063, -0.2621639, 0.2246482
7: -0.1515066, 0.1139310, -0.1189757, 0.0910305, -0.2425371, 0.2329068
8: -0.0891601, 0.1531008, -0.0762542, 0.1164095, -0.2055696, 0.2293550
9: -0.1065305, 0.0951241, -0.0782261, 0.0772622, -0.1837927, 0.1733502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0417515, 0.0497397, -0.0357408, 0.0429055, -0.0846570, 0.0854805
1: -0.0556880, 0.1014305, -0.0491050, 0.0904329, -0.1461209, 0.1505354
2: -0.0775476, 0.1421378, -0.0705103, 0.1306199, -0.2081676, 0.2126481
3: -0.0572431, 0.0248052, -0.0542673, 0.0139103, -0.0711534, 0.0790725
4: -0.0889620, 0.0873553, -0.0789175, 0.0799775, -0.1689395, 0.1662729
5: -0.0832388, 0.0846215, -0.0737076, 0.0785007, -0.1617395, 0.1583290
6: 0.8232958, 1.0238374, 0.8383240, 1.0206236, -0.1973277, 0.1855135
7: -0.1108923, 0.0826519, -0.1013530, 0.0740470, -0.1849393, 0.1840049
8: -0.0720555, 0.1070360, -0.0673169, 0.0956946, -0.1677501, 0.1743529
9: -0.0705574, 0.0728238, -0.0606189, 0.0681067, -0.1386641, 0.1334427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753131
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753131
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0417515, 0.0497397, -0.0482980, 0.0546283, -0.0963797, 0.0980377
1: -0.0556880, 0.1014305, -0.0623633, 0.1094206, -0.1651085, 0.1637938
2: -0.0775476, 0.1421378, -0.0831438, 0.1512001, -0.2287477, 0.2252816
3: -0.0572431, 0.0248052, -0.0598800, 0.0344400, -0.0916831, 0.0846852
4: -0.0889620, 0.0873553, -0.0974731, 0.0934588, -0.1824209, 0.1848284
5: -0.0832388, 0.0846215, -0.0901624, 0.0918843, -0.1751231, 0.1747839
6: 0.8232958, 1.0238374, 0.8110948, 1.0264063, -0.2031105, 0.2127427
7: -0.1108923, 0.0826519, -0.1189757, 0.0910305, -0.2019228, 0.2016276
8: -0.0720555, 0.1070360, -0.0762542, 0.1164095, -0.1884650, 0.1832902
9: -0.0705574, 0.0728238, -0.0782261, 0.0772622, -0.1478196, 0.1510499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753898
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753898
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0344318, 0.0404702, -0.0722859, 0.0681331, -0.1025650, 0.1127561
1: -0.0479478, 0.0865144, -0.0839835, 0.1317119, -0.1796597, 0.1704978
2: -0.0680216, 0.1269310, -0.1022765, 0.1765151, -0.2445367, 0.2292075
3: -0.0532069, 0.0122001, -0.0675970, 0.0652240, -0.1184309, 0.0797971
4: -0.0761058, 0.0773486, -0.1214425, 0.1128789, -0.1889847, 0.1987911
5: -0.0707593, 0.0771763, -0.1135913, 0.1132562, -0.1840155, 0.1907676
6: 0.8430520, 1.0197777, 0.7751502, 1.0338550, -0.1908029, 0.2446274
7: -0.0987408, 0.0712780, -0.1413061, 0.1183127, -0.2170535, 0.2125841
8: -0.0656908, 0.0923125, -0.0885291, 0.1546848, -0.2203757, 0.1808417
9: -0.0572914, 0.0670906, -0.1069229, 0.0895232, -0.1468146, 0.1740135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2463033, upper bound: 0.2548496
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2735248, upper bound: 0.2712177
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0344318, 0.0404702, -0.0959076, 0.0812030, -0.1156349, 0.1363779
1: -0.0479478, 0.0865144, -0.1051131, 0.1532850, -0.2012328, 0.1916275
2: -0.0680216, 0.1269310, -0.1210750, 0.2010149, -0.2690365, 0.2480060
3: -0.0532069, 0.0122001, -0.0750654, 0.0953639, -0.1485708, 0.0872655
4: -0.0761058, 0.0773486, -0.1446398, 0.1318370, -0.2079428, 0.2219884
5: -0.0707593, 0.0771763, -0.1366452, 0.1339397, -0.2046990, 0.2138215
6: 0.8430520, 1.0197777, 0.7401994, 1.0410634, -0.1980114, 0.2795782
7: -0.0987408, 0.0712780, -0.1629171, 0.1449144, -0.2436552, 0.2341951
8: -0.0656908, 0.0923125, -0.1004087, 0.1927879, -0.2584787, 0.1927212
9: -0.0572914, 0.0670906, -0.1353138, 0.1013893, -0.1586806, 0.2024045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2463033, upper bound: 0.2579004
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2735248, upper bound: 0.2746866
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0355394, 0.0425310, -0.0722859, 0.0681331, -0.1036726, 0.1148169
1: -0.0489270, 0.0898303, -0.0839835, 0.1317119, -0.1806389, 0.1738138
2: -0.0701276, 0.1300525, -0.1022765, 0.1765151, -0.2466427, 0.2323290
3: -0.0541042, 0.0136473, -0.0675970, 0.0652240, -0.1193282, 0.0812443
4: -0.0784851, 0.0795732, -0.1214425, 0.1128789, -0.1913640, 0.2010157
5: -0.0732541, 0.0782970, -0.1135913, 0.1132562, -0.1865104, 0.1918883
6: 0.8390511, 1.0204935, 0.7751502, 1.0338550, -0.1948038, 0.2453433
7: -0.1009513, 0.0736213, -0.1413061, 0.1183127, -0.2192640, 0.2149274
8: -0.0670668, 0.0951745, -0.0885291, 0.1546848, -0.2217517, 0.1837036
9: -0.0601072, 0.0679505, -0.1069229, 0.0895232, -0.1496304, 0.1748734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2648312
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2648312
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0355394, 0.0425310, -0.0959076, 0.0812030, -0.1167425, 0.1384386
1: -0.0489270, 0.0898303, -0.1051131, 0.1532850, -0.2022120, 0.1949434
2: -0.0701276, 0.1300525, -0.1210750, 0.2010149, -0.2711425, 0.2511275
3: -0.0541042, 0.0136473, -0.0750654, 0.0953639, -0.1494682, 0.0887127
4: -0.0784851, 0.0795732, -0.1446398, 0.1318370, -0.2103222, 0.2242130
5: -0.0732541, 0.0782970, -0.1366452, 0.1339397, -0.2071938, 0.2149422
6: 0.8390511, 1.0204935, 0.7401994, 1.0410634, -0.2020123, 0.2802941
7: -0.1009513, 0.0736213, -0.1629171, 0.1449144, -0.2458657, 0.2365384
8: -0.0670668, 0.0951745, -0.1004087, 0.1927879, -0.2598547, 0.1955831
9: -0.0601072, 0.0679505, -0.1353138, 0.1013893, -0.1614964, 0.2032643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2684189
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2682666
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0799113, 0.0766567, -0.0364308, 0.0440901, -0.1240013, 0.1130876
1: -0.0938252, 0.1457810, -0.0498137, 0.0923391, -0.1861643, 0.1955947
2: -0.1091351, 0.1924928, -0.0717248, 0.1324143, -0.2415494, 0.2642176
3: -0.0724676, 0.0782313, -0.0547831, 0.0150185, -0.0874862, 0.1330144
4: -0.1365708, 0.1221080, -0.0803104, 0.0812562, -0.2178270, 0.2024184
5: -0.1213607, 0.1267451, -0.0752373, 0.0791448, -0.2005056, 0.2019824
6: 0.7554948, 1.0385557, 0.8360240, 1.0210991, -0.2656043, 0.2025317
7: -0.1553999, 0.1318663, -0.1026236, 0.0754574, -0.2308573, 0.2344899
8: -0.0962765, 0.1592403, -0.0681212, 0.0973396, -0.1936161, 0.2273615
9: -0.1136055, 0.0972618, -0.0622832, 0.0686010, -0.1822065, 0.1595449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0799113, 0.0766567, -0.0501597, 0.0558908, -0.1358021, 0.1268164
1: -0.0938252, 0.1457810, -0.0641917, 0.1115045, -0.2053297, 0.2099727
2: -0.1091351, 0.1924928, -0.0846681, 0.1535666, -0.2627018, 0.2771609
3: -0.0724676, 0.0782313, -0.0606015, 0.0369923, -0.1094599, 0.1388328
4: -0.1365708, 0.1221080, -0.0997139, 0.0951208, -0.2316916, 0.2218219
5: -0.1213607, 0.1267451, -0.0919970, 0.0938822, -0.2152429, 0.2187422
6: 0.7554948, 1.0385557, 0.8078881, 1.0271025, -0.2716076, 0.2306677
7: -0.1553999, 0.1318663, -0.1210632, 0.0933952, -0.2487951, 0.2529295
8: -0.0962765, 0.1592403, -0.0774017, 0.1189944, -0.2152709, 0.2366420
9: -0.1136055, 0.0972618, -0.0803296, 0.0784084, -0.1920138, 0.1775914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0459928, 0.0530220, -0.0364308, 0.0440901, -0.0900828, 0.0894529
1: -0.0600691, 0.1067692, -0.0498137, 0.0923391, -0.1524082, 0.1565829
2: -0.0812485, 0.1481890, -0.0717248, 0.1324143, -0.2136628, 0.2199139
3: -0.0589622, 0.0312465, -0.0547831, 0.0150185, -0.0739807, 0.0860296
4: -0.0946221, 0.0913697, -0.0803104, 0.0812562, -0.1758784, 0.1716801
5: -0.0878875, 0.0893423, -0.0752373, 0.0791448, -0.1670323, 0.1645796
6: 0.8151492, 1.0255202, 0.8360240, 1.0210991, -0.2059499, 0.1894962
7: -0.1163196, 0.0880528, -0.1026236, 0.0754574, -0.1917770, 0.1906764
8: -0.0747941, 0.1132864, -0.0681212, 0.0973396, -0.1721338, 0.1814076
9: -0.0756463, 0.0758038, -0.0622832, 0.0686010, -0.1442473, 0.1380869

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753131
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753131
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0459928, 0.0530220, -0.0501597, 0.0558908, -0.1018836, 0.1031817
1: -0.0600691, 0.1067692, -0.0641917, 0.1115045, -0.1715736, 0.1709609
2: -0.0812485, 0.1481890, -0.0846681, 0.1535666, -0.2348152, 0.2328571
3: -0.0589622, 0.0312465, -0.0606015, 0.0369923, -0.0959545, 0.0918480
4: -0.0946221, 0.0913697, -0.0997139, 0.0951208, -0.1897430, 0.1910836
5: -0.0878875, 0.0893423, -0.0919970, 0.0938822, -0.1817697, 0.1813393
6: 0.8151492, 1.0255202, 0.8078881, 1.0271025, -0.2119533, 0.2176321
7: -0.1163196, 0.0880528, -0.1210632, 0.0933952, -0.2097148, 0.2091160
8: -0.0747941, 0.1132864, -0.0774017, 0.1189944, -0.1937886, 0.1906881
9: -0.0756463, 0.0758038, -0.0803296, 0.0784084, -0.1540547, 0.1561334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753898
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753898
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0312568, 0.0344295, -0.0661318, 0.0647282, -0.0959850, 0.1005613
1: -0.0450777, 0.0786921, -0.0784787, 0.1260915, -0.1711692, 0.1571707
2: -0.0624158, 0.1177806, -0.0973790, 0.1701324, -0.2325482, 0.2151595
3: -0.0505766, 0.0091108, -0.0656513, 0.0573718, -0.1079483, 0.0747622
4: -0.0691312, 0.0717474, -0.1153990, 0.1079397, -0.1770710, 0.1871464
5: -0.0640484, 0.0738913, -0.1075852, 0.1078676, -0.1719161, 0.1814765
6: 0.8541625, 1.0176791, 0.7842556, 1.0319767, -0.1778142, 0.2334235
7: -0.0932268, 0.0644093, -0.1356758, 0.1113822, -0.2046089, 0.2000852
8: -0.0620973, 0.0839233, -0.0854342, 0.1447582, -0.2068555, 0.1693575
9: -0.0496844, 0.0645702, -0.0995264, 0.0864318, -0.1361163, 0.1640966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2651781, upper bound: 0.2604437
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2604437
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0312568, 0.0344295, -0.0893008, 0.0775475, -0.1088042, 0.1237303
1: -0.0450777, 0.0786921, -0.0992033, 0.1472512, -0.1923289, 0.1778954
2: -0.0624158, 0.1177806, -0.1158172, 0.1941624, -0.2565782, 0.2335978
3: -0.0505766, 0.0091108, -0.0729766, 0.0869341, -0.1375107, 0.0820874
4: -0.0691312, 0.0717474, -0.1381517, 0.1265346, -0.1956658, 0.2098991
5: -0.0640484, 0.0738913, -0.1301972, 0.1281547, -0.1922032, 0.2040885
6: 0.8541625, 1.0176791, 0.7499749, 1.0390472, -0.1848847, 0.2677042
7: -0.0932268, 0.0644093, -0.1568727, 0.1374741, -0.2307008, 0.2212820
8: -0.0620973, 0.0839233, -0.0970860, 0.1821308, -0.2442281, 0.1810093
9: -0.0496844, 0.0645702, -0.1273731, 0.0980705, -0.1477549, 0.1919433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=43, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2651781, upper bound: 0.2604437
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2654327
time: 1.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0298840, 0.0314536, -0.0573624, 0.0598761, -0.0897601, 0.0888160
1: -0.0436637, 0.0769105, -0.0706345, 0.1180825, -0.1617462, 0.1475450
2: -0.0602743, 0.1132726, -0.0904001, 0.1610371, -0.2213114, 0.2036726
3: -0.0492808, 0.0087489, -0.0628788, 0.0461824, -0.0954632, 0.0716277
4: -0.0658121, 0.0699016, -0.1067872, 0.1009016, -0.1667137, 0.1766888
5: -0.0613684, 0.0722729, -0.0990266, 0.1001891, -0.1615575, 0.1712995
6: 0.8590221, 1.0168327, 0.7972310, 1.0293006, -0.1702785, 0.2196018
7: -0.0914694, 0.0612404, -0.1276528, 0.1015065, -0.1929760, 0.1888932
8: -0.0607639, 0.0799528, -0.0810240, 0.1306128, -0.1913767, 0.1609768
9: -0.0468134, 0.0633284, -0.0889865, 0.0820266, -0.1288400, 0.1523150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2582777, upper bound: 0.2515041
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2577288, upper bound: 0.2515041
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0298840, 0.0314536, -0.0796892, 0.0722295, -0.1021135, 0.1111428
1: -0.0436637, 0.0769105, -0.0906057, 0.1384731, -0.1821368, 0.1675163
2: -0.0602743, 0.1132726, -0.1081682, 0.1841937, -0.2444681, 0.2214407
3: -0.0492808, 0.0087489, -0.0699378, 0.0746702, -0.1239510, 0.0786867
4: -0.0658121, 0.0699016, -0.1287129, 0.1188205, -0.1846326, 0.1986145
5: -0.0613684, 0.0722729, -0.1208167, 0.1197387, -0.1811071, 0.1930897
6: 0.8590221, 1.0168327, 0.7641961, 1.0361142, -0.1770921, 0.2526366
7: -0.0914694, 0.0612404, -0.1480792, 0.1266500, -0.2181194, 0.2093196
8: -0.0607639, 0.0799528, -0.0922523, 0.1666269, -0.2273908, 0.1722051
9: -0.0468134, 0.0633284, -0.1158210, 0.0932422, -0.1400556, 0.1791494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2582777, upper bound: 0.2637625
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2577288, upper bound: 0.2634628
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0644407, 0.0707015, -0.0370790, 0.0450837, -0.1095244, 0.1077805
1: -0.0800402, 0.1351618, -0.0505974, 0.0939381, -0.1739783, 0.1857592
2: -0.0991513, 0.1803161, -0.0727490, 0.1339195, -0.2330708, 0.2530650
3: -0.0663706, 0.0655575, -0.0552158, 0.0163066, -0.0826771, 0.1207732
4: -0.1243479, 0.1099846, -0.0815115, 0.0823289, -0.2066768, 0.1914961
5: -0.1129265, 0.1092759, -0.0766446, 0.0796852, -0.1926117, 0.1859205
6: 0.7728992, 1.0339979, 0.8340948, 1.0215807, -0.2486815, 0.1999031
7: -0.1455528, 0.1093458, -0.1036895, 0.0767227, -0.2222755, 0.2130353
8: -0.0866527, 0.1463480, -0.0688132, 0.0987196, -0.1853723, 0.2151611
9: -0.1012571, 0.0918550, -0.0637384, 0.0690156, -0.1702727, 0.1555934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2688290, upper bound: 0.2689333
time: 1.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647270, upper bound: 0.2678041
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0644407, 0.0707015, -0.0513475, 0.0565481, -0.1209888, 0.1220490
1: -0.0800402, 0.1351618, -0.0652542, 0.1125893, -0.1926295, 0.2004160
2: -0.0991513, 0.1803161, -0.0856134, 0.1547986, -0.2539499, 0.2659295
3: -0.0663706, 0.0655575, -0.0609770, 0.0385078, -0.1048784, 0.1265345
4: -0.1243479, 0.1099846, -0.1008804, 0.0960742, -0.2204221, 0.2108650
5: -0.1129265, 0.1092759, -0.0931564, 0.0949223, -0.2078488, 0.2024322
6: 0.7728992, 1.0339979, 0.8061306, 1.0274649, -0.2545657, 0.2278672
7: -0.1455528, 0.1093458, -0.1221499, 0.0947329, -0.2402857, 0.2314957
8: -0.0866527, 0.1463480, -0.0779990, 0.1209106, -0.2075633, 0.2243470
9: -0.1012571, 0.0918550, -0.0817573, 0.0790050, -0.1802621, 0.1736123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2688290, upper bound: 0.2699453
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647270, upper bound: 0.2686479
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0378255, 0.0461127, -0.0370790, 0.0450837, -0.0829092, 0.0831917
1: -0.0514743, 0.0955939, -0.0505974, 0.0939381, -0.1454124, 0.1461913
2: -0.0738095, 0.1355318, -0.0727490, 0.1339195, -0.2077290, 0.2082807
3: -0.0556638, 0.0177538, -0.0552158, 0.0163066, -0.0719704, 0.0729696
4: -0.0828392, 0.0834398, -0.0815115, 0.0823289, -0.1651681, 0.1649513
5: -0.0781020, 0.0803555, -0.0766446, 0.0796852, -0.1577872, 0.1570001
6: 0.8320159, 1.0220795, 0.8340948, 1.0215807, -0.1895648, 0.1879846
7: -0.1048949, 0.0780332, -0.1036895, 0.0767227, -0.1816176, 0.1817227
8: -0.0695297, 0.1002338, -0.0688132, 0.0987196, -0.1682493, 0.1690470
9: -0.0652454, 0.0695308, -0.0637384, 0.0690156, -0.1342610, 0.1332692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2505233, upper bound: 0.2608429
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747039, upper bound: 0.2747000
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0378255, 0.0461127, -0.0513475, 0.0565481, -0.0943736, 0.0974602
1: -0.0514743, 0.0955939, -0.0652542, 0.1125893, -0.1640636, 0.1608481
2: -0.0738095, 0.1355318, -0.0856134, 0.1547986, -0.2286081, 0.2211452
3: -0.0556638, 0.0177538, -0.0609770, 0.0385078, -0.0941716, 0.0787308
4: -0.0828392, 0.0834398, -0.1008804, 0.0960742, -0.1789134, 0.1843202
5: -0.0781020, 0.0803555, -0.0931564, 0.0949223, -0.1730243, 0.1735118
6: 0.8320159, 1.0220795, 0.8061306, 1.0274649, -0.1954489, 0.2159488
7: -0.1048949, 0.0780332, -0.1221499, 0.0947329, -0.1996278, 0.2001831
8: -0.0695297, 0.1002338, -0.0779990, 0.1209106, -0.1904403, 0.1782328
9: -0.0652454, 0.0695308, -0.0817573, 0.0790050, -0.1442505, 0.1512882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2505233, upper bound: 0.2624015
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747039, upper bound: 0.2753898
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0327794, 0.0373962, -0.0683526, 0.0705228, -0.1033023, 0.1057489
1: -0.0464872, 0.0815675, -0.0822537, 0.1281198, -0.1746070, 0.1638212
2: -0.0648799, 0.1222744, -0.0991464, 0.1739708, -0.2388507, 0.2214207
3: -0.0518684, 0.0100409, -0.0685270, 0.0602054, -0.1120738, 0.0785679
4: -0.0725565, 0.0740299, -0.1175799, 0.1100000, -0.1825566, 0.1916098
5: -0.0670375, 0.0755046, -0.1097527, 0.1133810, -0.1804184, 0.1852573
6: 0.8490208, 1.0187097, 0.7786273, 1.0326544, -0.1836336, 0.2400823
7: -0.0954432, 0.0677825, -0.1395537, 0.1138833, -0.2093265, 0.2073362
8: -0.0636381, 0.0880433, -0.0882130, 0.1483406, -0.2119787, 0.1762564
9: -0.0530908, 0.0658080, -0.1030523, 0.0875474, -0.1406381, 0.1688603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2651773, upper bound: 0.2604437
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2604437
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0327794, 0.0373962, -0.0917244, 0.0936184, -0.1263978, 0.1291206
1: -0.0464872, 0.0815675, -0.1071412, 0.1494646, -0.1959518, 0.1887087
2: -0.0648799, 0.1222744, -0.1177460, 0.2016285, -0.2665085, 0.2400203
3: -0.0518684, 0.0100409, -0.0807548, 0.0900265, -0.1418948, 0.0907957
4: -0.0725565, 0.0740299, -0.1405319, 0.1293763, -0.2019328, 0.2145617
5: -0.0670375, 0.0755046, -0.1325626, 0.1417900, -0.2088275, 0.2080672
6: 0.8490208, 1.0187097, 0.7388321, 1.0397867, -0.1907659, 0.2798775
7: -0.0954432, 0.0677825, -0.1650456, 0.1402033, -0.2356465, 0.2328282
8: -0.0636381, 0.0880433, -0.1036666, 0.1860402, -0.2496784, 0.1917099
9: -0.0530908, 0.0658080, -0.1330495, 0.0992879, -0.1523787, 0.1988575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2651773, upper bound: 0.2654294
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2654150
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0312693, 0.0344545, -0.0595835, 0.0618572, -0.0931264, 0.0940380
1: -0.0450895, 0.0787085, -0.0729158, 0.1201111, -0.1652006, 0.1516244
2: -0.0624342, 0.1178183, -0.0921677, 0.1635935, -0.2260278, 0.2099859
3: -0.0505875, 0.0091140, -0.0639390, 0.0490165, -0.0996040, 0.0730530
4: -0.0691600, 0.0717628, -0.1089683, 0.1027300, -0.1718900, 0.1807311
5: -0.0640711, 0.0739048, -0.1011944, 0.1027216, -0.1667928, 0.1750992
6: 0.8541219, 1.0176877, 0.7935588, 1.0299783, -0.1758564, 0.2241289
7: -0.0932415, 0.0644377, -0.1299889, 0.1040078, -0.1972494, 0.1944266
8: -0.0621085, 0.0839579, -0.0824147, 0.1341954, -0.1963039, 0.1663726
9: -0.0497104, 0.0645806, -0.0917972, 0.0831423, -0.1328527, 0.1563778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=19, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646742, upper bound: 0.2604437
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2643144, upper bound: 0.2604437
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0312693, 0.0344545, -0.0821226, 0.0841299, -0.1153992, 0.1165771
1: -0.0450895, 0.0787085, -0.0969167, 0.1406954, -0.1857850, 0.1756252
2: -0.0624342, 0.1178183, -0.1101047, 0.1902658, -0.2527000, 0.2279229
3: -0.0505875, 0.0091140, -0.0757312, 0.0777750, -0.1283625, 0.0848452
4: -0.0691600, 0.0717628, -0.1311024, 0.1214159, -0.1905759, 0.2028652
5: -0.0640711, 0.0739048, -0.1231915, 0.1301187, -0.1941899, 0.1970963
6: 0.8541219, 1.0176877, 0.7551813, 1.0368567, -0.1827348, 0.2625064
7: -0.0932415, 0.0644377, -0.1545728, 0.1293903, -0.2226318, 0.2190105
8: -0.0621085, 0.0839579, -0.0973178, 0.1705520, -0.2326604, 0.1812757
9: -0.0497104, 0.0645806, -0.1207257, 0.0944646, -0.1441750, 0.1853063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646742, upper bound: 0.2652622
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2643144, upper bound: 0.2652492
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0668166, 0.0728965, -0.0379896, 0.0462643, -0.1130808, 0.1108861
1: -0.0825902, 0.1386940, -0.0516504, 0.0958379, -0.1784281, 0.1903444
2: -0.1014135, 0.1843139, -0.0739658, 0.1358079, -0.2372214, 0.2582797
3: -0.0673263, 0.0698248, -0.0557298, 0.0180484, -0.0853748, 0.1255547
4: -0.1280533, 0.1123542, -0.0830951, 0.0836035, -0.2116568, 0.1954493
5: -0.1160352, 0.1118576, -0.0783166, 0.0805338, -0.1965690, 0.1901742
6: 0.7676219, 1.0350617, 0.8316514, 1.0221530, -0.2545311, 0.2034103
7: -0.1491823, 0.1121410, -0.1051456, 0.0782262, -0.2274085, 0.2172866
8: -0.0881812, 0.1504646, -0.0696353, 0.1005181, -0.1886993, 0.2200999
9: -0.1044718, 0.0938479, -0.0654675, 0.0696684, -0.1741403, 0.1593153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2687752, upper bound: 0.2689225
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646901, upper bound: 0.2677799
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0668166, 0.0728965, -0.0536410, 0.0578170, -0.1246336, 0.1265376
1: -0.0825902, 0.1386940, -0.0673057, 0.1146839, -0.1972740, 0.2059997
2: -0.1014135, 0.1843139, -0.0874385, 0.1571774, -0.2585909, 0.2717524
3: -0.0673263, 0.0698248, -0.0617021, 0.0414342, -0.1087605, 0.1315269
4: -0.1280533, 0.1123542, -0.1031326, 0.0979149, -0.2259682, 0.2154868
5: -0.1160352, 0.1118576, -0.0953947, 0.0969305, -0.2129657, 0.2072522
6: 0.7676219, 1.0350617, 0.8027371, 1.0281649, -0.2605429, 0.2323246
7: -0.1491823, 0.1121410, -0.1242482, 0.0973157, -0.2464980, 0.2363892
8: -0.0881812, 0.1504646, -0.0791524, 0.1246100, -0.2127912, 0.2296170
9: -0.1044718, 0.0938479, -0.0845139, 0.0801572, -0.1846290, 0.1783617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2687752, upper bound: 0.2698907
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2646901, upper bound: 0.2685597
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0412052, 0.0492351, -0.0379896, 0.0462643, -0.0874694, 0.0872246
1: -0.0551017, 0.1006184, -0.0516504, 0.0958379, -0.1509396, 0.1522688
2: -0.0770275, 0.1412186, -0.0739658, 0.1358079, -0.2128354, 0.2151844
3: -0.0570234, 0.0238240, -0.0557298, 0.0180484, -0.0750718, 0.0795539
4: -0.0881101, 0.0868105, -0.0830951, 0.0836035, -0.1717136, 0.1699056
5: -0.0825240, 0.0840279, -0.0783166, 0.0805338, -0.1630578, 0.1623445
6: 0.8245090, 1.0235928, 0.8316514, 1.0221530, -0.1976440, 0.1919414
7: -0.1100577, 0.0820093, -0.1051456, 0.0782262, -0.1882839, 0.1871549
8: -0.0717041, 0.1060895, -0.0696353, 0.1005181, -0.1722222, 0.1757248
9: -0.0698183, 0.0723656, -0.0654675, 0.0696684, -0.1394867, 0.1378330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2479369, upper bound: 0.2603138
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747000, upper bound: 0.2747000
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0412052, 0.0492351, -0.0536410, 0.0578170, -0.0990222, 0.1028761
1: -0.0551017, 0.1006184, -0.0673057, 0.1146839, -0.1697855, 0.1679241
2: -0.0770275, 0.1412186, -0.0874385, 0.1571774, -0.2342049, 0.2286572
3: -0.0570234, 0.0238240, -0.0617021, 0.0414342, -0.0984576, 0.0855262
4: -0.0881101, 0.0868105, -0.1031326, 0.0979149, -0.1860249, 0.1899432
5: -0.0825240, 0.0840279, -0.0953947, 0.0969305, -0.1794546, 0.1794226
6: 0.8245090, 1.0235928, 0.8027371, 1.0281649, -0.2036558, 0.2208557
7: -0.1100577, 0.0820093, -0.1242482, 0.0973157, -0.2073735, 0.2062575
8: -0.0717041, 0.1060895, -0.0791524, 0.1246100, -0.1963140, 0.1852420
9: -0.0698183, 0.0723656, -0.0845139, 0.0801572, -0.1499755, 0.1568794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2479369, upper bound: 0.2613184
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747000, upper bound: 0.2753898
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1198033, 0.1204593, -0.0322822, 0.0364712, -0.1562745, 0.1527415
1: -0.1337633, 0.1824415, -0.0460478, 0.0800788, -0.2138421, 0.2284892
2: -0.1345164, 0.2499089, -0.0639346, 0.1208731, -0.2553895, 0.3138435
3: -0.1045529, 0.1169145, -0.0514656, 0.0093912, -0.1139441, 0.1683801
4: -0.1881926, 0.1468469, -0.0714885, 0.0730313, -0.2612239, 0.2183354
5: -0.1569170, 0.1726216, -0.0659176, 0.0750016, -0.2319185, 0.2385391
6: 0.6965920, 1.0518944, 0.8508168, 1.0183884, -0.3217964, 0.2010776
7: -0.1960988, 0.1847504, -0.0944510, 0.0667308, -0.2628295, 0.2792013
8: -0.1375105, 0.2200322, -0.0630204, 0.0867587, -0.2242692, 0.2830526
9: -0.1741723, 0.1381169, -0.0518267, 0.0654220, -0.2395943, 0.1899436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0638772, 0.0660198, -0.0513164, 0.0669105, -0.1307877, 0.1173362
1: -0.0774013, 0.1241177, -0.0678130, 0.1290613, -0.2064627, 0.1919307
2: -0.0955198, 0.1687807, -0.0952442, 0.1669825, -0.2625023, 0.2640249
3: -0.0662145, 0.0543911, -0.0647198, 0.0445989, -0.1108133, 0.1191109
4: -0.1132647, 0.1062221, -0.1078949, 0.1058920, -0.2191566, 0.2141169
5: -0.1053053, 0.1079026, -0.1075574, 0.0915550, -0.1968604, 0.2154599
6: 0.7863125, 1.0312638, 0.7917162, 1.0321604, -0.2458479, 0.2395476
7: -0.1345834, 0.1089936, -0.1271027, 0.1045181, -0.2391015, 0.2360963
8: -0.0852606, 0.1409900, -0.0840127, 0.1290322, -0.2142929, 0.2250027
9: -0.0973673, 0.0852583, -0.0957049, 0.0781229, -0.1754901, 0.1809632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0638772, 0.0660198, -0.0357408, 0.0429055, -0.1067828, 0.1017605
1: -0.0774013, 0.1241177, -0.0491050, 0.0904329, -0.1678343, 0.1732227
2: -0.0955198, 0.1687807, -0.0705103, 0.1306199, -0.2261398, 0.2392910
3: -0.0662145, 0.0543911, -0.0542673, 0.0139103, -0.0801248, 0.1086584
4: -0.1132647, 0.1062221, -0.0789175, 0.0799775, -0.1932422, 0.1851396
5: -0.1053053, 0.1079026, -0.0737076, 0.0785007, -0.1838060, 0.1816102
6: 0.7863125, 1.0312638, 0.8383240, 1.0206236, -0.2343111, 0.1929399
7: -0.1345834, 0.1089936, -0.1013530, 0.0740470, -0.2086304, 0.2103466
8: -0.0852606, 0.1409900, -0.0673169, 0.0956946, -0.1809552, 0.2083070
9: -0.0973673, 0.0852583, -0.0606189, 0.0681067, -0.1654740, 0.1458772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
time: 2.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
time: 1.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0892804, 0.0906576, -0.0241612, 0.0271348, -0.1164152, 0.1148188
1: -0.1029400, 0.1506097, -0.0375350, 0.0695968, -0.1725367, 0.1881447
2: -0.1132331, 0.2055423, -0.0510994, 0.1005710, -0.2138041, 0.2566417
3: -0.0835299, 0.0827910, -0.0440130, 0.0072076, -0.0907375, 0.1268040
4: -0.1471010, 0.1246749, -0.0569864, 0.0619016, -0.2090026, 0.1816613
5: -0.1286921, 0.1372997, -0.0543347, 0.0652583, -0.1939504, 0.1916344
6: 0.7455591, 1.0405499, 0.8755996, 1.0135194, -0.2679603, 0.1649503
7: -0.1624046, 0.1434043, -0.0842433, 0.0479117, -0.2103163, 0.2276475
8: -0.1087741, 0.1768932, -0.0549849, 0.0698630, -0.1786371, 0.2318780
9: -0.1322541, 0.1088458, -0.0400058, 0.0579464, -0.1902005, 0.1488516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1198033, 0.1204593, -0.0402175, 0.0483226, -0.1681260, 0.1606768
1: -0.1337633, 0.1824415, -0.0540417, 0.0991500, -0.2329133, 0.2364831
2: -0.1345164, 0.2499089, -0.0760871, 0.1395568, -0.2740732, 0.3259959
3: -0.1045529, 0.1169145, -0.0566261, 0.0220501, -0.1266030, 0.1735406
4: -0.1881926, 0.1468469, -0.0865698, 0.0858255, -0.2740180, 0.2334167
5: -0.1569170, 0.1726216, -0.0812318, 0.0829547, -0.2398717, 0.2538534
6: 0.6965920, 1.0518944, 0.8267028, 1.0231508, -0.3265588, 0.2251916
7: -0.1960988, 0.1847504, -0.1085491, 0.0808473, -0.2769461, 0.2932994
8: -0.1375105, 0.2200322, -0.0710686, 0.1043784, -0.2418889, 0.2911008
9: -0.1741723, 0.1381169, -0.0684820, 0.0715372, -0.2457095, 0.2065988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
time: 1.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0638772, 0.0660198, -0.0845892, 0.0799165, -0.1437937, 0.1506090
1: -0.0774013, 0.1241177, -0.0984807, 0.1511615, -0.2285628, 0.2225984
2: -0.0955198, 0.1687807, -0.1129812, 0.1986031, -0.2941229, 0.2817619
3: -0.0662145, 0.0543911, -0.0743302, 0.0847113, -0.1509258, 0.1287213
4: -0.1132647, 0.1062221, -0.1423562, 0.1263474, -0.2396121, 0.2485783
5: -0.1053053, 0.1079026, -0.1259774, 0.1319036, -0.2372089, 0.2338800
6: 0.7863125, 1.0312638, 0.7472675, 1.0403537, -0.2540412, 0.2839963
7: -0.1345834, 0.1089936, -0.1607898, 0.1379090, -0.2724924, 0.2697834
8: -0.0852606, 0.1409900, -0.0992392, 0.1655782, -0.2508388, 0.2402293
9: -0.0973673, 0.0852583, -0.1188408, 0.1002212, -0.1975884, 0.2040991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0638772, 0.0660198, -0.0482980, 0.0546283, -0.1185055, 0.1143178
1: -0.0774013, 0.1241177, -0.0623633, 0.1094206, -0.1868219, 0.1864810
2: -0.0955198, 0.1687807, -0.0831438, 0.1512001, -0.2467199, 0.2519245
3: -0.0662145, 0.0543911, -0.0598800, 0.0344400, -0.1006544, 0.1142711
4: -0.1132647, 0.1062221, -0.0974731, 0.0934588, -0.2067235, 0.2036952
5: -0.1053053, 0.1079026, -0.0901624, 0.0918843, -0.1971897, 0.1980650
6: 0.7863125, 1.0312638, 0.8110948, 1.0264063, -0.2400938, 0.2201691
7: -0.1345834, 0.1089936, -0.1189757, 0.0910305, -0.2256139, 0.2279693
8: -0.0852606, 0.1409900, -0.0762542, 0.1164095, -0.2016702, 0.2172442
9: -0.0973673, 0.0852583, -0.0782261, 0.0772622, -0.1746294, 0.1634843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1185531, 0.1192386, -0.0329238, 0.0376646, -0.1562177, 0.1521624
1: -0.1325008, 0.1811376, -0.0466148, 0.0819994, -0.2145001, 0.2277523
2: -0.1336446, 0.2480917, -0.0651543, 0.1226810, -0.2563256, 0.3132459
3: -0.1036917, 0.1155168, -0.0519852, 0.0102294, -0.1139212, 0.1675021
4: -0.1865094, 0.1459387, -0.0728665, 0.0743197, -0.2608291, 0.2188052
5: -0.1557608, 0.1711748, -0.0673624, 0.0756506, -0.2314114, 0.2385372
6: 0.6985979, 1.0514297, 0.8484997, 1.0188029, -0.3202050, 0.2029301
7: -0.1947186, 0.1830569, -0.0957311, 0.0680878, -0.2628064, 0.2787880
8: -0.1363333, 0.2182652, -0.0638174, 0.0884161, -0.2247494, 0.2820826
9: -0.1724553, 0.1369179, -0.0534575, 0.0659200, -0.2383753, 0.1903754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0662946, 0.0682150, -0.0521140, 0.0681331, -0.1344277, 0.1203290
1: -0.0797279, 0.1266387, -0.0687774, 0.1310290, -0.2107568, 0.1954161
2: -0.0972055, 0.1721313, -0.0965043, 0.1688346, -0.2660401, 0.2686357
3: -0.0676981, 0.0570936, -0.0652522, 0.0461838, -0.1138819, 0.1223458
4: -0.1161564, 0.1079781, -0.1093728, 0.1072119, -0.2233684, 0.2173509
5: -0.1074369, 0.1106999, -0.1092889, 0.0922199, -0.1996568, 0.2199889
6: 0.7824345, 1.0320067, 0.7893422, 1.0327530, -0.2503185, 0.2426645
7: -0.1370308, 0.1122681, -0.1284143, 0.1060751, -0.2431059, 0.2406824
8: -0.0871339, 0.1444066, -0.0848641, 0.1307303, -0.2178642, 0.2292707
9: -0.1006871, 0.0868026, -0.0974956, 0.0786330, -0.1793200, 0.1842983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
time: 1.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0662946, 0.0682150, -0.0364308, 0.0440901, -0.1103846, 0.1046458
1: -0.0797279, 0.1266387, -0.0498137, 0.0923391, -0.1720669, 0.1764524
2: -0.0972055, 0.1721313, -0.0717248, 0.1324143, -0.2296197, 0.2438562
3: -0.0676981, 0.0570936, -0.0547831, 0.0150185, -0.0827167, 0.1118767
4: -0.1161564, 0.1079781, -0.0803104, 0.0812562, -0.1974127, 0.1882885
5: -0.1074369, 0.1106999, -0.0752373, 0.0791448, -0.1865817, 0.1859372
6: 0.7824345, 1.0320067, 0.8360240, 1.0210991, -0.2386646, 0.1959827
7: -0.1370308, 0.1122681, -0.1026236, 0.0754574, -0.2124882, 0.2148917
8: -0.0871339, 0.1444066, -0.0681212, 0.0973396, -0.1844735, 0.2125278
9: -0.1006871, 0.0868026, -0.0622832, 0.0686010, -0.1692881, 0.1490858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0884787, 0.0898749, -0.0247185, 0.0275083, -0.1159871, 0.1145933
1: -0.1021304, 0.1497738, -0.0381318, 0.0703087, -0.1724390, 0.1879057
2: -0.1126741, 0.2043770, -0.0519929, 0.1017705, -0.2144445, 0.2563699
3: -0.0829777, 0.0818947, -0.0445241, 0.0073577, -0.0903354, 0.1264188
4: -0.1460217, 0.1240926, -0.0578166, 0.0626806, -0.2087024, 0.1819093
5: -0.1279507, 0.1363720, -0.0549949, 0.0659414, -0.1938921, 0.1913669
6: 0.7468452, 1.0402519, 0.8740098, 1.0138419, -0.2669967, 0.1662420
7: -0.1615197, 0.1423184, -0.0849448, 0.0492096, -0.2107294, 0.2272632
8: -0.1080194, 0.1757600, -0.0555477, 0.0708082, -0.1788277, 0.2313077
9: -0.1311531, 0.1080769, -0.0406402, 0.0584705, -0.1896237, 0.1487172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=10, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1185531, 0.1192386, -0.0415931, 0.0495934, -0.1681465, 0.1608317
1: -0.1325008, 0.1811376, -0.0555180, 0.1011950, -0.2336957, 0.2366555
2: -0.1336446, 0.2480917, -0.0773968, 0.1418713, -0.2755159, 0.3254885
3: -0.1036917, 0.1155168, -0.0571794, 0.0245207, -0.1282124, 0.1726963
4: -0.1865094, 0.1459387, -0.0887150, 0.0871974, -0.2737068, 0.2346537
5: -0.1557608, 0.1711748, -0.0830316, 0.0844493, -0.2402101, 0.2542064
6: 0.6985979, 1.0514297, 0.8236476, 1.0237666, -0.3251687, 0.2277821
7: -0.1947186, 0.1830569, -0.1106503, 0.0824657, -0.2771842, 0.2937072
8: -0.1363333, 0.2182652, -0.0719536, 0.1067616, -0.2430950, 0.2902188
9: -0.1724553, 0.1369179, -0.0703432, 0.0726909, -0.2451462, 0.2072610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=44, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0662946, 0.0682150, -0.0959076, 0.0812030, -0.1474976, 0.1641226
1: -0.0797279, 0.1266387, -0.1051131, 0.1532850, -0.2330128, 0.2317518
2: -0.0972055, 0.1721313, -0.1210750, 0.2010149, -0.2982204, 0.2932064
3: -0.0676981, 0.0570936, -0.0750654, 0.0953639, -0.1630621, 0.1321590
4: -0.1161564, 0.1079781, -0.1446398, 0.1318370, -0.2479935, 0.2526179
5: -0.1074369, 0.1106999, -0.1366452, 0.1339397, -0.2413765, 0.2473451
6: 0.7824345, 1.0320067, 0.7401994, 1.0410634, -0.2586290, 0.2918073
7: -0.1370308, 0.1122681, -0.1629171, 0.1449144, -0.2819452, 0.2751852
8: -0.0871339, 0.1444066, -0.1004087, 0.1927879, -0.2799218, 0.2448152
9: -0.1006871, 0.0868026, -0.1353138, 0.1013893, -0.2020763, 0.2221165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0662946, 0.0682150, -0.0501597, 0.0558908, -0.1221854, 0.1183747
1: -0.0797279, 0.1266387, -0.0641917, 0.1115045, -0.1912324, 0.1908304
2: -0.0972055, 0.1721313, -0.0846681, 0.1535666, -0.2507721, 0.2567994
3: -0.0676981, 0.0570936, -0.0606015, 0.0369923, -0.1046904, 0.1176950
4: -0.1161564, 0.1079781, -0.0997139, 0.0951208, -0.2112773, 0.2076920
5: -0.1074369, 0.1106999, -0.0919970, 0.0938822, -0.2013191, 0.2026970
6: 0.7824345, 1.0320067, 0.8078881, 1.0271025, -0.2446680, 0.2241186
7: -0.1370308, 0.1122681, -0.1210632, 0.0933952, -0.2304260, 0.2333313
8: -0.0871339, 0.1444066, -0.0774017, 0.1189944, -0.2061283, 0.2218083
9: -0.1006871, 0.0868026, -0.0803296, 0.0784084, -0.1790954, 0.1671323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1022070, 0.1039772, -0.0307960, 0.0335090, -0.1357161, 0.1347731
1: -0.1183036, 0.1590382, -0.0446403, 0.0780847, -0.1963883, 0.2036785
2: -0.1260882, 0.2140334, -0.0617364, 0.1163861, -0.2424743, 0.2757698
3: -0.0862392, 0.1034016, -0.0501758, 0.0089945, -0.0952337, 0.1535774
4: -0.1508259, 0.1380669, -0.0680694, 0.0711764, -0.2220024, 0.2061363
5: -0.1427931, 0.1545320, -0.0632111, 0.0733907, -0.2161838, 0.2177432
6: 0.7209831, 1.0429857, 0.8556657, 1.0173608, -0.2963777, 0.1873199
7: -0.1764792, 0.1520085, -0.0926832, 0.0633643, -0.2398435, 0.2446917
8: -0.1105978, 0.2029491, -0.0616848, 0.0826462, -0.1932440, 0.2646340
9: -0.1465038, 0.1045537, -0.0487260, 0.0641861, -0.2106899, 0.1532797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2670607, upper bound: 0.2662904
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2653607, upper bound: 0.2658532
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0915567, 0.0934526, -0.0297917, 0.0312455, -0.1228022, 0.1232443
1: -0.1069626, 0.1493115, -0.0435648, 0.0767916, -0.1837542, 0.1928763
2: -0.1176125, 0.2014300, -0.0601263, 0.1129573, -0.2305698, 0.2615563
3: -0.0806671, 0.0898124, -0.0491902, 0.0087241, -0.0893911, 0.1390025
4: -0.1403669, 0.1292372, -0.0655835, 0.0697725, -0.2101395, 0.1948207
5: -0.1323989, 0.1415861, -0.0611818, 0.0721597, -0.2045586, 0.2027679
6: 0.7391177, 1.0397354, 0.8593620, 1.0167792, -0.2776614, 0.1803734
7: -0.1648627, 0.1400145, -0.0913465, 0.0610253, -0.2258880, 0.2313610
8: -0.1035557, 0.1857696, -0.0606707, 0.0796801, -0.1832358, 0.2464403
9: -0.1328343, 0.0992036, -0.0466197, 0.0632416, -0.1960759, 0.1458233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=11, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2668597, upper bound: 0.2657769
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2651776, upper bound: 0.2652455
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0572997, 0.0598412, -0.0599442, 0.0665474, -0.1238471, 0.1197854
1: -0.0705784, 0.1180252, -0.0752141, 0.1284770, -0.1990555, 0.1932394
2: -0.0903502, 0.1609720, -0.0948699, 0.1727500, -0.2631002, 0.2558419
3: -0.0628589, 0.0461025, -0.0645617, 0.0574813, -0.1203402, 0.1106642
4: -0.1067254, 0.1008512, -0.1173352, 0.1055000, -0.2122254, 0.2181865
5: -0.0989653, 0.1001341, -0.1070431, 0.1043899, -0.2033553, 0.2071772
6: 0.7973238, 1.0292815, 0.7828866, 1.0319841, -0.2346603, 0.2463949
7: -0.1275954, 0.1014359, -0.1386838, 0.1040557, -0.2316511, 0.2401197
8: -0.0809924, 0.1305115, -0.0837599, 0.1385573, -0.2195497, 0.2142714
9: -0.0889111, 0.0819951, -0.0951731, 0.0880834, -0.1769945, 0.1771682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2646901
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2646901
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0572997, 0.0598412, -0.0370790, 0.0450837, -0.1023835, 0.0969203
1: -0.0705784, 0.1180252, -0.0505974, 0.0939381, -0.1645165, 0.1686226
2: -0.0903502, 0.1609720, -0.0727490, 0.1339195, -0.2242696, 0.2337209
3: -0.0628589, 0.0461025, -0.0552158, 0.0163066, -0.0791655, 0.1013183
4: -0.1067254, 0.1008512, -0.0815115, 0.0823289, -0.1890543, 0.1823627
5: -0.0989653, 0.1001341, -0.0766446, 0.0796852, -0.1786505, 0.1767788
6: 0.7973238, 1.0292815, 0.8340948, 1.0215807, -0.2242569, 0.1951867
7: -0.1275954, 0.1014359, -0.1036895, 0.0767227, -0.2043181, 0.2051254
8: -0.0809924, 0.1305115, -0.0688132, 0.0987196, -0.1797121, 0.1993246
9: -0.0889111, 0.0819951, -0.0637384, 0.0690156, -0.1579267, 0.1457335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2724788
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2724788
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1054612, 0.1071928, -0.0424877, 0.0504200, -0.1558812, 0.1496805
1: -0.1217688, 0.1620101, -0.0564782, 0.1025252, -0.2242939, 0.2184883
2: -0.1286779, 0.2178841, -0.0782487, 0.1433768, -0.2720547, 0.2961328
3: -0.0879417, 0.1075537, -0.0575393, 0.0261276, -0.1140693, 0.1650930
4: -0.1540216, 0.1407646, -0.0901104, 0.0880897, -0.2421113, 0.2308750
5: -0.1459690, 0.1584875, -0.0842022, 0.0854215, -0.2313906, 0.2426897
6: 0.7154424, 1.0439788, 0.8216604, 1.0241671, -0.3087247, 0.2223184
7: -0.1800286, 0.1556731, -0.1120171, 0.0835182, -0.2635468, 0.2676902
8: -0.1127494, 0.2081982, -0.0725292, 0.1083118, -0.2210612, 0.2807274
9: -0.1506803, 0.1061884, -0.0715537, 0.0734414, -0.2241218, 0.1777420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2721543, upper bound: 0.2718940
time: 1.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2681158, upper bound: 0.2710319
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0572997, 0.0598412, -0.0924975, 0.0793161, -0.1366159, 0.1523387
1: -0.0705784, 0.1180252, -0.1020627, 0.1501707, -0.2207491, 0.2200880
2: -0.0903502, 0.1609720, -0.1183612, 0.1974780, -0.2878281, 0.2793331
3: -0.0628589, 0.0461025, -0.0739873, 0.0910128, -0.1538717, 0.1200898
4: -0.1067254, 0.1008512, -0.1412908, 0.1291002, -0.2358256, 0.2421421
5: -0.0989653, 0.1001341, -0.1333170, 0.1309537, -0.2299190, 0.2334512
6: 0.7973238, 1.0292815, 0.7452451, 1.0400225, -0.2426987, 0.2840364
7: -0.1275954, 0.1014359, -0.1597973, 0.1410740, -0.2686694, 0.2612332
8: -0.0809924, 0.1305115, -0.0986937, 0.1872870, -0.2682794, 0.2292052
9: -0.0889111, 0.0819951, -0.1312151, 0.0996762, -0.1885874, 0.2132102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2483588, upper bound: 0.2563704
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2747176
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0572997, 0.0598412, -0.0513475, 0.0565481, -0.1138478, 0.1111887
1: -0.0705784, 0.1180252, -0.0652542, 0.1125893, -0.1831678, 0.1832794
2: -0.0903502, 0.1609720, -0.0856134, 0.1547986, -0.2451487, 0.2465854
3: -0.0628589, 0.0461025, -0.0609770, 0.0385078, -0.1013667, 0.1070795
4: -0.1067254, 0.1008512, -0.1008804, 0.0960742, -0.2027996, 0.2017316
5: -0.0989653, 0.1001341, -0.0931564, 0.0949223, -0.1938876, 0.1932905
6: 0.7973238, 1.0292815, 0.8061306, 1.0274649, -0.2301410, 0.2231508
7: -0.1275954, 0.1014359, -0.1221499, 0.0947329, -0.2223283, 0.2235858
8: -0.0809924, 0.1305115, -0.0779990, 0.1209106, -0.2019030, 0.2085105
9: -0.0889111, 0.0819951, -0.0817573, 0.0790050, -0.1679161, 0.1637524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2483588, upper bound: 0.2662662
time: 2.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1046644, 0.1064055, -0.0339039, 0.0394882, -0.1441526, 0.1403095
1: -0.1209205, 0.1612825, -0.0474813, 0.0849339, -0.2058543, 0.2087637
2: -0.1280439, 0.2169415, -0.0670179, 0.1254434, -0.2534873, 0.2839594
3: -0.0875249, 0.1065372, -0.0527793, 0.0115102, -0.0990351, 0.1593165
4: -0.1532393, 0.1401041, -0.0749720, 0.0762883, -0.2295276, 0.2150761
5: -0.1451915, 0.1575190, -0.0695703, 0.0766423, -0.2218338, 0.2270893
6: 0.7167990, 1.0437357, 0.8449589, 1.0194364, -0.3026373, 0.1987768
7: -0.1791596, 0.1547759, -0.0976873, 0.0701613, -0.2493209, 0.2524632
8: -0.1122227, 0.2069131, -0.0650351, 0.0909487, -0.2031713, 0.2719481
9: -0.1496578, 0.1057882, -0.0559494, 0.0666809, -0.2163387, 0.1617376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=46, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2707743, upper bound: 0.2693036
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684331
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1046644, 0.1064055, -0.0355394, 0.0425310, -0.1471954, 0.1419450
1: -0.1209205, 0.1612825, -0.0489270, 0.0898303, -0.2107508, 0.2102095
2: -0.1280439, 0.2169415, -0.0701276, 0.1300525, -0.2580964, 0.2870691
3: -0.0875249, 0.1065372, -0.0541042, 0.0136473, -0.1011722, 0.1606414
4: -0.1532393, 0.1401041, -0.0784851, 0.0795732, -0.2328125, 0.2185893
5: -0.1451915, 0.1575190, -0.0732541, 0.0782970, -0.2234886, 0.2307732
6: 0.7167990, 1.0437357, 0.8390511, 1.0204935, -0.3036945, 0.2046846
7: -0.1791596, 0.1547759, -0.1009513, 0.0736213, -0.2527809, 0.2557272
8: -0.1122227, 0.2069131, -0.0670668, 0.0951745, -0.2073971, 0.2739799
9: -0.1496578, 0.1057882, -0.0601072, 0.0679505, -0.2176083, 0.1658954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2707743, upper bound: 0.2693036
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684331
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0593296, 0.0616062, -0.0612744, 0.0677763, -0.1271059, 0.1228805
1: -0.0726455, 0.1198790, -0.0766418, 0.1304546, -0.2031001, 0.1965208
2: -0.0919656, 0.1632929, -0.0961365, 0.1749883, -0.2669538, 0.2594295
3: -0.0638062, 0.0486925, -0.0650968, 0.0598705, -0.1236767, 0.1137893
4: -0.1087190, 0.1025194, -0.1194097, 0.1068267, -0.2155456, 0.2219290
5: -0.1009464, 0.1024131, -0.1087835, 0.1058353, -0.2067818, 0.2111967
6: 0.7939912, 1.0299009, 0.7799321, 1.0325799, -0.2385887, 0.2499688
7: -0.1297120, 0.1037218, -0.1407160, 0.1056206, -0.2353325, 0.2444378
8: -0.0822468, 0.1337859, -0.0846157, 0.1408619, -0.2231087, 0.2184016
9: -0.0914712, 0.0830148, -0.0969730, 0.0891992, -0.1806704, 0.1799877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2646901
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2646901
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0593296, 0.0616062, -0.0379896, 0.0462643, -0.1055939, 0.0995957
1: -0.0726455, 0.1198790, -0.0516504, 0.0958379, -0.1684834, 0.1715294
2: -0.0919656, 0.1632929, -0.0739658, 0.1358079, -0.2277735, 0.2372587
3: -0.0638062, 0.0486925, -0.0557298, 0.0180484, -0.0818546, 0.1044223
4: -0.1087190, 0.1025194, -0.0830951, 0.0836035, -0.1923224, 0.1856144
5: -0.1009464, 0.1024131, -0.0783166, 0.0805338, -0.1814802, 0.1807298
6: 0.7939912, 1.0299009, 0.8316514, 1.0221530, -0.2281618, 0.1982495
7: -0.1297120, 0.1037218, -0.1051456, 0.0782262, -0.2079382, 0.2088674
8: -0.0822468, 0.1337859, -0.0696353, 0.1005181, -0.1827649, 0.2034212
9: -0.0914712, 0.0830148, -0.0654675, 0.0696684, -0.1611397, 0.1484822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2724788
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2724788
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1046644, 0.1064055, -0.0440931, 0.0516983, -0.1563627, 0.1504986
1: -0.1209205, 0.1612825, -0.0581785, 0.1045843, -0.2255047, 0.2194610
2: -0.1280439, 0.2169415, -0.0796867, 0.1457078, -0.2737517, 0.2966281
3: -0.0875249, 0.1065372, -0.0582058, 0.0286152, -0.1161401, 0.1647429
4: -0.1532393, 0.1401041, -0.0922727, 0.0896481, -0.2428874, 0.2323768
5: -0.1451915, 0.1575190, -0.0860127, 0.0872474, -0.2324390, 0.2435317
6: 0.7167990, 1.0437357, 0.8184901, 1.0247903, -0.3079913, 0.2252456
7: -0.1791596, 0.1547759, -0.1141309, 0.0855988, -0.2647583, 0.2689068
8: -0.1122227, 0.2069131, -0.0735911, 0.1107126, -0.2229353, 0.2805041
9: -0.1496578, 0.1057882, -0.0735203, 0.0746020, -0.2242599, 0.1793085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=50, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2721543, upper bound: 0.2718984
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2681158, upper bound: 0.2710343
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0593296, 0.0616062, -0.0949166, 0.0806546, -0.1399842, 0.1565228
1: -0.0726455, 0.1198790, -0.1042266, 0.1523800, -0.2250255, 0.2241057
2: -0.0919656, 0.1632929, -0.1202864, 0.1999868, -0.2919524, 0.2835793
3: -0.0638062, 0.0486925, -0.0747522, 0.0940995, -0.1579056, 0.1234446
4: -0.1087190, 0.1025194, -0.1436665, 0.1310417, -0.2397607, 0.2461859
5: -0.1009464, 0.1024131, -0.1356780, 0.1330720, -0.2340184, 0.2380911
6: 0.7939912, 1.0299009, 0.7416657, 1.0407609, -0.2467697, 0.2882352
7: -0.1297120, 0.1037218, -0.1620105, 0.1437984, -0.2735104, 0.2657323
8: -0.0822468, 0.1337859, -0.0999103, 0.1911893, -0.2734361, 0.2336961
9: -0.0914712, 0.0830148, -0.1341227, 0.1008915, -0.1923627, 0.2171374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=50, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2483524, upper bound: 0.2563121
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2747176
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0593296, 0.0616062, -0.0536410, 0.0578170, -0.1171466, 0.1152472
1: -0.0726455, 0.1198790, -0.0673057, 0.1146839, -0.1873294, 0.1871848
2: -0.0919656, 0.1632929, -0.0874385, 0.1571774, -0.2491429, 0.2507315
3: -0.0638062, 0.0486925, -0.0617021, 0.0414342, -0.1052404, 0.1103946
4: -0.1087190, 0.1025194, -0.1031326, 0.0979149, -0.2066338, 0.2056520
5: -0.1009464, 0.1024131, -0.0953947, 0.0969305, -0.1978770, 0.1978078
6: 0.7939912, 1.0299009, 0.8027371, 1.0281649, -0.2341737, 0.2271638
7: -0.1297120, 0.1037218, -0.1242482, 0.0973157, -0.2270277, 0.2279700
8: -0.0822468, 0.1337859, -0.0791524, 0.1246100, -0.2068568, 0.2129383
9: -0.0914712, 0.0830148, -0.0845139, 0.0801572, -0.1716284, 0.1675286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=45, inp2_unstable=45, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2483524, upper bound: 0.2662434
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
time: 1.31 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.11 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2475137, upper bound: 0.2553120
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2735462, upper bound: 0.2712177
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2475137, upper bound: 0.2590214
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2735462, upper bound: 0.2747244
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2648312
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2648312
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2684923
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2683649
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753131
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753131
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753898
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753898
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2463033, upper bound: 0.2548496
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2735248, upper bound: 0.2712177
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2463033, upper bound: 0.2579004
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2735248, upper bound: 0.2746866
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2648312
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2648312
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2684189
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2682666
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753131
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753131
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753898
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753898
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2651781, upper bound: 0.2604437
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2604437
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2651781, upper bound: 0.2604437
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2654327
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2582777, upper bound: 0.2515041
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2577288, upper bound: 0.2515041
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2582777, upper bound: 0.2637625
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2577288, upper bound: 0.2634628
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2688290, upper bound: 0.2689333
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2647270, upper bound: 0.2678041
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2688290, upper bound: 0.2699453
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2647270, upper bound: 0.2686479
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2505233, upper bound: 0.2608429
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747039, upper bound: 0.2747000
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2505233, upper bound: 0.2624015
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747039, upper bound: 0.2753898
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2651773, upper bound: 0.2604437
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2604437
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2651773, upper bound: 0.2654294
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2654150
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2646742, upper bound: 0.2604437
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2643144, upper bound: 0.2604437
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2646742, upper bound: 0.2652622
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2643144, upper bound: 0.2652492
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2687752, upper bound: 0.2689225
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2646901, upper bound: 0.2677799
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2687752, upper bound: 0.2698907
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2646901, upper bound: 0.2685597
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2479369, upper bound: 0.2603138
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747000, upper bound: 0.2747000
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2479369, upper bound: 0.2613184
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747000, upper bound: 0.2753898
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2670607, upper bound: 0.2662904
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2653607, upper bound: 0.2658532
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2668597, upper bound: 0.2657769
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2651776, upper bound: 0.2652455
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2646901
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2646901
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2724788
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2724788
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2721543, upper bound: 0.2718940
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2681158, upper bound: 0.2710319
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2483588, upper bound: 0.2563704
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2747176
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2483588, upper bound: 0.2662662
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2707743, upper bound: 0.2693036
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684331
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2707743, upper bound: 0.2693036
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684331
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2646901
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2646901
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2724788
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2724788
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2721543, upper bound: 0.2718984
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2681158, upper bound: 0.2710343
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2483524, upper bound: 0.2563121
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2747176
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2483524, upper bound: 0.2662434
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.11
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0301614, 0.0320787, -0.0700760, 0.0669105, -0.0970719, 0.1021547
1: -0.0439607, 0.0772676, -0.0820068, 0.1296936, -0.1736543, 0.1592744
2: -0.0607190, 0.1142195, -0.1005179, 0.1742233, -0.2349422, 0.2147374
3: -0.0495530, 0.0088236, -0.0668984, 0.0624043, -0.1119572, 0.0757220
4: -0.0664985, 0.0702893, -0.1192723, 0.1111052, -0.1776038, 0.1895617
5: -0.0619288, 0.0726128, -0.1114347, 0.1113212, -0.1732500, 0.1840475
6: 0.8580014, 1.0169933, 0.7784198, 1.0331804, -0.1751789, 0.2385735
7: -0.0918386, 0.0618864, -0.1392843, 0.1158241, -0.2076627, 0.2011706
8: -0.0610440, 0.0807719, -0.0874178, 0.1511204, -0.2121644, 0.1681897
9: -0.0473950, 0.0635892, -0.1042670, 0.0884130, -0.1358081, 0.1678563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2681665, upper bound: 0.2683644
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2671463, upper bound: 0.2640199
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0167887, 0.0221933, -0.0674328, 0.0654480, -0.0822367, 0.0896261
1: -0.0296397, 0.0601789, -0.0796424, 0.1272797, -0.1569195, 0.1398212
2: -0.0392798, 0.0847042, -0.0984143, 0.1714817, -0.2107615, 0.1831186
3: -0.0372522, 0.0052220, -0.0660627, 0.0590317, -0.0962839, 0.0712847
4: -0.0460040, 0.0515955, -0.1166765, 0.1089839, -0.1549879, 0.1682721
5: -0.0456022, 0.0562219, -0.1088549, 0.1090067, -0.1546089, 0.1650768
6: 0.8966302, 1.0092508, 0.7823308, 1.0323737, -0.1357435, 0.2269200
7: -0.0749625, 0.0307410, -0.1368660, 0.1128474, -0.1878099, 0.1676070
8: -0.0475401, 0.0573590, -0.0860884, 0.1468567, -0.1943968, 0.1434474
9: -0.0316127, 0.0510131, -0.1010901, 0.0870852, -0.1186980, 0.1521032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2487065, upper bound: 0.2578167
time: 1.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2477163, upper bound: 0.2567468
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0301614, 0.0320787, -0.0935822, 0.0799165, -0.1100778, 0.1256609
1: -0.0439607, 0.0772676, -0.1030330, 0.1511615, -0.1951222, 0.1803006
2: -0.0607190, 0.1142195, -0.1192245, 0.1986031, -0.2593220, 0.2334440
3: -0.0495530, 0.0088236, -0.0743302, 0.0923970, -0.1419499, 0.0831538
4: -0.0664985, 0.0702893, -0.1423562, 0.1299709, -0.1964694, 0.2126455
5: -0.0619288, 0.0726128, -0.1343758, 0.1319036, -0.1938324, 0.2069887
6: 0.8580014, 1.0169933, 0.7436399, 1.0403537, -0.1823522, 0.2733533
7: -0.0918386, 0.0618864, -0.1607898, 0.1422956, -0.2341342, 0.2226762
8: -0.0610440, 0.0807719, -0.0992392, 0.1890369, -0.2500809, 0.1800111
9: -0.0473950, 0.0635892, -0.1325190, 0.1002212, -0.1476162, 0.1961083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=44, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2496743, upper bound: 0.2486120
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2496743, upper bound: 0.2747244
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0288648, 0.0302875, -0.0668315, 0.0651153, -0.0939801, 0.0971189
1: -0.0425722, 0.0756054, -0.0791045, 0.1267304, -0.1693026, 0.1547099
2: -0.0586403, 0.1106940, -0.0979358, 0.1708580, -0.2294983, 0.2086298
3: -0.0483265, 0.0084744, -0.0658725, 0.0582645, -0.1065909, 0.0743469
4: -0.0639932, 0.0684768, -0.1160862, 0.1085012, -0.1724944, 0.1845630
5: -0.0599061, 0.0710236, -0.1082680, 0.1084803, -0.1683864, 0.1792916
6: 0.8621823, 1.0162426, 0.7832205, 1.0321901, -0.1700078, 0.2330221
7: -0.0901644, 0.0588665, -0.1363159, 0.1121701, -0.2023345, 0.1951824
8: -0.0597347, 0.0778406, -0.0857860, 0.1458868, -0.2056215, 0.1636267
9: -0.0453606, 0.0623699, -0.1003673, 0.0867833, -0.1321438, 0.1627372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=46, inp2_unstable=49, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=21, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 203

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2652846, upper bound: 0.2644577
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2647612, upper bound: 0.2605533
time: 1.39 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.39 seconds
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2681665, upper bound: 0.2683644
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2671463, upper bound: 0.2640199
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2487065, upper bound: 0.2578167
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2477163, upper bound: 0.2567468
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2496743, upper bound: 0.2486120
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2496743, upper bound: 0.2747244
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2652846, upper bound: 0.2644577
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.39
Output dim: 6, lower bound: -0.2647612, upper bound: 0.2605533
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2648312
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2684923
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2683649
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2738640
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2714420, upper bound: 0.2746704
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753131
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753131
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753898
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750855, upper bound: 0.2753898
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2735248, upper bound: 0.2712177
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2735248, upper bound: 0.2746866
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2648312
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2648312
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2682744, upper bound: 0.2684189
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2677887, upper bound: 0.2682666
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2738634
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2712192, upper bound: 0.2746282
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753131
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753131
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753898
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2750752, upper bound: 0.2753898
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2651781, upper bound: 0.2604437
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2604437
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2651781, upper bound: 0.2604437
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2654327
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2582777, upper bound: 0.2515041
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2582777, upper bound: 0.2637625
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2577288, upper bound: 0.2634628
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2688290, upper bound: 0.2689333
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2647270, upper bound: 0.2678041
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2688290, upper bound: 0.2699453
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2647270, upper bound: 0.2686479
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2505233, upper bound: 0.2608429
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747039, upper bound: 0.2747000
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2505233, upper bound: 0.2624015
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747039, upper bound: 0.2753898
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2651773, upper bound: 0.2604437
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2604437
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2651773, upper bound: 0.2654294
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2647598, upper bound: 0.2654150
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2646742, upper bound: 0.2604437
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2643144, upper bound: 0.2604437
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2646742, upper bound: 0.2652622
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2643144, upper bound: 0.2652492
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2687752, upper bound: 0.2689225
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2646901, upper bound: 0.2677799
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2687752, upper bound: 0.2698907
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2646901, upper bound: 0.2685597
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2479369, upper bound: 0.2603138
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747000, upper bound: 0.2747000
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2479369, upper bound: 0.2613184
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747000, upper bound: 0.2753898
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2742019, upper bound: 0.2733800
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2539348
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747129, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2742065, upper bound: 0.2733902
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2720479
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2746348, upper bound: 0.2753131
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2597768, upper bound: 0.2540045
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2747176, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2670607, upper bound: 0.2662904
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2653607, upper bound: 0.2658532
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2668597, upper bound: 0.2657769
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2651776, upper bound: 0.2652455
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2646901
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2646901
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2724788
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2724788
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2721543, upper bound: 0.2718940
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2681158, upper bound: 0.2710319
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2747176
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2483588, upper bound: 0.2662662
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2707743, upper bound: 0.2693036
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684331
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2707743, upper bound: 0.2693036
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2682534, upper bound: 0.2684331
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2646901
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2646901
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2689601, upper bound: 0.2724788
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2685597, upper bound: 0.2724788
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2721543, upper bound: 0.2718984
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2681158, upper bound: 0.2710343
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2747176
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2483524, upper bound: 0.2662434
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.39
Output dim: 6, lower bound: -0.2753898, upper bound: 0.2753898

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.00 + 596.19 = 600.19 seconds
