## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.45381564


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1188473, 0.1115622, -0.1188473, 0.1115622, -0.2304095, 0.2304095)
1: (-0.1388211, 0.1340919, -0.1388211, 0.1340919, -0.2729130, 0.2729130)
2: (-0.1108225, 0.2044321, -0.1108225, 0.2044321, -0.3152546, 0.3152546)
3: (-0.1114923, 0.3334921, -0.1114923, 0.3334921, -0.4351088, 0.4351088)
4: (-0.1245532, 0.1320058, -0.1245532, 0.1320058, -0.2565590, 0.2565590)
5: (-0.1196867, 0.1536920, -0.1196867, 0.1536920, -0.2733786, 0.2733786)
6: (-0.1380765, 0.1495728, -0.1380765, 0.1495728, -0.2876493, 0.2876493)
7: (0.4551308, 1.2201583, 0.4551308, 1.2201583, -0.7650275, 0.7650275)
8: (-0.1529319, 0.1898467, -0.1529319, 0.1898467, -0.3312583, 0.3312583)
9: (-0.1462240, 0.1769303, -0.1462240, 0.1769303, -0.3231543, 0.3231543)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.29 + 2.57 = 3.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5042396, upper bound: 0.5042396

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5023537, upper bound: 0.4987494
time: 1.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5023537, upper bound: 0.5023537
time: 1.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.32
Output dim: 7, lower bound: -0.5023537, upper bound: 0.4987494
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.32
Output dim: 7, lower bound: -0.5023537, upper bound: 0.5023537

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.1137624, 0.1053849, -0.1179813, 0.1101806, -0.2239430, 0.2233662
1: -0.1333474, 0.1264705, -0.1374423, 0.1326095, -0.2659569, 0.2639128
2: -0.0957818, 0.1931498, -0.1042600, 0.2011875, -0.2969692, 0.2974098
3: -0.1011733, 0.3292410, -0.1069925, 0.3327902, -0.4240527, 0.4262560
4: -0.1162999, 0.1266064, -0.1226705, 0.1306857, -0.2469856, 0.2492770
5: -0.1134919, 0.1466847, -0.1186215, 0.1523528, -0.2658446, 0.2653061
6: -0.1297890, 0.1398134, -0.1359096, 0.1464612, -0.2762501, 0.2757229
7: 0.4604659, 1.1938167, 0.4564869, 1.2084773, -0.7480114, 0.7373297
8: -0.1371465, 0.1805816, -0.1460013, 0.1876132, -0.3127587, 0.3144639
9: -0.1384123, 0.1673063, -0.1440685, 0.1746108, -0.3130231, 0.3113748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4978769, upper bound: 0.4941241
time: 1.74 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4944776
time: 1.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.1182157, 0.1106787, -0.1186461, 0.1112809, -0.2294966, 0.2293248
1: -0.1379064, 0.1330914, -0.1385290, 0.1337742, -0.2716806, 0.2716203
2: -0.1068675, 0.2024228, -0.1095517, 0.2037896, -0.3106571, 0.3119746
3: -0.1085434, 0.3328443, -0.1105367, 0.3332894, -0.4308138, 0.4335088
4: -0.1232962, 0.1311522, -0.1241505, 0.1317341, -0.2550304, 0.2553027
5: -0.1189359, 0.1527949, -0.1194479, 0.1534068, -0.2723426, 0.2722428
6: -0.1366822, 0.1476558, -0.1376319, 0.1489583, -0.2856405, 0.2852876
7: 0.4562120, 1.2134137, 0.4554717, 1.2179892, -0.7617772, 0.7579420
8: -0.1486503, 0.1884146, -0.1515492, 0.1893904, -0.3240613, 0.3284087
9: -0.1448846, 0.1754513, -0.1457964, 0.1764586, -0.3213432, 0.3212477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4978769, upper bound: 0.4977015
time: 1.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4978512
time: 1.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.60 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 7, lower bound: -0.4978769, upper bound: 0.4941241
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4944776
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 7, lower bound: -0.4978769, upper bound: 0.4977015
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.60
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4978512

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.1098652, 0.1019280, -0.1067651, 0.1003221, -0.2101874, 0.2086931
1: -0.1299324, 0.1201803, -0.1276082, 0.1145421, -0.2444745, 0.2477885
2: -0.0948636, 0.1882578, -0.1016612, 0.1871525, -0.2820162, 0.2899189
3: -0.1001671, 0.3194932, -0.1042887, 0.3049699, -0.3952395, 0.4126037
4: -0.1115456, 0.1230549, -0.1090414, 0.1204789, -0.2320245, 0.2320963
5: -0.1086522, 0.1417175, -0.1047622, 0.1381355, -0.2467876, 0.2464796
6: -0.1251002, 0.1359971, -0.1224134, 0.1354641, -0.2605643, 0.2584105
7: 0.4719100, 1.1927882, 0.4892060, 1.2055507, -0.7336407, 0.7035823
8: -0.1360022, 0.1754411, -0.1428718, 0.1728422, -0.2967338, 0.3040006
9: -0.1341512, 0.1621232, -0.1318080, 0.1597919, -0.2939431, 0.2939312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4737412
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4694741
time: 1.51 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.1068032, 0.0991785, -0.1044018, 0.0986944, -0.2054976, 0.2035803
1: -0.1272620, 0.1152947, -0.1258007, 0.1103986, -0.2376607, 0.2410954
2: -0.0941785, 0.1844349, -0.1054231, 0.1855578, -0.2797363, 0.2898580
3: -0.0994018, 0.3120946, -0.1068486, 0.2964579, -0.3860172, 0.4078335
4: -0.1078501, 0.1202805, -0.1068019, 0.1184680, -0.2263180, 0.2270824
5: -0.1048375, 0.1378190, -0.1017456, 0.1352547, -0.2400922, 0.2395646
6: -0.1214329, 0.1330324, -0.1201434, 0.1344759, -0.2559088, 0.2531758
7: 0.4806227, 1.1920898, 0.4993515, 1.2123797, -0.7317570, 0.6927383
8: -0.1351364, 0.1713741, -0.1467970, 0.1704036, -0.2934677, 0.3040534
9: -0.1308537, 0.1580205, -0.1297734, 0.1574509, -0.2883046, 0.2877939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4744452
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
time: 1.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.1142935, 0.1072338, -0.1073780, 0.1013941, -0.2156876, 0.2146117
1: -0.1344761, 0.1268130, -0.1286448, 0.1156494, -0.2501255, 0.2554578
2: -0.1059432, 0.1975113, -0.1069330, 0.1896675, -0.2956108, 0.3044443
3: -0.1075895, 0.3231943, -0.1078874, 0.3054771, -0.4020540, 0.4200312
4: -0.1185263, 0.1276021, -0.1104511, 0.1214890, -0.2400153, 0.2380532
5: -0.1141060, 0.1478372, -0.1055474, 0.1391593, -0.2532653, 0.2533846
6: -0.1319747, 0.1438089, -0.1240667, 0.1378781, -0.2698528, 0.2678756
7: 0.4675605, 1.2123502, 0.4882182, 1.2150204, -0.7474599, 0.7241320
8: -0.1475388, 0.1832459, -0.1484491, 0.1745448, -0.3079421, 0.3180327
9: -0.1406110, 0.1702733, -0.1334730, 0.1615884, -0.3021994, 0.3037463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4777546
time: 1.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728350, upper bound: 0.4726490
time: 1.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.1110510, 0.1043784, -0.1050538, 0.0997538, -0.2108048, 0.2094322
1: -0.1316796, 0.1216547, -0.1268203, 0.1114963, -0.2431759, 0.2484750
2: -0.1052363, 0.1934648, -0.1106415, 0.1880075, -0.2932438, 0.3041062
3: -0.1068461, 0.3155254, -0.1104606, 0.2969918, -0.3929074, 0.4150163
4: -0.1145821, 0.1246936, -0.1081959, 0.1194590, -0.2340411, 0.2328895
5: -0.1101135, 0.1437805, -0.1025436, 0.1362774, -0.2463909, 0.2463241
6: -0.1281200, 0.1406627, -0.1217680, 0.1368494, -0.2649695, 0.2624307
7: 0.4765947, 1.2115552, 0.4983739, 1.2217712, -0.7451765, 0.7131814
8: -0.1466791, 0.1789758, -0.1523649, 0.1721111, -0.3047702, 0.3177403
9: -0.1371137, 0.1660035, -0.1313991, 0.1592259, -0.2963396, 0.2974026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4781554
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728350, upper bound: 0.4728350
time: 2.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4737412
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4694741
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4744452
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4777546
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4728350, upper bound: 0.4726490
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4751232, upper bound: 0.4781554
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.88
Output dim: 7, lower bound: -0.4728350, upper bound: 0.4728350

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0999883, 0.0929706, -0.1067651, 0.1003221, -0.2003104, 0.1997358
1: -0.1216323, 0.1051631, -0.1276082, 0.1145421, -0.2361744, 0.2327713
2: -0.0929622, 0.1760693, -0.1016612, 0.1871525, -0.2801147, 0.2777304
3: -0.0981591, 0.2993918, -0.1042887, 0.3049699, -0.3920397, 0.3924900
4: -0.0995975, 0.1145196, -0.1090414, 0.1204789, -0.2200764, 0.2235610
5: -0.0965260, 0.1292124, -0.1047622, 0.1381355, -0.2346615, 0.2339746
6: -0.1136280, 0.1267890, -0.1224134, 0.1354641, -0.2490921, 0.2492023
7: 0.4947502, 1.1910176, 0.4892060, 1.2055507, -0.7108005, 0.7018117
8: -0.1337706, 0.1623348, -0.1428718, 0.1728422, -0.2924221, 0.2908070
9: -0.1239739, 0.1488539, -0.1318080, 0.1597919, -0.2837659, 0.2806619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4694741
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4694741
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0813998, 0.0839573, -0.0956413, 0.0914879, -0.1728877, 0.1795986
1: -0.1088750, 0.0794717, -0.1182310, 0.0969484, -0.2058234, 0.1977026
2: -0.1162073, 0.1812061, -0.0994119, 0.1744765, -0.2906838, 0.2806180
3: -0.1134678, 0.2562633, -0.1020517, 0.2823921, -0.3849582, 0.3480303
4: -0.0890317, 0.0973113, -0.0965980, 0.1104174, -0.1994491, 0.1939093
5: -0.0766463, 0.1083534, -0.0913993, 0.1241543, -0.2008006, 0.1997527
6: -0.1088061, 0.1160462, -0.1103211, 0.1246136, -0.2334197, 0.2263672
7: 0.5428693, 1.2331438, 0.5151707, 1.2034724, -0.6606030, 0.7179731
8: -0.1575721, 0.1525174, -0.1403224, 0.1593117, -0.3039250, 0.2803233
9: -0.1107504, 0.1334492, -0.1208594, 0.1446966, -0.2554470, 0.2543086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4694741
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4694741
time: 1.68 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0968850, 0.0913247, -0.1044018, 0.0986944, -0.1955793, 0.1957265
1: -0.1189266, 0.1001908, -0.1258007, 0.1103986, -0.2293252, 0.2259915
2: -0.0922837, 0.1731038, -0.1054231, 0.1855578, -0.2778414, 0.2785269
3: -0.0974172, 0.2919357, -0.1068486, 0.2964579, -0.3828604, 0.3876593
4: -0.0965664, 0.1116928, -0.1068019, 0.1184680, -0.2150343, 0.2184948
5: -0.0932511, 0.1252685, -0.1017456, 0.1352547, -0.2285058, 0.2270140
6: -0.1105044, 0.1237817, -0.1201434, 0.1344759, -0.2449802, 0.2439251
7: 0.5035098, 1.1903324, 0.4993515, 1.2123797, -0.7088699, 0.6909809
8: -0.1329259, 0.1591198, -0.1467970, 0.1704036, -0.2892405, 0.2917162
9: -0.1213179, 0.1446903, -0.1297734, 0.1574509, -0.2787688, 0.2744637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
time: 1.79 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0794310, 0.0824907, -0.0940438, 0.0908837, -0.1703147, 0.1765344
1: -0.1069470, 0.0784923, -0.1171055, 0.0932345, -0.2001815, 0.1955978
2: -0.1158648, 0.1807053, -0.1032063, 0.1743155, -0.2901803, 0.2839116
3: -0.1129549, 0.2518110, -0.1047487, 0.2752568, -0.3773973, 0.3464074
4: -0.0874353, 0.0959499, -0.0957913, 0.1085765, -0.1960118, 0.1917412
5: -0.0749969, 0.1058259, -0.0890548, 0.1224624, -0.1974593, 0.1948807
6: -0.1083803, 0.1150033, -0.1093351, 0.1239130, -0.2322932, 0.2243384
7: 0.5488831, 1.2325625, 0.5239916, 1.2103490, -0.6614659, 0.7085709
8: -0.1569282, 0.1509073, -0.1443663, 0.1584408, -0.3025589, 0.2829664
9: -0.1090921, 0.1312530, -0.1195226, 0.1433951, -0.2524872, 0.2507755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4697319
time: 6.13 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4697319
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1039269, 0.0979006, -0.1073780, 0.1013941, -0.2053210, 0.2052786
1: -0.1258298, 0.1110555, -0.1286448, 0.1156494, -0.2414792, 0.2397004
2: -0.1039909, 0.1847036, -0.1069330, 0.1896675, -0.2936584, 0.2916366
3: -0.1056452, 0.3025747, -0.1078874, 0.3054771, -0.3988491, 0.3994054
4: -0.1059136, 0.1186842, -0.1104511, 0.1214890, -0.2274026, 0.2291353
5: -0.1014436, 0.1348257, -0.1055474, 0.1391593, -0.2406029, 0.2403731
6: -0.1199777, 0.1341332, -0.1240667, 0.1378781, -0.2578558, 0.2581999
7: 0.4910190, 1.2104201, 0.4882182, 1.2150204, -0.7240014, 0.7222019
8: -0.1453359, 0.1695338, -0.1484491, 0.1745448, -0.3035214, 0.3042773
9: -0.1299080, 0.1564579, -0.1334730, 0.1615884, -0.2914964, 0.2899309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4777544
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4777546
time: 1.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0849535, 0.0874003, -0.0961106, 0.0924627, -0.1774162, 0.1835109
1: -0.1123080, 0.0822461, -0.1191783, 0.0980462, -0.2103542, 0.2014244
2: -0.1277222, 0.1921514, -0.1046732, 0.1768934, -0.3046156, 0.2968246
3: -0.1214749, 0.2575781, -0.1057030, 0.2828110, -0.3922688, 0.3531427
4: -0.0930117, 0.1007401, -0.0978376, 0.1113993, -0.2044110, 0.1985776
5: -0.0797170, 0.1129163, -0.0920933, 0.1250767, -0.2047937, 0.2050096
6: -0.1167506, 0.1251797, -0.1118133, 0.1270430, -0.2437936, 0.2369930
7: 0.5415164, 1.2528732, 0.5143085, 1.2129190, -0.6714026, 0.7385647
8: -0.1696321, 0.1569364, -0.1459411, 0.1607975, -0.3154341, 0.2910525
9: -0.1150942, 0.1392309, -0.1224346, 0.1464104, -0.2615046, 0.2616656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4726488
time: 1.76 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4726490
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1006980, 0.0949942, -0.1050538, 0.0997538, -0.2004518, 0.2000480
1: -0.1230041, 0.1058907, -0.1268203, 0.1114963, -0.2345004, 0.2327110
2: -0.1033009, 0.1806534, -0.1106415, 0.1880075, -0.2913084, 0.2912949
3: -0.1049303, 0.2949116, -0.1104606, 0.2969918, -0.3897489, 0.3943983
4: -0.1020112, 0.1157478, -0.1081959, 0.1194590, -0.2214702, 0.2239437
5: -0.0974108, 0.1307188, -0.1025436, 0.1362774, -0.2336882, 0.2332624
6: -0.1160945, 0.1309932, -0.1217680, 0.1368494, -0.2529440, 0.2527611
7: 0.5000437, 1.2097397, 0.4983739, 1.2217712, -0.7217275, 0.7113658
8: -0.1445079, 0.1652364, -0.1523649, 0.1721111, -0.3004306, 0.3039629
9: -0.1264198, 0.1521213, -0.1313991, 0.1592259, -0.2856457, 0.2835204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4781516
time: 2.56 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4781554
time: 2.02 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0827587, 0.0857941, -0.0945394, 0.0918364, -0.1745951, 0.1803335
1: -0.1102499, 0.0809977, -0.1180295, 0.0943400, -0.2045900, 0.1990272
2: -0.1273754, 0.1916493, -0.1084133, 0.1767226, -0.3040979, 0.3000627
3: -0.1209681, 0.2531343, -0.1084106, 0.2756950, -0.3847640, 0.3514847
4: -0.0912667, 0.0991545, -0.0970165, 0.1095424, -0.2008091, 0.1961710
5: -0.0778483, 0.1101742, -0.0897558, 0.1233775, -0.2012258, 0.1999300
6: -0.1162944, 0.1243602, -0.1107851, 0.1262931, -0.2425875, 0.2351453
7: 0.5474752, 1.2522730, 0.5231370, 1.2197213, -0.6722462, 0.7291359
8: -0.1689903, 0.1551694, -0.1499723, 0.1599083, -0.3140911, 0.2934316
9: -0.1138178, 0.1368314, -0.1210619, 0.1451492, -0.2589670, 0.2578933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728179
time: 1.39 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728350
time: 4.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.94 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4694741
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4694741
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4694741
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4694741
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4697319
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4698497, upper bound: 0.4697319
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4777544
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4777546
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4726488
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4726490
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4781516
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4722419, upper bound: 0.4781554
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728179
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.94
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728350

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0999883, 0.0929706, -0.0966908, 0.0922397, -0.1922280, 0.1896614
1: -0.1216323, 0.1051631, -0.1191430, 0.0989152, -0.2205475, 0.2243061
2: -0.0929622, 0.1760693, -0.0997415, 0.1755807, -0.2685429, 0.2758108
3: -0.0981591, 0.2993918, -0.1023841, 0.2847304, -0.3717827, 0.3905467
4: -0.0995975, 0.1145196, -0.0975880, 0.1115815, -0.2111790, 0.2121076
5: -0.0965260, 0.1292124, -0.0928040, 0.1254127, -0.2219387, 0.2220164
6: -0.1136280, 0.1267890, -0.1112906, 0.1259372, -0.2395652, 0.2380795
7: 0.4947502, 1.1910176, 0.5123284, 1.2038170, -0.7090669, 0.6786892
8: -0.1337706, 0.1623348, -0.1407144, 0.1603930, -0.2799348, 0.2886171
9: -0.1239739, 0.1488539, -0.1219512, 0.1462149, -0.2701889, 0.2708051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3978770, upper bound: 0.4038258
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4718074
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0999883, 0.0929706, -0.0790813, 0.0829898, -0.1829781, 0.1720519
1: -0.1216323, 0.1051631, -0.1066912, 0.0790559, -0.2006882, 0.2118543
2: -0.0929622, 0.1760693, -0.1241397, 0.1884066, -0.2813688, 0.3002089
3: -0.0981591, 0.2993918, -0.1184856, 0.2461607, -0.3333484, 0.4068323
4: -0.0995975, 0.1145196, -0.0880874, 0.0965359, -0.1961334, 0.2026070
5: -0.0965260, 0.1292124, -0.0748360, 0.1056261, -0.2021521, 0.2040484
6: -0.1136280, 0.1267890, -0.1138027, 0.1211870, -0.2348150, 0.2405916
7: 0.4947502, 1.1910176, 0.5567919, 1.2464657, -0.7517155, 0.6342257
8: -0.1337706, 0.1623348, -0.1653083, 0.1517164, -0.2717162, 0.3145214
9: -0.1239739, 0.1488539, -0.1114469, 0.1325051, -0.2564790, 0.2603008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3978770, upper bound: 0.4038258
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4718074
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0813998, 0.0839573, -0.0919658, 0.0879711, -0.1693709, 0.1759231
1: -0.1088750, 0.0794717, -0.1146619, 0.0907363, -0.1996113, 0.1941336
2: -0.1162073, 0.1812061, -0.0908873, 0.1680875, -0.2842948, 0.2720934
3: -0.1134678, 0.2562633, -0.0959731, 0.2794071, -0.3818786, 0.3419231
4: -0.0890317, 0.0973113, -0.0921613, 0.1059376, -0.1949693, 0.1894725
5: -0.0766463, 0.1083534, -0.0868437, 0.1192777, -0.1959240, 0.1951971
6: -0.1088061, 0.1160462, -0.1057830, 0.1176760, -0.2264821, 0.2218292
7: 0.5428693, 1.2331438, 0.5184907, 1.1888580, -0.6459887, 0.7146531
8: -0.1575721, 0.1525174, -0.1312434, 0.1544773, -0.2987251, 0.2710342
9: -0.1107504, 0.1334492, -0.1158958, 0.1381802, -0.2489306, 0.2493449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4481566, upper bound: 0.4372778
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4358216
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0813998, 0.0839573, -0.0958161, 0.0919450, -0.1733448, 0.1797734
1: -0.1088750, 0.0794717, -0.1186418, 0.0973871, -0.2062621, 0.1981134
2: -0.1162073, 0.1812061, -0.1020170, 0.1756287, -0.2918360, 0.2832231
3: -0.1134678, 0.2562633, -0.1036745, 0.2824283, -0.3849949, 0.3497382
4: -0.0890317, 0.0973113, -0.0971502, 0.1108445, -0.1998761, 0.1944614
5: -0.0766463, 0.1083534, -0.0916632, 0.1245512, -0.2011975, 0.2000166
6: -0.1088061, 0.1160462, -0.1110136, 0.1257623, -0.2345684, 0.2270598
7: 0.5428693, 1.2331438, 0.5149540, 1.2083722, -0.6655029, 0.7181898
8: -0.1575721, 0.1525174, -0.1430107, 0.1600209, -0.3046523, 0.2832653
9: -0.1107504, 0.1334492, -0.1216111, 0.1454768, -0.2562272, 0.2550602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4481566, upper bound: 0.4372778
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4358216
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0968850, 0.0913247, -0.0948213, 0.0914474, -0.1883323, 0.1861461
1: -0.1189266, 0.1001908, -0.1177913, 0.0947484, -0.2136750, 0.2179821
2: -0.0922837, 0.1731038, -0.1034684, 0.1751383, -0.2674220, 0.2765721
3: -0.0974172, 0.2919357, -0.1050051, 0.2771213, -0.3635240, 0.3857809
4: -0.0965664, 0.1116928, -0.0965326, 0.1094680, -0.2060343, 0.2082254
5: -0.0932511, 0.1252685, -0.0901166, 0.1233898, -0.2166409, 0.2153850
6: -0.1105044, 0.1237817, -0.1100833, 0.1249103, -0.2354147, 0.2338651
7: 0.5035098, 1.1903324, 0.5216764, 1.2106252, -0.7071154, 0.6686560
8: -0.1329259, 0.1591198, -0.1446762, 0.1592509, -0.2780517, 0.2895865
9: -0.1213179, 0.1446903, -0.1203588, 0.1445103, -0.2658282, 0.2650490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3950987, upper bound: 0.4019828
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4725613
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0968850, 0.0913247, -0.0785769, 0.0825573, -0.1794423, 0.1699016
1: -0.1189266, 0.1001908, -0.1061112, 0.0789149, -0.1978415, 0.2063021
2: -0.0922837, 0.1731038, -0.1290746, 0.1930449, -0.2853286, 0.3021784
3: -0.0974172, 0.2919357, -0.1219872, 0.2421024, -0.3286428, 0.4030218
4: -0.0965664, 0.1116928, -0.0880814, 0.0962213, -0.1927876, 0.1997742
5: -0.0932511, 0.1252685, -0.0740625, 0.1052055, -0.1984566, 0.1993309
6: -0.1105044, 0.1237817, -0.1170231, 0.1248829, -0.2353872, 0.2408049
7: 0.5035098, 1.1903324, 0.5631775, 1.2543977, -0.7508880, 0.6271549
8: -0.1329259, 0.1591198, -0.1702888, 0.1516773, -0.2710779, 0.3166752
9: -0.1213179, 0.1446903, -0.1139579, 0.1323656, -0.2536835, 0.2586482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3950987, upper bound: 0.4019828
time: 4.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4725613
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0794310, 0.0824907, -0.0903793, 0.0874235, -0.1668545, 0.1728700
1: -0.1069470, 0.0784923, -0.1135943, 0.0880056, -0.1949525, 0.1920866
2: -0.1158648, 0.1807053, -0.0947748, 0.1680224, -0.2838872, 0.2754801
3: -0.1129549, 0.2518110, -0.0986815, 0.2723750, -0.3744308, 0.3403109
4: -0.0874353, 0.0959499, -0.0917293, 0.1041687, -0.1916040, 0.1876792
5: -0.0749969, 0.1058259, -0.0849761, 0.1176012, -0.1925980, 0.1908021
6: -0.1083803, 0.1150033, -0.1050782, 0.1170835, -0.2254638, 0.2200815
7: 0.5488831, 1.2325625, 0.5268908, 1.1957979, -0.6469148, 0.7056718
8: -0.1569282, 0.1509073, -0.1352973, 0.1541424, -0.2979603, 0.2736867
9: -0.1090921, 0.1312530, -0.1146483, 0.1374793, -0.2465714, 0.2459013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4472543, upper bound: 0.4359672
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4348568
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0794310, 0.0824907, -0.0942120, 0.0913096, -0.1707406, 0.1767027
1: -0.1069470, 0.0784923, -0.1174844, 0.0936532, -0.2006002, 0.1959766
2: -0.1158648, 0.1807053, -0.1056845, 0.1754242, -0.2912890, 0.2863898
3: -0.1129549, 0.2518110, -0.1063218, 0.2752956, -0.3774369, 0.3480315
4: -0.0874353, 0.0959499, -0.0963118, 0.1089739, -0.1964092, 0.1922617
5: -0.0749969, 0.1058259, -0.0893047, 0.1228357, -0.1978326, 0.1951306
6: -0.1083803, 0.1150033, -0.1099711, 0.1249934, -0.2333737, 0.2249744
7: 0.5488831, 1.2325625, 0.5237939, 1.2150544, -0.6661713, 0.7087686
8: -0.1569282, 0.1509073, -0.1469461, 0.1591021, -0.3032407, 0.2856616
9: -0.1090921, 0.1312530, -0.1202286, 0.1441636, -0.2532558, 0.2514816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4472543, upper bound: 0.4359716
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4348568
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1039269, 0.0979006, -0.1025316, 0.0954127, -0.1993396, 0.2004322
1: -0.1258298, 0.1110555, -0.1234824, 0.1083179, -0.2341477, 0.2345379
2: -0.1039909, 0.1847036, -0.0931705, 0.1791269, -0.2831178, 0.2778741
3: -0.1056452, 0.3025747, -0.0983380, 0.3011208, -0.3956665, 0.3898019
4: -0.1059136, 0.1186842, -0.1027178, 0.1163296, -0.2222432, 0.2214020
5: -0.1014436, 0.1348257, -0.0995113, 0.1323421, -0.2337857, 0.2343370
6: -0.1199777, 0.1341332, -0.1162374, 0.1288685, -0.2488462, 0.2503707
7: 0.4910190, 1.2104201, 0.4935061, 1.1910164, -0.6999974, 0.7169140
8: -0.1453359, 0.1695338, -0.1339180, 0.1657467, -0.2969326, 0.2893060
9: -0.1299080, 0.1564579, -0.1261601, 0.1523517, -0.2822597, 0.2826180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3987449, upper bound: 0.4136790
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1039269, 0.0979006, -0.1069579, 0.1008030, -0.2047299, 0.2048585
1: -0.1258298, 0.1110555, -0.1280418, 0.1149894, -0.2408192, 0.2390973
2: -0.1039909, 0.1847036, -0.1042567, 0.1883373, -0.2923282, 0.2889603
3: -0.1056452, 0.3025747, -0.1058743, 0.3050239, -0.3983936, 0.3962065
4: -0.1059136, 0.1186842, -0.1096246, 0.1209224, -0.2268360, 0.2283088
5: -0.1014436, 0.1348257, -0.1050485, 0.1385587, -0.2400023, 0.2398742
6: -0.1199777, 0.1341332, -0.1231418, 0.1366101, -0.2565878, 0.2572750
7: 0.4910190, 1.2104201, 0.4889502, 1.2104372, -0.7194182, 0.7213405
8: -0.1453359, 0.1695338, -0.1455300, 0.1735884, -0.3025331, 0.2986974
9: -0.1299080, 0.1564579, -0.1325868, 0.1606001, -0.2905082, 0.2890447

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3987449, upper bound: 0.4197089
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0849535, 0.0874003, -0.0919658, 0.0879711, -0.1729246, 0.1793661
1: -0.1123080, 0.0822461, -0.1146619, 0.0907363, -0.2030443, 0.1969080
2: -0.1277222, 0.1921514, -0.0908873, 0.1680875, -0.2958097, 0.2830387
3: -0.1214749, 0.2575781, -0.0959731, 0.2794071, -0.3899583, 0.3433636
4: -0.0930117, 0.1007401, -0.0921613, 0.1059376, -0.1989493, 0.1929013
5: -0.0797170, 0.1129163, -0.0868437, 0.1192777, -0.1989947, 0.1997600
6: -0.1167506, 0.1251797, -0.1057830, 0.1176760, -0.2344266, 0.2309627
7: 0.5415164, 1.2528732, 0.5184907, 1.1888580, -0.6473416, 0.7343825
8: -0.1696321, 0.1569364, -0.1312434, 0.1544773, -0.3111530, 0.2759106
9: -0.1150942, 0.1392309, -0.1158958, 0.1381802, -0.2532743, 0.2551267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4480923, upper bound: 0.4387902
time: 2.31 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4375069
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0849535, 0.0874003, -0.0958161, 0.0919450, -0.1768985, 0.1832164
1: -0.1123080, 0.0822461, -0.1186418, 0.0973871, -0.2096951, 0.2008879
2: -0.1277222, 0.1921514, -0.1020170, 0.1756287, -0.3033508, 0.2941684
3: -0.1214749, 0.2575781, -0.1036745, 0.2824283, -0.3918820, 0.3499256
4: -0.0930117, 0.1007401, -0.0971502, 0.1108445, -0.2038562, 0.1978903
5: -0.0797170, 0.1129163, -0.0916632, 0.1245512, -0.2042682, 0.2045795
6: -0.1167506, 0.1251797, -0.1110136, 0.1257623, -0.2425129, 0.2361933
7: 0.5415164, 1.2528732, 0.5149540, 1.2083722, -0.6668558, 0.7379192
8: -0.1696321, 0.1569364, -0.1430107, 0.1600209, -0.3146236, 0.2854687
9: -0.1150942, 0.1392309, -0.1216111, 0.1454768, -0.2605710, 0.2608420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4480923, upper bound: 0.4390680
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4378241
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1006980, 0.0949942, -0.1001837, 0.0938807, -0.1945787, 0.1951779
1: -0.1230041, 0.1058907, -0.1217969, 0.1044137, -0.2274178, 0.2276876
2: -0.1033009, 0.1806534, -0.0969808, 0.1777420, -0.2810429, 0.2776342
3: -0.1049303, 0.2949116, -0.1008942, 0.2929802, -0.3868989, 0.3847696
4: -0.1020112, 0.1157478, -0.1005926, 0.1144728, -0.2164840, 0.2163404
5: -0.0974108, 0.1307188, -0.0966088, 0.1295979, -0.2270087, 0.2273276
6: -0.1160945, 0.1309932, -0.1141504, 0.1280757, -0.2441702, 0.2451436
7: 0.5000437, 1.2097397, 0.5031124, 1.1978893, -0.6978456, 0.7066272
8: -0.1445079, 0.1652364, -0.1378702, 0.1634037, -0.2939513, 0.2890868
9: -0.1264198, 0.1521213, -0.1243135, 0.1501663, -0.2765861, 0.2764349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3962398, upper bound: 0.4120043
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
time: 2.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1006980, 0.0949942, -0.1046151, 0.0991437, -0.1998418, 0.1996093
1: -0.1230041, 0.1058907, -0.1262071, 0.1108189, -0.2338230, 0.2320977
2: -0.1033009, 0.1806534, -0.1078963, 0.1866663, -0.2899672, 0.2885497
3: -0.1049303, 0.2949116, -0.1083936, 0.2965139, -0.3892672, 0.3911547
4: -0.1020112, 0.1157478, -0.1073618, 0.1188785, -0.2208897, 0.2231095
5: -0.0974108, 0.1307188, -0.1020240, 0.1356526, -0.2330634, 0.2327428
6: -0.1160945, 0.1309932, -0.1208249, 0.1355493, -0.2516438, 0.2518181
7: 0.5000437, 1.2097397, 0.4991210, 1.2170916, -0.7170478, 0.7106187
8: -0.1445079, 0.1652364, -0.1493586, 0.1711274, -0.2994151, 0.2984524
9: -0.1264198, 0.1521213, -0.1305054, 0.1582098, -0.2846296, 0.2826267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3962398, upper bound: 0.4192048
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0827587, 0.0857941, -0.0903793, 0.0874235, -0.1701822, 0.1761734
1: -0.1102499, 0.0809977, -0.1135943, 0.0880056, -0.1982555, 0.1945920
2: -0.1273754, 0.1916493, -0.0947748, 0.1680224, -0.2953978, 0.2864241
3: -0.1209681, 0.2531343, -0.0986815, 0.2723750, -0.3825160, 0.3416952
4: -0.0912667, 0.0991545, -0.0917293, 0.1041687, -0.1954353, 0.1908838
5: -0.0778483, 0.1101742, -0.0849761, 0.1176012, -0.1954495, 0.1951503
6: -0.1162944, 0.1243602, -0.1050782, 0.1170835, -0.2333779, 0.2294384
7: 0.5474752, 1.2522730, 0.5268908, 1.1957979, -0.6483228, 0.7253822
8: -0.1689903, 0.1551694, -0.1352973, 0.1541424, -0.3103898, 0.2783883
9: -0.1138178, 0.1368314, -0.1146483, 0.1374793, -0.2512971, 0.2514797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4472118, upper bound: 0.4374831
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4363794
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0827587, 0.0857941, -0.0942120, 0.0913096, -0.1740683, 0.1800061
1: -0.1102499, 0.0809977, -0.1174844, 0.0936532, -0.2039031, 0.1984821
2: -0.1273754, 0.1916493, -0.1056845, 0.1754242, -0.3027995, 0.2973338
3: -0.1209681, 0.2531343, -0.1063218, 0.2752956, -0.3843615, 0.3482476
4: -0.0912667, 0.0991545, -0.0963118, 0.1089739, -0.2002405, 0.1954663
5: -0.0778483, 0.1101742, -0.0893047, 0.1228357, -0.2006840, 0.1994789
6: -0.1162944, 0.1243602, -0.1099711, 0.1249934, -0.2412878, 0.2343312
7: 0.5474752, 1.2522730, 0.5237939, 1.2150544, -0.6675792, 0.7284790
8: -0.1689903, 0.1551694, -0.1469461, 0.1591021, -0.3132522, 0.2878841
9: -0.1138178, 0.1368314, -0.1202286, 0.1441636, -0.2579814, 0.2570600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4472118, upper bound: 0.4377823
time: 6.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4367529
time: 3.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.78 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3978770, upper bound: 0.4038258
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4718074
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3978770, upper bound: 0.4038258
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4718074
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4481566, upper bound: 0.4372778
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4358216
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4481566, upper bound: 0.4372778
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4358216
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3950987, upper bound: 0.4019828
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4725613
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3950987, upper bound: 0.4019828
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4731118, upper bound: 0.4725613
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4472543, upper bound: 0.4359672
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4348568
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4472543, upper bound: 0.4359716
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4348568
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3987449, upper bound: 0.4136790
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3987449, upper bound: 0.4197089
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4480923, upper bound: 0.4387902
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4375069
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4480923, upper bound: 0.4390680
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4351033, upper bound: 0.4378241
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3962398, upper bound: 0.4120043
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.3962398, upper bound: 0.4192048
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4472118, upper bound: 0.4374831
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4363794
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4472118, upper bound: 0.4377823
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.78
Output dim: 7, lower bound: -0.4348568, upper bound: 0.4367529

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0945299, 0.0896996, -0.0962243, 0.0919084, -0.1864383, 0.1859239
1: -0.1167871, 0.0954802, -0.1187221, 0.0979567, -0.2147437, 0.2142023
2: -0.0910936, 0.1705679, -0.0995516, 0.1750794, -0.2661730, 0.2701194
3: -0.0960815, 0.2860352, -0.1021765, 0.2835062, -0.3684660, 0.3767208
4: -0.0942421, 0.1089562, -0.0971392, 0.1110194, -0.2052615, 0.2060954
5: -0.0901165, 0.1223517, -0.0921577, 0.1248430, -0.2149595, 0.2145094
6: -0.1080817, 0.1207348, -0.1108168, 0.1253286, -0.2334103, 0.2315516
7: 0.5106101, 1.1891675, 0.5138314, 1.2036275, -0.6856990, 0.6753361
8: -0.1314576, 0.1567063, -0.1404800, 0.1599167, -0.2770545, 0.2820111
9: -0.1187168, 0.1413438, -0.1214172, 0.1455514, -0.2642682, 0.2627610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4440647, upper bound: 0.4569693
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4432776, upper bound: 0.4413858
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0945299, 0.0896996, -0.0787274, 0.0826630, -0.1771929, 0.1684270
1: -0.1167871, 0.0954802, -0.1063567, 0.0788356, -0.1956227, 0.2018369
2: -0.0910936, 0.1705679, -0.1240100, 0.1882341, -0.2793277, 0.2945779
3: -0.0960815, 0.2860352, -0.1183095, 0.2454806, -0.3305774, 0.3929082
4: -0.0942421, 0.1089562, -0.0877884, 0.0962273, -0.1904694, 0.1967447
5: -0.0901165, 0.1223517, -0.0744729, 0.1052935, -0.1954100, 0.1968246
6: -0.1080817, 0.1207348, -0.1136641, 0.1209735, -0.2290552, 0.2343989
7: 0.5106101, 1.1891675, 0.5578717, 1.2462914, -0.7286588, 0.6312958
8: -0.1314576, 0.1567063, -0.1650997, 0.1513642, -0.2689641, 0.3077185
9: -0.1187168, 0.1413438, -0.1113057, 0.1320125, -0.2507293, 0.2526495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4718074
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4718074
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0922080, 0.0880849, -0.0943609, 0.0911206, -0.1833286, 0.1824458
1: -0.1147620, 0.0910671, -0.1173746, 0.0937976, -0.2085596, 0.2084417
2: -0.0904635, 0.1681751, -0.1032834, 0.1746530, -0.2651166, 0.2714586
3: -0.0953868, 0.2799105, -0.1048074, 0.2759159, -0.3602825, 0.3733140
4: -0.0922176, 0.1061745, -0.0960898, 0.1089106, -0.2011282, 0.2022642
5: -0.0870866, 0.1195433, -0.0894774, 0.1228269, -0.2099135, 0.2090206
6: -0.1058689, 0.1177872, -0.1096133, 0.1243086, -0.2301776, 0.2274006
7: 0.5180469, 1.1885061, 0.5231481, 1.2104418, -0.6856322, 0.6653580
8: -0.1306501, 0.1545769, -0.1444510, 0.1587808, -0.2752175, 0.2840400
9: -0.1161025, 0.1383375, -0.1198286, 0.1438669, -0.2599694, 0.2581661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4439411, upper bound: 0.4574385
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4431447, upper bound: 0.4412662
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0922080, 0.0880849, -0.0782198, 0.0822321, -0.1744401, 0.1663047
1: -0.1147620, 0.0910671, -0.1057885, 0.0786967, -0.1934587, 0.1968556
2: -0.0904635, 0.1681751, -0.1289457, 0.1928785, -0.2833421, 0.2971209
3: -0.0953868, 0.2799105, -0.1218182, 0.2414766, -0.3259764, 0.3904271
4: -0.0922176, 0.1061745, -0.0877877, 0.0959146, -0.1881322, 0.1939621
5: -0.0870866, 0.1195433, -0.0736959, 0.1049084, -0.1919950, 0.1932392
6: -0.1058689, 0.1177872, -0.1168847, 0.1246818, -0.2305507, 0.2346719
7: 0.5180469, 1.1885061, 0.5642164, 1.2542219, -0.7307686, 0.6242898
8: -0.1306501, 0.1545769, -0.1700871, 0.1513093, -0.2683525, 0.3109268
9: -0.1161025, 0.1383375, -0.1138210, 0.1318924, -0.2479949, 0.2521585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4725613
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4725613
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0980176, 0.0933236, -0.1019002, 0.0948438, -0.1928614, 0.1952239
1: -0.1205506, 0.1016270, -0.1229223, 0.1073123, -0.2278630, 0.2245493
2: -0.1021434, 0.1777585, -0.0929751, 0.1783258, -0.2804692, 0.2707336
3: -0.1036194, 0.2885600, -0.0981199, 0.2996220, -0.3921191, 0.3752758
4: -0.0991454, 0.1132611, -0.1019460, 0.1157530, -0.2148984, 0.2152071
5: -0.0944820, 0.1272565, -0.0987258, 0.1315345, -0.2260165, 0.2259823
6: -0.1131543, 0.1282237, -0.1154703, 0.1282414, -0.2413958, 0.2436939
7: 0.5075354, 1.2085791, 0.4952647, 1.1908231, -0.6832877, 0.7133144
8: -0.1430569, 0.1622415, -0.1336758, 0.1649056, -0.2937140, 0.2809568
9: -0.1239207, 0.1484493, -0.1254751, 0.1514987, -0.2754194, 0.2739244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
time: 1.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0980176, 0.0933236, -0.1063434, 0.1002468, -0.1982643, 0.1996670
1: -0.1205506, 0.1016270, -0.1274937, 0.1140160, -0.2345666, 0.2291207
2: -0.1021434, 0.1777585, -0.1040626, 0.1875516, -0.2896950, 0.2818211
3: -0.1036194, 0.2885600, -0.1056612, 0.3035831, -0.3948977, 0.3817151
4: -0.0991454, 0.1132611, -0.1088711, 0.1203621, -0.2195075, 0.2221322
5: -0.0944820, 0.1272565, -0.1042847, 0.1377717, -0.2322537, 0.2315412
6: -0.1131543, 0.1282237, -0.1223939, 0.1359932, -0.2491475, 0.2506176
7: 0.5075354, 1.2085791, 0.4906482, 1.2102445, -0.6927387, 0.7158009
8: -0.1430569, 0.1622415, -0.1452908, 0.1727666, -0.2993443, 0.2903908
9: -0.1239207, 0.1484493, -0.1319191, 0.1597664, -0.2836871, 0.2803684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=28, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4757986
time: 1.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4757986
time: 1.88 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0953981, 0.0916163, -0.0995520, 0.0933251, -0.1887233, 0.1911683
1: -0.1182526, 0.0965717, -0.1212368, 0.1034147, -0.2216673, 0.2178085
2: -0.1014742, 0.1750970, -0.0967879, 0.1769514, -0.2784256, 0.2718849
3: -0.1029333, 0.2818809, -0.1006839, 0.2915137, -0.3834125, 0.3712580
4: -0.0966621, 0.1103385, -0.0998286, 0.1138991, -0.2105612, 0.2101671
5: -0.0910942, 0.1240418, -0.0958328, 0.1287887, -0.2198829, 0.2198746
6: -0.1105486, 0.1251440, -0.1133920, 0.1274482, -0.2379968, 0.2385361
7: 0.5156112, 1.2079191, 0.5048338, 1.1976990, -0.6820877, 0.7030853
8: -0.1422549, 0.1595788, -0.1376341, 0.1625727, -0.2907753, 0.2823656
9: -0.1211655, 0.1448117, -0.1236390, 0.1493127, -0.2704782, 0.2684507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
time: 1.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
time: 2.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0953981, 0.0916163, -0.1039928, 0.0986007, -0.1939988, 0.1956091
1: -0.1182526, 0.0965717, -0.1256639, 0.1098554, -0.2281080, 0.2222356
2: -0.1014742, 0.1750970, -0.1077050, 0.1858966, -0.2873708, 0.2828020
3: -0.1029333, 0.2818809, -0.1081906, 0.2950955, -0.3858258, 0.3776753
4: -0.0966621, 0.1103385, -0.1066178, 0.1183246, -0.2149867, 0.2169563
5: -0.0910942, 0.1240418, -0.1012675, 0.1348797, -0.2259739, 0.2253093
6: -0.1105486, 0.1251440, -0.1200805, 0.1349440, -0.2454926, 0.2452245
7: 0.5156112, 1.2079191, 0.5007885, 1.2168962, -0.6927971, 0.7071307
8: -0.1422549, 0.1595788, -0.1491283, 0.1703056, -0.2962530, 0.2917615
9: -0.1211655, 0.1448117, -0.1298466, 0.1573922, -0.2785577, 0.2746583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=29, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4762455
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4762455
time: 1.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.05 seconds
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4440647, upper bound: 0.4569693
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4432776, upper bound: 0.4413858
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4718074
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4718074
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4439411, upper bound: 0.4574385
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4431447, upper bound: 0.4412662
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4725613
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4703448, upper bound: 0.4725613
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4757986
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4757986
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4757986
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4702913, upper bound: 0.4762455
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4762455
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 7, lower bound: -0.4704583, upper bound: 0.4762455

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0945299, 0.0896996, -0.0958862, 0.0913559, -0.1858858, 0.1855858
1: -0.1167871, 0.0954802, -0.1181414, 0.0972361, -0.2140232, 0.2136216
2: -0.0910936, 0.1705679, -0.0969293, 0.1738072, -0.2649008, 0.2674972
3: -0.0960815, 0.2860352, -0.1002414, 0.2831293, -0.3680876, 0.3747804
4: -0.0942421, 0.1089562, -0.0963984, 0.1104170, -0.2046591, 0.2053546
5: -0.0901165, 0.1223517, -0.0916730, 0.1242611, -0.2143776, 0.2140248
6: -0.1080817, 0.1207348, -0.1099672, 0.1239953, -0.2320770, 0.2307020
7: 0.5106101, 1.1891675, 0.5144495, 1.1991783, -0.6816639, 0.6747180
8: -0.1314576, 0.1567063, -0.1376486, 0.1590964, -0.2761941, 0.2791132
9: -0.1187168, 0.1413438, -0.1205558, 0.1445639, -0.2632807, 0.2618996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4432776, upper bound: 0.4413858
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4432776, upper bound: 0.4413858
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0945299, 0.0896996, -0.0761158, 0.0795623, -0.1740922, 0.1658155
1: -0.1167871, 0.0954802, -0.1038504, 0.0765682, -0.1933553, 0.1993306
2: -0.0910936, 0.1705679, -0.1151079, 0.1797361, -0.2708297, 0.2856758
3: -0.0960815, 0.2860352, -0.1119466, 0.2453812, -0.3304402, 0.3865403
4: -0.0942421, 0.1089562, -0.0847186, 0.0932137, -0.1874558, 0.1936748
5: -0.0901165, 0.1223517, -0.0717047, 0.1024508, -0.1925673, 0.1940564
6: -0.1080817, 0.1207348, -0.1075332, 0.1134778, -0.2215595, 0.2282680
7: 0.5106101, 1.1891675, 0.5587059, 1.2313881, -0.7142676, 0.6304616
8: -0.1314576, 0.1567063, -0.1556388, 0.1477753, -0.2650838, 0.2980534
9: -0.1187168, 0.1413438, -0.1065869, 0.1269252, -0.2456420, 0.2479307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4478654, upper bound: 0.4383911
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4378229
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0945299, 0.0896996, -0.0788487, 0.0830182, -0.1775481, 0.1685484
1: -0.1167871, 0.0954802, -0.1066481, 0.0791675, -0.1959546, 0.2021283
2: -0.0910936, 0.1705679, -0.1266037, 0.1906471, -0.2817407, 0.2971715
3: -0.0960815, 0.2860352, -0.1199942, 0.2454812, -0.3305782, 0.3946778
4: -0.0942421, 0.1089562, -0.0882129, 0.0965762, -0.1908183, 0.1971692
5: -0.0901165, 0.1223517, -0.0746496, 0.1055876, -0.1957041, 0.1970013
6: -0.1080817, 0.1207348, -0.1153729, 0.1229490, -0.2310307, 0.2361076
7: 0.5106101, 1.1891675, 0.5577480, 1.2509485, -0.7362764, 0.6314195
8: -0.1314576, 0.1567063, -0.1677420, 0.1519142, -0.2695504, 0.3105272
9: -0.1187168, 0.1413438, -0.1127585, 0.1327212, -0.2514380, 0.2541022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4478654, upper bound: 0.4383911
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4378229
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0922080, 0.0880849, -0.0940412, 0.0905985, -0.1828064, 0.1821261
1: -0.1147620, 0.0910671, -0.1168278, 0.0931071, -0.2078691, 0.2078949
2: -0.0904635, 0.1681751, -0.1007317, 0.1734028, -0.2638663, 0.2689068
3: -0.0953868, 0.2799105, -0.1028684, 0.2755496, -0.3599146, 0.3713765
4: -0.0922176, 0.1061745, -0.0953860, 0.1083403, -0.2005579, 0.2015604
5: -0.0870866, 0.1195433, -0.0890198, 0.1222762, -0.2093628, 0.2085631
6: -0.1058689, 0.1177872, -0.1088137, 0.1230371, -0.2289060, 0.2266009
7: 0.5180469, 1.1885061, 0.5237361, 1.2060909, -0.6817403, 0.6647701
8: -0.1306501, 0.1545769, -0.1416499, 0.1580071, -0.2743974, 0.2811725
9: -0.1161025, 0.1383375, -0.1190091, 0.1428967, -0.2589992, 0.2573466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4431447, upper bound: 0.4412662
time: 2.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4431447, upper bound: 0.4412662
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0922080, 0.0880849, -0.0758553, 0.0795418, -0.1717498, 0.1639402
1: -0.1147620, 0.0910671, -0.1035565, 0.0766824, -0.1914444, 0.1946236
2: -0.0904635, 0.1681751, -0.1202199, 0.1845890, -0.2750525, 0.2883951
3: -0.0953868, 0.2799105, -0.1156263, 0.2412043, -0.3257071, 0.3842253
4: -0.0922176, 0.1061745, -0.0850359, 0.0932920, -0.1855095, 0.1912104
5: -0.0870866, 0.1195433, -0.0712834, 0.1024657, -0.1895523, 0.1908267
6: -0.1058689, 0.1177872, -0.1109328, 0.1173709, -0.2232398, 0.2287200
7: 0.5180469, 1.1885061, 0.5651016, 1.2394700, -0.7165791, 0.6234045
8: -0.1306501, 0.1545769, -0.1608626, 0.1479594, -0.2646754, 0.3015589
9: -0.1161025, 0.1383375, -0.1086527, 0.1272821, -0.2433846, 0.2469902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4473879, upper bound: 0.4368800
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4364967
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0922080, 0.0880849, -0.0783610, 0.0825609, -0.1747689, 0.1664459
1: -0.1147620, 0.0910671, -0.1060604, 0.0790037, -0.1937657, 0.1971275
2: -0.0904635, 0.1681751, -0.1314175, 0.1951520, -0.2856155, 0.2995927
3: -0.0953868, 0.2799105, -0.1233880, 0.2414814, -0.3259807, 0.3920191
4: -0.0922176, 0.1061745, -0.0881881, 0.0962407, -0.1884583, 0.1943626
5: -0.0870866, 0.1195433, -0.0738686, 0.1051806, -0.1922672, 0.1934118
6: -0.1058689, 0.1177872, -0.1185002, 0.1265640, -0.2324330, 0.2362874
7: 0.5180469, 1.1885061, 0.5641072, 1.2585995, -0.7371166, 0.6243989
8: -0.1306501, 0.1545769, -0.1726033, 0.1518381, -0.2689196, 0.3134668
9: -0.1161025, 0.1383375, -0.1151759, 0.1326195, -0.2487220, 0.2535134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4473879, upper bound: 0.4368800
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4365001
time: 1.50 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0980176, 0.0933236, -0.0930971, 0.0887546, -0.1867722, 0.1864207
1: -0.1205506, 0.1016270, -0.1155468, 0.0922943, -0.2128449, 0.2171738
2: -0.1021434, 0.1777585, -0.0911013, 0.1691961, -0.2713395, 0.2688598
3: -0.1036194, 0.2885600, -0.0961842, 0.2812617, -0.3737528, 0.3733015
4: -0.0991454, 0.1132611, -0.0930036, 0.1071670, -0.2063124, 0.2062647
5: -0.0944820, 0.1272565, -0.0880727, 0.1206183, -0.2151003, 0.2153293
6: -0.1131543, 0.1282237, -0.1066625, 0.1189895, -0.2321438, 0.2348862
7: 0.5075354, 1.2085791, 0.5164529, 1.1890945, -0.6815591, 0.6921262
8: -0.1430569, 0.1622415, -0.1315080, 0.1553129, -0.2840344, 0.2787729
9: -0.1239207, 0.1484493, -0.1170031, 0.1394462, -0.2633669, 0.2654524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4478637, upper bound: 0.4404642
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4399143
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0980176, 0.0933236, -0.0761158, 0.0795623, -0.1775799, 0.1694394
1: -0.1205506, 0.1016270, -0.1038504, 0.0765682, -0.1971188, 0.2054774
2: -0.1021434, 0.1777585, -0.1151079, 0.1797361, -0.2818795, 0.2928664
3: -0.1036194, 0.2885600, -0.1119466, 0.2453812, -0.3380867, 0.3891454
4: -0.0991454, 0.1132611, -0.0847186, 0.0932137, -0.1923591, 0.1979796
5: -0.0944820, 0.1272565, -0.0717047, 0.1024508, -0.1969328, 0.1989612
6: -0.1131543, 0.1282237, -0.1075332, 0.1134778, -0.2266321, 0.2357569
7: 0.5075354, 1.2085791, 0.5587059, 1.2313881, -0.7238527, 0.6498731
8: -0.1430569, 0.1622415, -0.1556388, 0.1477753, -0.2771515, 0.3040256
9: -0.1239207, 0.1484493, -0.1065869, 0.1269252, -0.2508459, 0.2550362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=22, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4478637, upper bound: 0.4404642
time: 1.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4399143
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0980176, 0.0933236, -0.0964043, 0.0923696, -0.1903872, 0.1897280
1: -0.1205506, 0.1016270, -0.1191372, 0.0984067, -0.2189574, 0.2207642
2: -0.1021434, 0.1777585, -0.1021561, 0.1762315, -0.2783749, 0.2799146
3: -0.1036194, 0.2885600, -0.1037953, 0.2835480, -0.3748972, 0.3798031
4: -0.0991454, 0.1132611, -0.0976963, 0.1114530, -0.2105984, 0.2109574
5: -0.0944820, 0.1272565, -0.0924292, 0.1252460, -0.2197280, 0.2196857
6: -0.1131543, 0.1282237, -0.1115145, 0.1264840, -0.2396384, 0.2397382
7: 0.5075354, 1.2085791, 0.5136040, 1.2085259, -0.6900761, 0.6920959
8: -0.1430569, 0.1622415, -0.1431650, 0.1606317, -0.2871283, 0.2882426
9: -0.1239207, 0.1484493, -0.1221744, 0.1463398, -0.2702605, 0.2706238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4487093, upper bound: 0.4406043
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4428944, upper bound: 0.4400874
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0980176, 0.0933236, -0.0788487, 0.0830182, -0.1810357, 0.1721723
1: -0.1205506, 0.1016270, -0.1066481, 0.0791675, -0.1997181, 0.2082751
2: -0.1021434, 0.1777585, -0.1266037, 0.1906471, -0.2927905, 0.3043622
3: -0.1036194, 0.2885600, -0.1199942, 0.2454812, -0.3369997, 0.3961152
4: -0.0991454, 0.1132611, -0.0882129, 0.0965762, -0.1957216, 0.2014740
5: -0.0944820, 0.1272565, -0.0746496, 0.1055876, -0.2000696, 0.2019061
6: -0.1131543, 0.1282237, -0.1153729, 0.1229490, -0.2361033, 0.2435966
7: 0.5075354, 1.2085791, 0.5577480, 1.2509485, -0.7321801, 0.6508311
8: -0.1430569, 0.1622415, -0.1677420, 0.1519142, -0.2790404, 0.3140015
9: -0.1239207, 0.1484493, -0.1127585, 0.1327212, -0.2566419, 0.2612078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4487093, upper bound: 0.4406043
time: 1.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4428944, upper bound: 0.4400874
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0953981, 0.0916163, -0.0913716, 0.0881016, -0.1834997, 0.1829879
1: -0.1182526, 0.0965717, -0.1143781, 0.0893361, -0.2075887, 0.2109498
2: -0.1014742, 0.1750970, -0.0949522, 0.1689547, -0.2704288, 0.2700492
3: -0.1029333, 0.2818809, -0.0988476, 0.2741237, -0.3660016, 0.3693880
4: -0.0966621, 0.1103385, -0.0924354, 0.1052661, -0.2019283, 0.2027739
5: -0.0910942, 0.1240418, -0.0860273, 0.1187842, -0.2098784, 0.2100692
6: -0.1105486, 0.1251440, -0.1058512, 0.1182322, -0.2287808, 0.2309952
7: 0.5156112, 1.2079191, 0.5250270, 1.1959902, -0.6803790, 0.6828921
8: -0.1422549, 0.1595788, -0.1355142, 0.1548248, -0.2829148, 0.2802418
9: -0.1211655, 0.1448117, -0.1156420, 0.1385195, -0.2596850, 0.2604537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4473706, upper bound: 0.4388623
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4386334
time: 1.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0953981, 0.0916163, -0.0757490, 0.0795418, -0.1749399, 0.1673653
1: -0.1182526, 0.0965717, -0.1035341, 0.0766824, -0.1949351, 0.2001058
2: -0.1014742, 0.1750970, -0.1202021, 0.1845890, -0.2860631, 0.2952992
3: -0.1029333, 0.2818809, -0.1156263, 0.2412026, -0.3333519, 0.3863027
4: -0.0966621, 0.1103385, -0.0850042, 0.0932920, -0.1899541, 0.1953426
5: -0.0910942, 0.1240418, -0.0712487, 0.1024657, -0.1935599, 0.1952905
6: -0.1105486, 0.1251440, -0.1109118, 0.1173709, -0.2279195, 0.2360558
7: 0.5156112, 1.2079191, 0.5651016, 1.2394238, -0.7238125, 0.6428175
8: -0.1422549, 0.1595788, -0.1608626, 0.1478653, -0.2766477, 0.3069438
9: -0.1211655, 0.1448117, -0.1086527, 0.1272723, -0.2484378, 0.2534644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4473706, upper bound: 0.4388623
time: 2.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4386334
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0953981, 0.0916163, -0.0945307, 0.0915479, -0.1869460, 0.1861470
1: -0.1182526, 0.0965717, -0.1177548, 0.0942195, -0.2124721, 0.2143265
2: -0.1014742, 0.1750970, -0.1057613, 0.1757644, -0.2772385, 0.2808583
3: -0.1029333, 0.2818809, -0.1063783, 0.2759556, -0.3666794, 0.3758206
4: -0.0966621, 0.1103385, -0.0966120, 0.1093097, -0.2059718, 0.2069505
5: -0.0910942, 0.1240418, -0.0897294, 0.1232021, -0.2142963, 0.2137712
6: -0.1105486, 0.1251440, -0.1102509, 0.1253917, -0.2359403, 0.2353949
7: 0.5156112, 1.2079191, 0.5229481, 1.2151481, -0.6900586, 0.6849710
8: -0.1422549, 0.1595788, -0.1470300, 0.1594442, -0.2853071, 0.2896604
9: -0.1211655, 0.1448117, -0.1205365, 0.1446387, -0.2658042, 0.2653482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4483144, upper bound: 0.4390799
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4427581, upper bound: 0.4388257
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0953981, 0.0916163, -0.0783610, 0.0825609, -0.1779590, 0.1699773
1: -0.1182526, 0.0965717, -0.1060604, 0.0790037, -0.1972563, 0.2026321
2: -0.1014742, 0.1750970, -0.1314175, 0.1951520, -0.2966262, 0.3065146
3: -0.1029333, 0.2818809, -0.1233880, 0.2414814, -0.3324310, 0.3929774
4: -0.0966621, 0.1103385, -0.0881881, 0.0962407, -0.1929028, 0.1985266
5: -0.0910942, 0.1240418, -0.0738686, 0.1051806, -0.1962748, 0.1979104
6: -0.1105486, 0.1251440, -0.1185002, 0.1265640, -0.2371126, 0.2436442
7: 0.5156112, 1.2079191, 0.5641072, 1.2585995, -0.7340215, 0.6438119
8: -0.1422549, 0.1595788, -0.1726033, 0.1518381, -0.2784777, 0.3165087
9: -0.1211655, 0.1448117, -0.1151759, 0.1326195, -0.2537850, 0.2599876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4483144, upper bound: 0.4390799
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4427581, upper bound: 0.4388257
time: 1.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.25 seconds
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4432776, upper bound: 0.4413858
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4432776, upper bound: 0.4413858
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4478654, upper bound: 0.4383911
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4378229
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4478654, upper bound: 0.4383911
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4378229
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4431447, upper bound: 0.4412662
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4431447, upper bound: 0.4412662
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4473879, upper bound: 0.4368800
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4364967
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4473879, upper bound: 0.4368800
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4365001
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4478637, upper bound: 0.4404642
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4399143
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4478637, upper bound: 0.4404642
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4422314, upper bound: 0.4399143
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4487093, upper bound: 0.4406043
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4428944, upper bound: 0.4400874
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4487093, upper bound: 0.4406043
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4428944, upper bound: 0.4400874
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4473706, upper bound: 0.4388623
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4386334
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4473706, upper bound: 0.4388623
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4420188, upper bound: 0.4386334
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4483144, upper bound: 0.4390799
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4427581, upper bound: 0.4388257
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4483144, upper bound: 0.4390799
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.25
Output dim: 7, lower bound: -0.4427581, upper bound: 0.4388257

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.86 + 302.07 = 305.93 seconds
