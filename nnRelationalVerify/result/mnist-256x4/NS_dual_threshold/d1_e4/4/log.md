## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.45381564


## IAR start

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
execution time: IAR + RelationalAnalysis = 1.59 + 2.55 = 4.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.5042396, upper bound: 0.5042396

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5023537, upper bound: 0.4987494
time: 1.59 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.5023537, upper bound: 0.5023537
time: 1.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 7, lower bound: -0.5023537, upper bound: 0.4987494
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 7, lower bound: -0.5023537, upper bound: 0.5023537

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4977015, upper bound: 0.4945513
time: 1.60 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4944776
time: 2.04 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4977015, upper bound: 0.4978769
time: 1.51 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4978512
time: 1.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.83 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.83
Output dim: 7, lower bound: -0.4977015, upper bound: 0.4945513
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.83
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4944776
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.83
Output dim: 7, lower bound: -0.4977015, upper bound: 0.4978769
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.83
Output dim: 7, lower bound: -0.4978512, upper bound: 0.4978512

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.1025316, 0.0954127, -0.1140622, 0.1067414, -0.2092730, 0.2094748
1: -0.1234824, 0.1083179, -0.1340205, 0.1263413, -0.2498237, 0.2423383
2: -0.0931705, 0.1791269, -0.1033388, 0.1962902, -0.2894607, 0.2824657
3: -0.0983380, 0.3011208, -0.1060256, 0.3231406, -0.4103740, 0.3971621
4: -0.1027178, 0.1163296, -0.1179077, 0.1271436, -0.2298615, 0.2342373
5: -0.0995113, 0.1323421, -0.1137977, 0.1474018, -0.2469132, 0.2461397
6: -0.1162374, 0.1288685, -0.1312135, 0.1426290, -0.2588664, 0.2600820
7: 0.4935061, 1.1910164, 0.4678289, 1.2074243, -0.7139181, 0.7231876
8: -0.1339180, 0.1657467, -0.1448863, 0.1824541, -0.3022541, 0.2983426
9: -0.1261601, 0.1523517, -0.1398058, 0.1694430, -0.2956031, 0.2921575

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4749762, upper bound: 0.4744459
time: 1.43 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4726488, upper bound: 0.4697319
time: 1.47 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.1001837, 0.0938807, -0.1108260, 0.1038881, -0.2040718, 0.2047067
1: -0.1217969, 0.1044137, -0.1312292, 0.1211879, -0.2429848, 0.2356429
2: -0.0969808, 0.1777420, -0.1026273, 0.1922505, -0.2892312, 0.2803693
3: -0.1008942, 0.2929802, -0.1052667, 0.3154703, -0.4053324, 0.3883317
4: -0.1005926, 0.1144728, -0.1139675, 0.1242389, -0.2248315, 0.2284403
5: -0.0966088, 0.1295979, -0.1098093, 0.1433474, -0.2399562, 0.2394072
6: -0.1141504, 0.1280757, -0.1273665, 0.1394898, -0.2536401, 0.2554421
7: 0.5031124, 1.1978893, 0.4768579, 1.2066244, -0.7035120, 0.7210314
8: -0.1378702, 0.1634037, -0.1440144, 0.1781929, -0.3020654, 0.2952644
9: -0.1243135, 0.1501663, -0.1363135, 0.1651773, -0.2894908, 0.2864799

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4781516, upper bound: 0.4722419
time: 1.28 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
time: 1.50 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.1069579, 0.1008030, -0.1147216, 0.1078322, -0.2147901, 0.2155246
1: -0.1280418, 0.1149894, -0.1350921, 0.1274899, -0.2555317, 0.2500814
2: -0.1042567, 0.1883373, -0.1086248, 0.1988687, -0.3031254, 0.2969621
3: -0.1058743, 0.3050239, -0.1095912, 0.3236419, -0.4172436, 0.4047559
4: -0.1096246, 0.1209224, -0.1193759, 0.1281788, -0.2378035, 0.2402983
5: -0.1050485, 0.1385587, -0.1146138, 0.1484452, -0.2534937, 0.2531725
6: -0.1231418, 0.1366101, -0.1329160, 0.1451022, -0.2682439, 0.2695261
7: 0.4889502, 1.2104372, 0.4668229, 1.2169327, -0.7279825, 0.7436143
8: -0.1455300, 0.1735884, -0.1504474, 0.1842157, -0.3134827, 0.3123648
9: -0.1325868, 0.1606001, -0.1415166, 0.1712740, -0.3038608, 0.3021167

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4941241, upper bound: 0.4978769
time: 1.68 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4941241, upper bound: 0.4978769
time: 2.36 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.1046151, 0.0991437, -0.1114746, 0.1049708, -0.2095858, 0.2106183
1: -0.1262071, 0.1108189, -0.1322879, 0.1223214, -0.2485285, 0.2431068
2: -0.1078963, 0.1866663, -0.1079138, 0.1948106, -0.3027069, 0.2945801
3: -0.1083936, 0.2965139, -0.1088516, 0.3159684, -0.4121971, 0.3955602
4: -0.1073618, 0.1188785, -0.1154241, 0.1252629, -0.2326247, 0.2343026
5: -0.1020240, 0.1356526, -0.1106140, 0.1443815, -0.2464055, 0.2462666
6: -0.1208249, 0.1355493, -0.1290512, 0.1419450, -0.2627698, 0.2646005
7: 0.4991210, 1.2170916, 0.4758689, 1.2161398, -0.7170188, 0.7412227
8: -0.1493586, 0.1711274, -0.1495894, 0.1799367, -0.3132571, 0.3090771
9: -0.1305054, 0.1582098, -0.1380105, 0.1669945, -0.2974999, 0.2962203

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4944776, upper bound: 0.4978512
time: 1.72 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4944776, upper bound: 0.4978512
time: 1.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.09 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4749762, upper bound: 0.4744459
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4726488, upper bound: 0.4697319
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4781516, upper bound: 0.4722419
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4728179, upper bound: 0.4697319
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4941241, upper bound: 0.4978769
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4941241, upper bound: 0.4978769
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4944776, upper bound: 0.4978512
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.09
Output dim: 7, lower bound: -0.4944776, upper bound: 0.4978512

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0935850, 0.0890990, -0.1140622, 0.1067414, -0.2003264, 0.2031612
1: -0.1159833, 0.0932957, -0.1340205, 0.1263413, -0.2423246, 0.2273162
2: -0.0912960, 0.1697145, -0.1033388, 0.1962902, -0.2875862, 0.2730533
3: -0.0964000, 0.2825388, -0.1060256, 0.3231406, -0.4083996, 0.3785720
4: -0.0934708, 0.1077525, -0.1179077, 0.1271436, -0.2206144, 0.2256602
5: -0.0887473, 0.1212136, -0.1137977, 0.1474018, -0.2361491, 0.2350112
6: -0.1071536, 0.1196248, -0.1312135, 0.1426290, -0.2497826, 0.2508383
7: 0.5148906, 1.1892891, 0.4678289, 1.2074243, -0.6925337, 0.7214602
8: -0.1317495, 0.1558080, -0.1448863, 0.1824541, -0.3000562, 0.2883072
9: -0.1175579, 0.1401378, -0.1398058, 0.1694430, -0.2870010, 0.2799436

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4302200, upper bound: 0.4110871
time: 0.93 seconds

## Relational analysis of NS_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4724923, upper bound: 0.4718855
time: 2.34 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0764507, 0.0798936, -0.1025266, 0.0962936, -0.1727442, 0.1824202
1: -0.1041767, 0.0767915, -0.1243403, 0.1088258, -0.2130025, 0.2011318
2: -0.1152426, 0.1799014, -0.1010853, 0.1819609, -0.2972035, 0.2809867
3: -0.1121249, 0.2460560, -0.1037452, 0.2999514, -0.4011524, 0.3395794
4: -0.0850097, 0.0935270, -0.1038499, 0.1172050, -0.2022147, 0.1973768
5: -0.0720735, 0.1027499, -0.0996737, 0.1328741, -0.2049476, 0.2024236
6: -0.1076666, 0.1137021, -0.1178306, 0.1317947, -0.2394613, 0.2315327
7: 0.5576016, 1.2315543, 0.4943240, 1.2051967, -0.6475952, 0.7372302
8: -0.1558525, 0.1481317, -0.1423054, 0.1671399, -0.3101162, 0.2778684
9: -0.1068383, 0.1274213, -0.1278757, 0.1539827, -0.2608209, 0.2552969

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4245768, upper bound: 0.4070506
time: 1.09 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4701703, upper bound: 0.4672139
time: 1.04 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1001837, 0.0938807, -0.1005025, 0.0945081, -0.1946918, 0.1943832
1: -0.1217969, 0.1044137, -0.1225684, 0.1054424, -0.2272393, 0.2269820
2: -0.0969808, 0.1777420, -0.1006860, 0.1794602, -0.2764409, 0.2784279
3: -0.1008942, 0.2929802, -0.1033181, 0.2948547, -0.3847120, 0.3852003
4: -0.1005926, 0.1144728, -0.1014224, 0.1153024, -0.2158950, 0.2158952
5: -0.0966088, 0.1295979, -0.0971214, 0.1302919, -0.2269007, 0.2267193
6: -0.1141504, 0.1280757, -0.1153617, 0.1298403, -0.2439907, 0.2434374
7: 0.5031124, 1.1978893, 0.5002915, 1.2048177, -0.7017052, 0.6975979
8: -0.1378702, 0.1634037, -0.1418167, 0.1644832, -0.2883135, 0.2909994
9: -0.1243135, 0.1501663, -0.1256375, 0.1513056, -0.2756191, 0.2758039

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4413121, upper bound: 0.4498335
time: 1.64 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4410916, upper bound: 0.4446392
time: 1.39 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0903793, 0.0874235, -0.0823781, 0.0854310, -0.1758103, 0.1698016
1: -0.1135943, 0.0880056, -0.1098935, 0.0806609, -0.1942552, 0.1978991
2: -0.0947748, 0.1680224, -0.1247383, 0.1892326, -0.2840073, 0.2927607
3: -0.0986815, 0.2723750, -0.1192892, 0.2531271, -0.3416879, 0.3807851
4: -0.0917293, 0.1041687, -0.0907563, 0.0987995, -0.1905288, 0.1949250
5: -0.0849761, 0.1176012, -0.0775832, 0.1098682, -0.1948443, 0.1951843
6: -0.1050782, 0.1170835, -0.1145328, 0.1224016, -0.2274798, 0.2316163
7: 0.5268908, 1.1957979, 0.5476046, 1.2474949, -0.7206042, 0.6481933
8: -0.1352973, 0.1541424, -0.1663473, 0.1543822, -0.2775701, 0.3075972
9: -0.1146483, 0.1374793, -0.1126396, 0.1361240, -0.2507724, 0.2501189

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3956732, upper bound: 0.4109053
time: 1.11 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4703183, upper bound: 0.4672139
time: 1.63 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1069579, 0.1008030, -0.1098652, 0.1019280, -0.2088859, 0.2106682
1: -0.1280418, 0.1149894, -0.1299324, 0.1201803, -0.2482221, 0.2449217
2: -0.1042567, 0.1883373, -0.0948636, 0.1882578, -0.2925144, 0.2832009
3: -0.1058743, 0.3050239, -0.1001671, 0.3194932, -0.4142761, 0.3952934
4: -0.1096246, 0.1209224, -0.1115456, 0.1230549, -0.2326795, 0.2324681
5: -0.1050485, 0.1385587, -0.1086522, 0.1417175, -0.2467660, 0.2472109
6: -0.1231418, 0.1366101, -0.1251002, 0.1359971, -0.2591389, 0.2617102
7: 0.4889502, 1.2104372, 0.4719100, 1.1927882, -0.7038381, 0.7385272
8: -0.1455300, 0.1735884, -0.1360022, 0.1754411, -0.3069392, 0.2974959
9: -0.1325868, 0.1606001, -0.1341512, 0.1621232, -0.2947100, 0.2947513

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4737412, upper bound: 0.4751231
time: 1.43 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4694741, upper bound: 0.4728179
time: 1.34 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1069579, 0.1008030, -0.1142935, 0.1072338, -0.2141917, 0.2150964
1: -0.1280418, 0.1149894, -0.1344761, 0.1268130, -0.2548547, 0.2494655
2: -0.1042567, 0.1883373, -0.1059432, 0.1975113, -0.3017681, 0.2942805
3: -0.1058743, 0.3050239, -0.1075895, 0.3231943, -0.4167928, 0.4015989
4: -0.1096246, 0.1209224, -0.1185263, 0.1276021, -0.2372268, 0.2394488
5: -0.1050485, 0.1385587, -0.1141060, 0.1478372, -0.2528857, 0.2526647
6: -0.1231418, 0.1366101, -0.1319747, 0.1438089, -0.2669507, 0.2685847
7: 0.4889502, 1.2104372, 0.4675605, 1.2123502, -0.7234001, 0.7428766
8: -0.1455300, 0.1735884, -0.1475388, 0.1832459, -0.3124831, 0.3069562
9: -0.1325868, 0.1606001, -0.1406110, 0.1702733, -0.3028601, 0.3012111

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4737412, upper bound: 0.4751232
time: 1.90 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4694741, upper bound: 0.4728350
time: 1.29 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1046151, 0.0991437, -0.1068032, 0.0991785, -0.2037935, 0.2059469
1: -0.1262071, 0.1108189, -0.1272620, 0.1152947, -0.2415018, 0.2380809
2: -0.1078963, 0.1866663, -0.0941785, 0.1844349, -0.2923312, 0.2808449
3: -0.1083936, 0.2965139, -0.0994018, 0.3120946, -0.4094330, 0.3860735
4: -0.1073618, 0.1188785, -0.1078501, 0.1202805, -0.2276422, 0.2267286
5: -0.1020240, 0.1356526, -0.1048375, 0.1378190, -0.2398430, 0.2404902
6: -0.1208249, 0.1355493, -0.1214329, 0.1330324, -0.2538573, 0.2569822
7: 0.4991210, 1.2170916, 0.4806227, 1.1920898, -0.6929688, 0.7364689
8: -0.1493586, 0.1711274, -0.1351364, 0.1713741, -0.3067527, 0.2942068
9: -0.1305054, 0.1582098, -0.1308537, 0.1580205, -0.2885259, 0.2890635

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4744452, upper bound: 0.4751231
time: 1.51 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728179
time: 1.34 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1046151, 0.0991437, -0.1110510, 0.1043784, -0.2089934, 0.2101947
1: -0.1262071, 0.1108189, -0.1316796, 0.1216547, -0.2478618, 0.2424985
2: -0.1078963, 0.1866663, -0.1052363, 0.1934648, -0.3013610, 0.2919026
3: -0.1083936, 0.2965139, -0.1068461, 0.3155254, -0.4117502, 0.3924260
4: -0.1073618, 0.1188785, -0.1145821, 0.1246936, -0.2320554, 0.2334606
5: -0.1020240, 0.1356526, -0.1101135, 0.1437805, -0.2458045, 0.2457662
6: -0.1208249, 0.1355493, -0.1281200, 0.1406627, -0.2614876, 0.2636693
7: 0.4991210, 1.2170916, 0.4765947, 1.2115552, -0.7124343, 0.7404968
8: -0.1493586, 0.1711274, -0.1466791, 0.1789758, -0.3122662, 0.3037558
9: -0.1305054, 0.1582098, -0.1371137, 0.1660035, -0.2965088, 0.2953234

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4744452, upper bound: 0.4751232
time: 1.55 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728350
time: 1.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.84 seconds
NS_A1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4302200, upper bound: 0.4110871
NS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4724923, upper bound: 0.4718855
NS_A1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4245768, upper bound: 0.4070506
NS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4701703, upper bound: 0.4672139
NS_A1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4413121, upper bound: 0.4498335
NS_A1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4410916, upper bound: 0.4446392
NS_A1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.3956732, upper bound: 0.4109053
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4703183, upper bound: 0.4672139
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4737412, upper bound: 0.4751231
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4694741, upper bound: 0.4728179
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4737412, upper bound: 0.4751232
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4694741, upper bound: 0.4728350
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4744452, upper bound: 0.4751231
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728179
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4744452, upper bound: 0.4751232
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.84
Output dim: 7, lower bound: -0.4697319, upper bound: 0.4728350

## BFS NS instance: NS_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0935850, 0.0890990, -0.1093490, 0.1026416, -0.1962266, 0.1984480
1: -0.1159833, 0.0932957, -0.1298180, 0.1185851, -0.2345683, 0.2231137
2: -0.0912960, 0.1697145, -0.1021791, 0.1903590, -0.2816551, 0.2718936
3: -0.0964000, 0.2825388, -0.1047856, 0.3106183, -0.3957053, 0.3761551
4: -0.0934708, 0.1077525, -0.1121742, 0.1227691, -0.2162399, 0.2199267
5: -0.0887473, 0.1212136, -0.1079548, 0.1414181, -0.2301655, 0.2291684
6: -0.1071536, 0.1196248, -0.1254673, 0.1379221, -0.2450756, 0.2450921
7: 0.5148906, 1.1892891, 0.4827304, 1.2060466, -0.6911560, 0.7065586
8: -0.1317495, 0.1558080, -0.1434481, 0.1762606, -0.2934486, 0.2847758
9: -0.1175579, 0.1401378, -0.1345427, 0.1632463, -0.2808042, 0.2746805

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4469151, upper bound: 0.4540035
time: 1.26 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4443815, upper bound: 0.4368476
time: 1.30 seconds

## BFS NS instance: NS_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0764507, 0.0798936, -0.0977860, 0.0928688, -0.1693195, 0.1776796
1: -0.1041767, 0.0767915, -0.1200521, 0.1009589, -0.2051356, 0.1968436
2: -0.1152426, 0.1799014, -0.0999377, 0.1765995, -0.2918422, 0.2798391
3: -0.1121249, 0.2460560, -0.1025446, 0.2872041, -0.3882860, 0.3383632
4: -0.0850097, 0.0935270, -0.0986221, 0.1127340, -0.1977437, 0.1921491
5: -0.0720735, 0.1027499, -0.0941214, 0.1267809, -0.1988544, 0.1968713
6: -0.1076666, 0.1137021, -0.1123789, 0.1270595, -0.2347261, 0.2260809
7: 0.5576016, 1.2315543, 0.5095199, 1.2039645, -0.6463629, 0.7220343
8: -0.1558525, 0.1481317, -0.1408978, 0.1614636, -0.3041657, 0.2764481
9: -0.1068383, 0.1274213, -0.1230215, 0.1476693, -0.2545075, 0.2504427

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4701703, upper bound: 0.4670762
time: 1.36 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4701703, upper bound: 0.4672139
time: 1.11 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0867598, 0.0848978, -0.0823781, 0.0854310, -0.1721908, 0.1672759
1: -0.1102822, 0.0820967, -0.1098935, 0.0806609, -0.1909431, 0.1919903
2: -0.0937371, 0.1642655, -0.1247383, 0.1892326, -0.2829697, 0.2890038
3: -0.0975739, 0.2615685, -0.1192892, 0.2531271, -0.3405728, 0.3698638
4: -0.0889792, 0.0996733, -0.0907563, 0.0987995, -0.1877786, 0.1904296
5: -0.0807488, 0.1131730, -0.0775832, 0.1098682, -0.1906170, 0.1907561
6: -0.1016709, 0.1123729, -0.1145328, 0.1224016, -0.2240725, 0.2269057
7: 0.5398569, 1.1945708, 0.5476046, 1.2474949, -0.7076381, 0.6469662
8: -0.1339188, 0.1514051, -0.1663473, 0.1543822, -0.2761952, 0.3045724
9: -0.1103270, 0.1336575, -0.1126396, 0.1361240, -0.2464510, 0.2462971

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4253721, upper bound: 0.4070975
time: 1.26 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4253721, upper bound: 0.4672139
time: 1.19 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.1069579, 0.1008030, -0.0999883, 0.0929706, -0.1999285, 0.2007913
1: -0.1280418, 0.1149894, -0.1216323, 0.1051631, -0.2332049, 0.2366216
2: -0.1042567, 0.1883373, -0.0929622, 0.1760693, -0.2803260, 0.2812994
3: -0.1058743, 0.3050239, -0.0981591, 0.2993918, -0.3941625, 0.3920937
4: -0.1096246, 0.1209224, -0.0995975, 0.1145196, -0.2241443, 0.2205200
5: -0.1050485, 0.1385587, -0.0965260, 0.1292124, -0.2342609, 0.2350847
6: -0.1231418, 0.1366101, -0.1136280, 0.1267890, -0.2499307, 0.2502381
7: 0.4889502, 1.2104372, 0.4947502, 1.1910176, -0.7020675, 0.7156870
8: -0.1455300, 0.1735884, -0.1337706, 0.1623348, -0.2937457, 0.2931855
9: -0.1325868, 0.1606001, -0.1239739, 0.1488539, -0.2814407, 0.2845740

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
time: 1.29 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4402362, upper bound: 0.4463402
time: 1.30 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0958161, 0.0919450, -0.0813998, 0.0839573, -0.1797734, 0.1733448
1: -0.1186418, 0.0973871, -0.1088750, 0.0794717, -0.1981134, 0.2062621
2: -0.1020170, 0.1756287, -0.1162073, 0.1812061, -0.2832231, 0.2918360
3: -0.1036745, 0.2824283, -0.1134678, 0.2562633, -0.3497382, 0.3849950
4: -0.0971502, 0.1108445, -0.0890317, 0.0973113, -0.1944614, 0.1998761
5: -0.0916632, 0.1245512, -0.0766463, 0.1083534, -0.2000166, 0.2011975
6: -0.1110136, 0.1257623, -0.1088061, 0.1160462, -0.2270598, 0.2345684
7: 0.5149540, 1.2083722, 0.5428693, 1.2331438, -0.7181898, 0.6655029
8: -0.1430107, 0.1600209, -0.1575721, 0.1525174, -0.2832653, 0.3046522
9: -0.1216111, 0.1454768, -0.1107504, 0.1334492, -0.2550602, 0.2562272

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4053852, upper bound: 0.4243654
time: 2.04 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4703186
time: 1.49 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1069579, 0.1008030, -0.1039269, 0.0979006, -0.2048585, 0.2047299
1: -0.1280418, 0.1149894, -0.1258298, 0.1110555, -0.2390973, 0.2408192
2: -0.1042567, 0.1883373, -0.1039909, 0.1847036, -0.2889603, 0.2923282
3: -0.1058743, 0.3050239, -0.1056452, 0.3025747, -0.3962066, 0.3983936
4: -0.1096246, 0.1209224, -0.1059136, 0.1186842, -0.2283088, 0.2268360
5: -0.1050485, 0.1385587, -0.1014436, 0.1348257, -0.2398742, 0.2400023
6: -0.1231418, 0.1366101, -0.1199777, 0.1341332, -0.2572750, 0.2565878
7: 0.4889502, 1.2104372, 0.4910190, 1.2104201, -0.7213404, 0.7194182
8: -0.1455300, 0.1735884, -0.1453359, 0.1695338, -0.2986974, 0.3025331
9: -0.1325868, 0.1606001, -0.1299080, 0.1564579, -0.2890447, 0.2905082

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4411829, upper bound: 0.4531463
time: 1.43 seconds

## Relational analysis of NS_A2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4406129, upper bound: 0.4463420
time: 1.97 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0958161, 0.0919450, -0.0849535, 0.0874003, -0.1832164, 0.1768985
1: -0.1186418, 0.0973871, -0.1123080, 0.0822461, -0.2008879, 0.2096951
2: -0.1020170, 0.1756287, -0.1277222, 0.1921514, -0.2941684, 0.3033508
3: -0.1036745, 0.2824283, -0.1214749, 0.2575781, -0.3499256, 0.3918820
4: -0.0971502, 0.1108445, -0.0930117, 0.1007401, -0.1978903, 0.2038562
5: -0.0916632, 0.1245512, -0.0797170, 0.1129163, -0.2045795, 0.2042682
6: -0.1110136, 0.1257623, -0.1167506, 0.1251797, -0.2361933, 0.2425129
7: 0.5149540, 1.2083722, 0.5415164, 1.2528732, -0.7379192, 0.6668558
8: -0.1430107, 0.1600209, -0.1696321, 0.1569364, -0.2854687, 0.3146236
9: -0.1216111, 0.1454768, -0.1150942, 0.1392309, -0.2608420, 0.2605710

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4116144, upper bound: 0.4286138
time: 1.39 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4672186, upper bound: 0.4703279
time: 1.92 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1046151, 0.0991437, -0.0968850, 0.0913247, -0.1959398, 0.1960287
1: -0.1262071, 0.1108189, -0.1189266, 0.1001908, -0.2263979, 0.2297455
2: -0.1078963, 0.1866663, -0.0922837, 0.1731038, -0.2810001, 0.2789500
3: -0.1083936, 0.2965139, -0.0974172, 0.2919357, -0.3892588, 0.3829168
4: -0.1073618, 0.1188785, -0.0965664, 0.1116928, -0.2190546, 0.2154449
5: -0.1020240, 0.1356526, -0.0932511, 0.1252685, -0.2272924, 0.2289037
6: -0.1208249, 0.1355493, -0.1105044, 0.1237817, -0.2446066, 0.2460536
7: 0.4991210, 1.2170916, 0.5035098, 1.1903324, -0.6912115, 0.7135818
8: -0.1493586, 0.1711274, -0.1329259, 0.1591198, -0.2944155, 0.2899835
9: -0.1305054, 0.1582098, -0.1213179, 0.1446903, -0.2751957, 0.2795277

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
time: 1.72 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4389902, upper bound: 0.4462141
time: 1.26 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0942120, 0.0913096, -0.0794310, 0.0824907, -0.1767027, 0.1707406
1: -0.1174844, 0.0936532, -0.1069470, 0.0784923, -0.1959766, 0.2006002
2: -0.1056845, 0.1754242, -0.1158648, 0.1807053, -0.2863898, 0.2912890
3: -0.1063218, 0.2752956, -0.1129549, 0.2518110, -0.3480314, 0.3774368
4: -0.0963118, 0.1089739, -0.0874353, 0.0959499, -0.1922617, 0.1964092
5: -0.0893047, 0.1228357, -0.0749969, 0.1058259, -0.1951306, 0.1978326
6: -0.1099711, 0.1249934, -0.1083803, 0.1150033, -0.2249744, 0.2333737
7: 0.5237939, 1.2150544, 0.5488831, 1.2325625, -0.7087686, 0.6661713
8: -0.1469461, 0.1591021, -0.1569282, 0.1509073, -0.2856615, 0.3032407
9: -0.1202286, 0.1441636, -0.1090921, 0.1312530, -0.2514816, 0.2532558

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4070975, upper bound: 0.4253721
time: 1.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4703183
time: 1.38 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1046151, 0.0991437, -0.1006980, 0.0949942, -0.1996093, 0.1998418
1: -0.1262071, 0.1108189, -0.1230041, 0.1058907, -0.2320977, 0.2338230
2: -0.1078963, 0.1866663, -0.1033009, 0.1806534, -0.2885497, 0.2899672
3: -0.1083936, 0.2965139, -0.1049303, 0.2949116, -0.3911546, 0.3892672
4: -0.1073618, 0.1188785, -0.1020112, 0.1157478, -0.2231095, 0.2208897
5: -0.1020240, 0.1356526, -0.0974108, 0.1307188, -0.2327428, 0.2330634
6: -0.1208249, 0.1355493, -0.1160945, 0.1309932, -0.2518181, 0.2516438
7: 0.4991210, 1.2170916, 0.5000437, 1.2097397, -0.7106187, 0.7170478
8: -0.1493586, 0.1711274, -0.1445079, 0.1652364, -0.2984524, 0.2994151
9: -0.1305054, 0.1582098, -0.1264198, 0.1521213, -0.2826267, 0.2846296

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
time: 1.37 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4394293, upper bound: 0.4462281
time: 1.63 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0942120, 0.0913096, -0.0827587, 0.0857941, -0.1800061, 0.1740683
1: -0.1174844, 0.0936532, -0.1102499, 0.0809977, -0.1984821, 0.2039031
2: -0.1056845, 0.1754242, -0.1273754, 0.1916493, -0.2973338, 0.3027995
3: -0.1063218, 0.2752956, -0.1209681, 0.2531343, -0.3482476, 0.3843615
4: -0.0963118, 0.1089739, -0.0912667, 0.0991545, -0.1954663, 0.2002405
5: -0.0893047, 0.1228357, -0.0778483, 0.1101742, -0.1994789, 0.2006840
6: -0.1099711, 0.1249934, -0.1162944, 0.1243602, -0.2343312, 0.2412878
7: 0.5237939, 1.2150544, 0.5474752, 1.2522730, -0.7284790, 0.6675792
8: -0.1469461, 0.1591021, -0.1689903, 0.1551694, -0.2878840, 0.3132522
9: -0.1202286, 0.1441636, -0.1138178, 0.1368314, -0.2570600, 0.2579814

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4134762, upper bound: 0.4294821
time: 0.99 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4674206, upper bound: 0.4703273
time: 1.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.18 seconds
NS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4469151, upper bound: 0.4540035
NS_A1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4443815, upper bound: 0.4368476
NS_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4701703, upper bound: 0.4670762
NS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4701703, upper bound: 0.4672139
NS_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4253721, upper bound: 0.4070975
NS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4253721, upper bound: 0.4672139
NS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
NS_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4402362, upper bound: 0.4463402
NS_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4053852, upper bound: 0.4243654
NS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4703186
NS_A2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4411829, upper bound: 0.4531463
NS_A2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4406129, upper bound: 0.4463420
NS_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4116144, upper bound: 0.4286138
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4672186, upper bound: 0.4703279
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
NS_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4389902, upper bound: 0.4462141
NS_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4070975, upper bound: 0.4253721
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4703183
NS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
NS_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4394293, upper bound: 0.4462281
NS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4134762, upper bound: 0.4294821
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 7, lower bound: -0.4674206, upper bound: 0.4703273

## BFS NS instance: NS_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0933805, 0.0888441, -0.1083648, 0.1013636, -0.1947441, 0.1972089
1: -0.1156907, 0.0929024, -0.1285212, 0.1170265, -0.2327172, 0.2214235
2: -0.0903364, 0.1691786, -0.0983846, 0.1879209, -0.2782573, 0.2675632
3: -0.0956253, 0.2822672, -0.1017878, 0.3092800, -0.3935928, 0.3729061
4: -0.0931082, 0.1074474, -0.1103754, 0.1215679, -0.2146761, 0.2178228
5: -0.0884628, 0.1209092, -0.1066798, 0.1399637, -0.2284265, 0.2275890
6: -0.1067639, 0.1190558, -0.1236118, 0.1357023, -0.2424662, 0.2426675
7: 0.5152664, 1.1878800, 0.4844824, 1.2005260, -0.6852596, 0.7033976
8: -0.1307035, 0.1554419, -0.1393781, 0.1744109, -0.2904478, 0.2801530
9: -0.1171892, 0.1396630, -0.1329147, 0.1611971, -0.2783863, 0.2725777

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4469151, upper bound: 0.4540035
time: 1.30 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4469151, upper bound: 0.4540035
time: 1.29 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0764507, 0.0798936, -0.0920738, 0.0889880, -0.1654387, 0.1719674
1: -0.1041767, 0.0767915, -0.1149471, 0.0897028, -0.1938796, 0.1917386
2: -0.1152426, 0.1799014, -0.0983068, 0.1707202, -0.2859628, 0.2782083
3: -0.1121249, 0.2460560, -0.1009108, 0.2715472, -0.3726614, 0.3367104
4: -0.0850097, 0.0935270, -0.0933924, 0.1059650, -0.1909747, 0.1869194
5: -0.0720735, 0.1027499, -0.0865601, 0.1197841, -0.1918576, 0.1893100
6: -0.1076666, 0.1137021, -0.1066773, 0.1199452, -0.2276117, 0.2203794
7: 0.5576016, 1.2315543, 0.5284995, 1.2022598, -0.6446582, 0.7030547
8: -0.1558525, 0.1481317, -0.1389538, 0.1558971, -0.2985215, 0.2744907
9: -0.1068383, 0.1274213, -0.1165727, 0.1400096, -0.2468478, 0.2439940

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4106508
time: 1.20 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4670149
time: 1.21 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0764507, 0.0798936, -0.0903612, 0.0882943, -0.1647450, 0.1702548
1: -0.1041767, 0.0767915, -0.1137008, 0.0867008, -0.1908775, 0.1904923
2: -0.1152426, 0.1799014, -0.1021193, 0.1704603, -0.2857029, 0.2820207
3: -0.1121249, 0.2460560, -0.1036478, 0.2642413, -0.3653646, 0.3395182
4: -0.0850097, 0.0935270, -0.0927985, 0.1039791, -0.1889888, 0.1863255
5: -0.0720735, 0.1027499, -0.0845034, 0.1179460, -0.1900195, 0.1872533
6: -0.1076666, 0.1137021, -0.1057331, 0.1190665, -0.2267331, 0.2194351
7: 0.5576016, 1.2315543, 0.5373719, 1.2091129, -0.6515113, 0.6941823
8: -0.1558525, 0.1481317, -0.1429931, 0.1553694, -0.2980044, 0.2788052
9: -0.1068383, 0.1274213, -0.1150885, 0.1391758, -0.2460141, 0.2425098

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 213

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4108216
time: 1.13 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4671675
time: 5.38 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0867598, 0.0848978, -0.0788230, 0.0827219, -0.1694817, 0.1637208
1: -0.1102822, 0.0820967, -0.1063726, 0.0788340, -0.1891162, 0.1884693
2: -0.0937371, 0.1642655, -0.1240770, 0.1882870, -0.2820241, 0.2883425
3: -0.0975739, 0.2615685, -0.1183691, 0.2450995, -0.3324215, 0.3689349
4: -0.0889792, 0.0996733, -0.0878371, 0.0962745, -0.1852537, 0.1875104
5: -0.0807488, 0.1131730, -0.0745458, 0.1053667, -0.1861155, 0.1877187
6: -0.1016709, 0.1123729, -0.1137164, 0.1210049, -0.2226758, 0.2260893
7: 0.5398569, 1.1945708, 0.5585858, 1.2463262, -0.7064694, 0.6359850
8: -0.1339188, 0.1514051, -0.1651459, 0.1513891, -0.2728848, 0.3033837
9: -0.1103270, 0.1336575, -0.1113263, 0.1320693, -0.2423963, 0.2449837

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4109053, upper bound: 0.4671675
time: 1.06 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4109053, upper bound: 0.4671675
time: 1.09 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1059746, 0.0995225, -0.0997388, 0.0927127, -0.1986873, 0.1992613
1: -0.1267498, 0.1134369, -0.1212963, 0.1047702, -0.2315200, 0.2347332
2: -0.1004348, 0.1859061, -0.0919915, 0.1754946, -0.2759295, 0.2778976
3: -0.1028480, 0.3036845, -0.0973913, 0.2990547, -0.3908067, 0.3899809
4: -0.1078390, 0.1197228, -0.0991821, 0.1142109, -0.2220499, 0.2189048
5: -0.1037750, 0.1371027, -0.0962387, 0.1288410, -0.2326160, 0.2333414
6: -0.1212912, 0.1343957, -0.1131843, 0.1262127, -0.2475038, 0.2475800
7: 0.4907004, 1.2048273, 0.4951886, 1.1896045, -0.6989042, 0.7096387
8: -0.1414087, 0.1717344, -0.1327279, 0.1619141, -0.2890582, 0.2901818
9: -0.1309635, 0.1585477, -0.1235998, 0.1483247, -0.2792882, 0.2821475

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
time: 1.40 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
time: 1.40 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0922461, 0.0894378, -0.0813998, 0.0839573, -0.1762033, 0.1708376
1: -0.1153477, 0.0901182, -0.1088750, 0.0794717, -0.1948194, 0.1989932
2: -0.1009092, 0.1718859, -0.1162073, 0.1812061, -0.2821153, 0.2880932
3: -0.1025453, 0.2715841, -0.1134678, 0.2562633, -0.3485959, 0.3740480
4: -0.0939324, 0.1063829, -0.0890317, 0.0973113, -0.1912436, 0.1954146
5: -0.0868111, 0.1201753, -0.0766463, 0.1083534, -0.1951644, 0.1968216
6: -0.1073529, 0.1210766, -0.1088061, 0.1160462, -0.2233991, 0.2298827
7: 0.5282941, 1.2071550, 0.5428693, 1.2331438, -0.7048497, 0.6642857
8: -0.1416453, 0.1565874, -0.1575721, 0.1525174, -0.2818858, 0.3009295
9: -0.1173118, 0.1408071, -0.1107504, 0.1334492, -0.2507610, 0.2515576

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4107314, upper bound: 0.3957002
time: 1.11 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107314, upper bound: 0.4703186
time: 1.20 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0922461, 0.0894378, -0.0849535, 0.0874003, -0.1796463, 0.1743913
1: -0.1153477, 0.0901182, -0.1123080, 0.0822461, -0.1975938, 0.2024262
2: -0.1009092, 0.1718859, -0.1277222, 0.1921514, -0.2930607, 0.2996081
3: -0.1025453, 0.2715841, -0.1214749, 0.2575781, -0.3487809, 0.3809482
4: -0.0939324, 0.1063829, -0.0930117, 0.1007401, -0.1946725, 0.1993946
5: -0.0868111, 0.1201753, -0.0797170, 0.1129163, -0.1997274, 0.1998923
6: -0.1073529, 0.1210766, -0.1167506, 0.1251797, -0.2325326, 0.2378273
7: 0.5282941, 1.2071550, 0.5415164, 1.2528732, -0.7196514, 0.6656386
8: -0.1416453, 0.1565874, -0.1696321, 0.1569364, -0.2840977, 0.3108923
9: -0.1173118, 0.1408071, -0.1150942, 0.1392309, -0.2565427, 0.2559013

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4352931, upper bound: 0.4482035
time: 1.18 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4340374, upper bound: 0.4343172
time: 1.38 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1034723, 0.0978180, -0.0966374, 0.0910687, -0.1945410, 0.1944553
1: -0.1248683, 0.1091899, -0.1185940, 0.0998014, -0.2246697, 0.2277839
2: -0.1040871, 0.1842016, -0.0913194, 0.1725340, -0.2766212, 0.2755210
3: -0.1053236, 0.2950933, -0.0966478, 0.2916008, -0.3858709, 0.3807200
4: -0.1054879, 0.1176440, -0.0961542, 0.1113868, -0.2168747, 0.2137982
5: -0.1006514, 0.1341312, -0.0929663, 0.1249000, -0.2255515, 0.2270975
6: -0.1189018, 0.1333211, -0.1100651, 0.1232101, -0.2421118, 0.2433861
7: 0.5009334, 1.2114611, 0.5039439, 1.1889242, -0.6879908, 0.7075171
8: -0.1452011, 0.1691238, -0.1318837, 0.1587028, -0.2896931, 0.2868277
9: -0.1288489, 0.1560887, -0.1209473, 0.1441649, -0.2730138, 0.2770360

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
time: 1.62 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
time: 1.89 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0905257, 0.0887121, -0.0794310, 0.0824907, -0.1730164, 0.1681432
1: -0.1140689, 0.0870910, -0.1069470, 0.0784923, -0.1925611, 0.1940380
2: -0.1045961, 0.1715959, -0.1158648, 0.1807053, -0.2853014, 0.2874607
3: -0.1052302, 0.2642792, -0.1129549, 0.2518110, -0.3469294, 0.3663307
4: -0.0933045, 0.1043664, -0.0874353, 0.0959499, -0.1892544, 0.1918017
5: -0.0847376, 0.1183122, -0.0749969, 0.1058259, -0.1905636, 0.1933091
6: -0.1063513, 0.1201283, -0.1083803, 0.1150033, -0.2213546, 0.2285085
7: 0.5371861, 1.2138150, 0.5488831, 1.2325625, -0.6953765, 0.6649319
8: -0.1455738, 0.1560097, -0.1569282, 0.1509073, -0.2842838, 0.2998825
9: -0.1157812, 0.1399214, -0.1090921, 0.1312530, -0.2470342, 0.2490136

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4109053, upper bound: 0.3956732
time: 1.38 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4109053, upper bound: 0.4703183
time: 1.47 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1034723, 0.0978180, -0.1004517, 0.0946701, -0.1981424, 0.1982697
1: -0.1248683, 0.1091899, -0.1226787, 0.1055009, -0.2303692, 0.2318686
2: -0.1040871, 0.1842016, -0.1023253, 0.1800359, -0.2841230, 0.2865269
3: -0.1053236, 0.2950933, -0.1041475, 0.2945782, -0.3877628, 0.3870669
4: -0.1054879, 0.1176440, -0.1015603, 0.1154449, -0.2209328, 0.2192043
5: -0.1006514, 0.1341312, -0.0970913, 0.1303525, -0.2310040, 0.2312225
6: -0.1189018, 0.1333211, -0.1156272, 0.1304291, -0.2493308, 0.2489483
7: 0.5009334, 1.2114611, 0.5004776, 1.2082846, -0.7073512, 0.7109696
8: -0.1452011, 0.1691238, -0.1434397, 0.1647675, -0.2936466, 0.2962456
9: -0.1288489, 0.1560887, -0.1260087, 0.1516017, -0.2804506, 0.2820974

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
time: 2.08 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
time: 1.39 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0905257, 0.0887121, -0.0827587, 0.0857941, -0.1763198, 0.1714709
1: -0.1140689, 0.0870910, -0.1102499, 0.0809977, -0.1950666, 0.1973410
2: -0.1045961, 0.1715959, -0.1273754, 0.1916493, -0.2962454, 0.2989713
3: -0.1052302, 0.2642792, -0.1209681, 0.2531343, -0.3471455, 0.3732609
4: -0.0933045, 0.1043664, -0.0912667, 0.0991545, -0.1924590, 0.1956330
5: -0.0847376, 0.1183122, -0.0778483, 0.1101742, -0.1949118, 0.1961605
6: -0.1063513, 0.1201283, -0.1162944, 0.1243602, -0.2307115, 0.2364226
7: 0.5371861, 1.2138150, 0.5474752, 1.2522730, -0.7150869, 0.6663398
8: -0.1455738, 0.1560097, -0.1689903, 0.1551694, -0.2865116, 0.3098790
9: -0.1157812, 0.1399214, -0.1138178, 0.1368314, -0.2526125, 0.2537392

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 52

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4274005, upper bound: 0.4140792
time: 1.20 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4274005, upper bound: 0.4703273
time: 1.13 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.86 seconds
NS_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4469151, upper bound: 0.4540035
NS_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4469151, upper bound: 0.4540035
NS_A1_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4106508
NS_A1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4670149
NS_A1_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4108216
NS_A1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4671675
NS_A1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4109053, upper bound: 0.4671675
NS_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4109053, upper bound: 0.4671675
NS_A2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
NS_A2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
NS_A2_A1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4107314, upper bound: 0.3957002
NS_A2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4107314, upper bound: 0.4703186
NS_A2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4352931, upper bound: 0.4482035
NS_A2_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4340374, upper bound: 0.4343172
NS_A2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
NS_A2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
NS_A2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4109053, upper bound: 0.3956732
NS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4109053, upper bound: 0.4703183
NS_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
NS_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
NS_A2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4274005, upper bound: 0.4140792
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.86
Output dim: 7, lower bound: -0.4274005, upper bound: 0.4703273

## BFS NS instance: NS_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0933805, 0.0888441, -0.0980853, 0.0925865, -0.1859670, 0.1869294
1: -0.1156907, 0.0929024, -0.1198996, 0.1013532, -0.2170439, 0.2128020
2: -0.0903364, 0.1691786, -0.0964998, 0.1756759, -0.2660123, 0.2656784
3: -0.0956253, 0.2822672, -0.0998248, 0.2887648, -0.3730428, 0.3709024
4: -0.0931082, 0.1074474, -0.0982846, 0.1126628, -0.2057710, 0.2057320
5: -0.0884628, 0.1209092, -0.0943381, 0.1269484, -0.2154113, 0.2152472
6: -0.1067639, 0.1190558, -0.1119509, 0.1261252, -0.2328891, 0.2310066
7: 0.5152664, 1.1878800, 0.5078099, 1.1987830, -0.6835166, 0.6800701
8: -0.1307035, 0.1554419, -0.1371828, 0.1612044, -0.2771909, 0.2779099
9: -0.1171892, 0.1396630, -0.1226450, 0.1473843, -0.2645735, 0.2623079

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
time: 1.64 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
time: 1.72 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0933805, 0.0888441, -0.0803002, 0.0835063, -0.1768868, 0.1691443
1: -0.1156907, 0.0929024, -0.1074573, 0.0792735, -0.1949642, 0.2003596
2: -0.0903364, 0.1691786, -0.1205310, 0.1848320, -0.2751683, 0.2897096
3: -0.0956253, 0.2822672, -0.1156189, 0.2487625, -0.3332376, 0.3868794
4: -0.0931082, 0.1074474, -0.0884542, 0.0969458, -0.1900540, 0.1959016
5: -0.0884628, 0.1209092, -0.0757205, 0.1069769, -0.1954397, 0.1966296
6: -0.1067639, 0.1190558, -0.1113572, 0.1183493, -0.2251133, 0.2304129
7: 0.5152664, 1.1878800, 0.5535604, 1.2413981, -0.7261317, 0.6343197
8: -0.1307035, 0.1554419, -0.1614545, 0.1520908, -0.2685842, 0.3034907
9: -0.1171892, 0.1396630, -0.1102858, 0.1329709, -0.2501601, 0.2499488

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
time: 1.44 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
time: 1.45 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0920738, 0.0889880, -0.1628426, 0.1693840
1: -0.1015633, 0.0750528, -0.1149471, 0.0897028, -0.1912661, 0.1900000
2: -0.1146101, 0.1790866, -0.0983068, 0.1707202, -0.2853303, 0.2773934
3: -0.1112695, 0.2397249, -0.1009108, 0.2715472, -0.3717985, 0.3302076
4: -0.0827738, 0.0911205, -0.0933924, 0.1059650, -0.1887388, 0.1845129
5: -0.0691687, 0.1004435, -0.0865601, 0.1197841, -0.1889528, 0.1870036
6: -0.1069577, 0.1124067, -0.1066773, 0.1199452, -0.2269028, 0.2190840
7: 0.5678793, 1.2305298, 0.5284995, 1.2022598, -0.6343805, 0.7020302
8: -0.1547139, 0.1452890, -0.1389538, 0.1558971, -0.2973952, 0.2712741
9: -0.1051957, 0.1236046, -0.1165727, 0.1400096, -0.2452053, 0.2401773

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 147

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3926313, upper bound: 0.4672882
time: 1.40 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3926313, upper bound: 0.4672882
time: 1.15 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0903612, 0.0882943, -0.1621490, 0.1676714
1: -0.1015633, 0.0750528, -0.1137008, 0.0867008, -0.1882641, 0.1887536
2: -0.1146101, 0.1790866, -0.1021193, 0.1704603, -0.2850704, 0.2812059
3: -0.1112695, 0.2397249, -0.1036478, 0.2642413, -0.3645017, 0.3330418
4: -0.0827738, 0.0911205, -0.0927985, 0.1039791, -0.1867529, 0.1839190
5: -0.0691687, 0.1004435, -0.0845034, 0.1179460, -0.1871147, 0.1849470
6: -0.1069577, 0.1124067, -0.1057331, 0.1190665, -0.2260242, 0.2181398
7: 0.5678793, 1.2305298, 0.5373719, 1.2091129, -0.6412336, 0.6931579
8: -0.1547139, 0.1452890, -0.1429931, 0.1553694, -0.2968782, 0.2756086
9: -0.1051957, 0.1236046, -0.1150885, 0.1391758, -0.2443715, 0.2386931

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4671675
time: 1.23 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4671675
time: 1.67 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0867598, 0.0848978, -0.0764527, 0.0798426, -0.1666024, 0.1613505
1: -0.1102822, 0.0820967, -0.1040135, 0.0767050, -0.1869871, 0.1861103
2: -0.0937371, 0.1642655, -0.1152034, 0.1798107, -0.2735478, 0.2794690
3: -0.0975739, 0.2615685, -0.1120417, 0.2448600, -0.3320874, 0.3625788
4: -0.0889792, 0.0996733, -0.0849470, 0.0934772, -0.1824564, 0.1846203
5: -0.0807488, 0.1131730, -0.0720161, 0.1027371, -0.1834859, 0.1851891
6: -0.1016709, 0.1123729, -0.1076145, 0.1135762, -0.2152471, 0.2199874
7: 0.5398569, 1.1945708, 0.5594605, 1.2314520, -0.6915951, 0.6351103
8: -0.1339188, 0.1514051, -0.1557337, 0.1480022, -0.2690696, 0.2937506
9: -0.1103270, 0.1336575, -0.1067124, 0.1273035, -0.2376305, 0.2403699

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4671675
time: 1.61 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4671675
time: 1.46 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0867598, 0.0848978, -0.0789451, 0.0830775, -0.1698373, 0.1638429
1: -0.1102822, 0.0820967, -0.1066639, 0.0791660, -0.1894481, 0.1887607
2: -0.0937371, 0.1642655, -0.1266763, 0.1907043, -0.2844414, 0.2909418
3: -0.0975739, 0.2615685, -0.1200563, 0.2450995, -0.3324219, 0.3706635
4: -0.0889792, 0.0996733, -0.0882618, 0.0966228, -0.1856019, 0.1879351
5: -0.0807488, 0.1131730, -0.0747229, 0.1056612, -0.1864100, 0.1878959
6: -0.1016709, 0.1123729, -0.1154277, 0.1229839, -0.2246548, 0.2278006
7: 0.5398569, 1.1945708, 0.5584624, 1.2509956, -0.7111388, 0.6361084
8: -0.1339188, 0.1514051, -0.1677930, 0.1519392, -0.2734690, 0.3061555
9: -0.1103270, 0.1336575, -0.1127825, 0.1327723, -0.2430993, 0.2464399

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_A2_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4333769, upper bound: 0.4445809
time: 1.30 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4322512, upper bound: 0.4322013
time: 1.34 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0960689, 0.0917158, -0.0997388, 0.0927127, -0.1887816, 0.1914546
1: -0.1184357, 0.0978159, -0.1212963, 0.1047702, -0.2232058, 0.2191122
2: -0.0985660, 0.1746778, -0.0919915, 0.1754946, -0.2740607, 0.2666693
3: -0.1009334, 0.2837050, -0.0973913, 0.2990547, -0.3888512, 0.3699838
4: -0.0967473, 0.1108283, -0.0991821, 0.1142109, -0.2109582, 0.2100103
5: -0.0919634, 0.1246278, -0.0962387, 0.1288410, -0.2208044, 0.2208665
6: -0.1104989, 0.1248560, -0.1131843, 0.1262127, -0.2367116, 0.2380403
7: 0.5135758, 1.2031047, 0.4951886, 1.1896045, -0.6760287, 0.7079161
8: -0.1392483, 0.1596952, -0.1327279, 0.1619141, -0.2868285, 0.2781008
9: -0.1212771, 0.1451643, -0.1235998, 0.1483247, -0.2696017, 0.2687641

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
time: 1.42 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
time: 1.83 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0786245, 0.0824976, -0.0997388, 0.0927127, -0.1713372, 0.1822364
1: -0.1061348, 0.0787401, -0.1212963, 0.1047702, -0.2109049, 0.2000364
2: -0.1228928, 0.1868956, -0.0919915, 0.1754946, -0.2983875, 0.2788872
3: -0.1169206, 0.2456895, -0.0973913, 0.2990547, -0.4050065, 0.3321034
4: -0.0874815, 0.0960312, -0.0991821, 0.1142109, -0.2016924, 0.1952133
5: -0.0743530, 0.1051416, -0.0962387, 0.1288410, -0.2031940, 0.2013803
6: -0.1127464, 0.1197582, -0.1131843, 0.1262127, -0.2389591, 0.2329425
7: 0.5576000, 1.2456722, 0.4951886, 1.1896045, -0.6320046, 0.7504836
8: -0.1636541, 0.1512056, -0.1327279, 0.1619141, -0.3123918, 0.2700606
9: -0.1105810, 0.1316433, -0.1235998, 0.1483247, -0.2589057, 0.2552431

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
time: 1.63 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
time: 1.53 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0922461, 0.0894378, -0.0779800, 0.0813531, -0.1735991, 0.1674178
1: -0.1153477, 0.0901182, -0.1055406, 0.0777137, -0.1930614, 0.1956588
2: -0.1009092, 0.1718859, -0.1155450, 0.1802684, -0.2811776, 0.2874309
3: -0.1025453, 0.2715841, -0.1125347, 0.2485853, -0.3407917, 0.3731069
4: -0.0939324, 0.1063829, -0.0862538, 0.0948797, -0.1888121, 0.1926367
5: -0.0868111, 0.1201753, -0.0737139, 0.1040866, -0.1908976, 0.1938891
6: -0.1073529, 0.1210766, -0.1080149, 0.1143332, -0.2216861, 0.2290916
7: 0.5282941, 1.2071550, 0.5533967, 1.2319946, -0.7037005, 0.6537583
8: -0.1416453, 0.1565874, -0.1563694, 0.1496641, -0.2786515, 0.2997372
9: -0.1173118, 0.1408071, -0.1078933, 0.1295430, -0.2468548, 0.2487005

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107314, upper bound: 0.4702707
time: 1.45 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4107314, upper bound: 0.4702707
time: 1.32 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.0966374, 0.0910687, -0.1852183, 0.1875125
1: -0.1170394, 0.0935416, -0.1185940, 0.0998014, -0.2168408, 0.2121356
2: -0.1021949, 0.1741340, -0.0913194, 0.1725340, -0.2747289, 0.2654534
3: -0.1034629, 0.2760351, -0.0966478, 0.2916008, -0.3839701, 0.3616617
4: -0.0956366, 0.1086472, -0.0961542, 0.1113868, -0.2070234, 0.2048013
5: -0.0892088, 0.1225371, -0.0929663, 0.1249000, -0.2141088, 0.2155034
6: -0.1092287, 0.1237412, -0.1100651, 0.1232101, -0.2324388, 0.2338062
7: 0.5229961, 1.2097473, 0.5039439, 1.1889242, -0.6659281, 0.7058034
8: -0.1430700, 0.1584855, -0.1318837, 0.1587028, -0.2875176, 0.2761472
9: -0.1196212, 0.1433626, -0.1209473, 0.1441649, -0.2637861, 0.2643099

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
time: 1.33 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
time: 1.62 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0777754, 0.0820198, -0.0966374, 0.0910687, -0.1688442, 0.1786571
1: -0.1054629, 0.0785578, -0.1185940, 0.0998014, -0.2052643, 0.1971518
2: -0.1275831, 0.1913313, -0.0913194, 0.1725340, -0.3001171, 0.2826506
3: -0.1202425, 0.2416290, -0.0966478, 0.2916008, -0.4009505, 0.3273985
4: -0.0873437, 0.0956680, -0.0961542, 0.1113868, -0.1987304, 0.1918222
5: -0.0734341, 0.1047147, -0.0929663, 0.1249000, -0.1983341, 0.1976810
6: -0.1157950, 0.1232789, -0.1100651, 0.1232101, -0.2390051, 0.2333439
7: 0.5640168, 1.2531587, 0.5039439, 1.1889242, -0.6249074, 0.7492148
8: -0.1684172, 0.1508236, -0.1318837, 0.1587028, -0.3141850, 0.2690744
9: -0.1129952, 0.1313578, -0.1209473, 0.1441649, -0.2571601, 0.2523052

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 159

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
time: 1.29 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
time: 1.31 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0905257, 0.0887121, -0.0764527, 0.0798426, -0.1703683, 0.1651648
1: -0.1140689, 0.0870910, -0.1040135, 0.0767050, -0.1907738, 0.1911046
2: -0.1045961, 0.1715959, -0.1152034, 0.1798107, -0.2844068, 0.2867993
3: -0.1052302, 0.2642792, -0.1120417, 0.2448600, -0.3398532, 0.3654094
4: -0.0933045, 0.1043664, -0.0849470, 0.0934772, -0.1867818, 0.1893134
5: -0.0847376, 0.1183122, -0.0720161, 0.1027371, -0.1874747, 0.1903283
6: -0.1063513, 0.1201283, -0.1076145, 0.1135762, -0.2199275, 0.2277427
7: 0.5371861, 1.2138150, 0.5594605, 1.2314520, -0.6942659, 0.6543545
8: -0.1455738, 0.1560097, -0.1557337, 0.1480022, -0.2810864, 0.2986999
9: -0.1157812, 0.1399214, -0.1067124, 0.1273035, -0.2430847, 0.2466339

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4108216, upper bound: 0.4700172
time: 1.21 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4108216, upper bound: 0.4700172
time: 1.17 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.1004517, 0.0946701, -0.1888197, 0.1913268
1: -0.1170394, 0.0935416, -0.1226787, 0.1055009, -0.2225403, 0.2162203
2: -0.1021949, 0.1741340, -0.1023253, 0.1800359, -0.2822308, 0.2764594
3: -0.1034629, 0.2760351, -0.1041475, 0.2945782, -0.3858597, 0.3680039
4: -0.0956366, 0.1086472, -0.1015603, 0.1154449, -0.2110816, 0.2102075
5: -0.0892088, 0.1225371, -0.0970913, 0.1303525, -0.2195613, 0.2196285
6: -0.1092287, 0.1237412, -0.1156272, 0.1304291, -0.2396578, 0.2393684
7: 0.5229961, 1.2097473, 0.5004776, 1.2082846, -0.6852885, 0.7083125
8: -0.1430700, 0.1584855, -0.1434397, 0.1647675, -0.2914931, 0.2855222
9: -0.1196212, 0.1433626, -0.1260087, 0.1516017, -0.2712229, 0.2693712

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
time: 1.59 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
time: 1.87 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0777754, 0.0820198, -0.1004517, 0.0946701, -0.1724456, 0.1824715
1: -0.1054629, 0.0785578, -0.1226787, 0.1055009, -0.2109639, 0.2012365
2: -0.1275831, 0.1913313, -0.1023253, 0.1800359, -0.3076190, 0.2936566
3: -0.1202425, 0.2416290, -0.1041475, 0.2945782, -0.4029191, 0.3338267
4: -0.0873437, 0.0956680, -0.1015603, 0.1154449, -0.2027886, 0.1972284
5: -0.0734341, 0.1047147, -0.0970913, 0.1303525, -0.2037866, 0.2018060
6: -0.1157950, 0.1232789, -0.1156272, 0.1304291, -0.2462240, 0.2389061
7: 0.5640168, 1.2531587, 0.5004776, 1.2082846, -0.6442678, 0.7526811
8: -0.1684172, 0.1508236, -0.1434397, 0.1647675, -0.3183270, 0.2785937
9: -0.1129952, 0.1313578, -0.1260087, 0.1516017, -0.2645969, 0.2573665

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
time: 1.48 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
time: 1.31 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0905257, 0.0887121, -0.0791171, 0.0830775, -0.1736032, 0.1678292
1: -0.1140689, 0.0870910, -0.1067001, 0.0791660, -0.1932348, 0.1937912
2: -0.1045961, 0.1715959, -0.1267049, 0.1907043, -0.2953004, 0.2983008
3: -0.1052302, 0.2642792, -0.1200563, 0.2451021, -0.3390093, 0.3723400
4: -0.0933045, 0.1043664, -0.0883132, 0.0966228, -0.1899273, 0.1926796
5: -0.0847376, 0.1183122, -0.0747792, 0.1056612, -0.1903988, 0.1930914
6: -0.1063513, 0.1201283, -0.1154611, 0.1229839, -0.2293352, 0.2355894
7: 0.5371861, 1.2138150, 0.5584624, 1.2510691, -0.7138830, 0.6553526
8: -0.1455738, 0.1560097, -0.1677930, 0.1520914, -0.2831322, 0.3086958
9: -0.1157812, 0.1399214, -0.1127825, 0.1327881, -0.2485693, 0.2527039

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4266608, upper bound: 0.4700176
time: 1.49 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4266608, upper bound: 0.4136123
time: 1.18 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 8.25 seconds
NS_A1_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
NS_A1_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
NS_A1_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
NS_A1_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4452555, upper bound: 0.4540035
NS_A1_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.3926313, upper bound: 0.4672882
NS_A1_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.3926313, upper bound: 0.4672882
NS_A1_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4671675
NS_A1_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.3926270, upper bound: 0.4671675
NS_A1_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4671675
NS_A1_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4671675
NS_A1_A2_B2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4333769, upper bound: 0.4445809
NS_A1_A2_B2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4322512, upper bound: 0.4322013
NS_A2_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
NS_A2_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
NS_A2_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
NS_A2_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4550466, upper bound: 0.4486251
NS_A2_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4107314, upper bound: 0.4702707
NS_A2_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4107314, upper bound: 0.4702707
NS_A2_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
NS_A2_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
NS_A2_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
NS_A2_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4554192, upper bound: 0.4486022
NS_A2_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4108216, upper bound: 0.4700172
NS_A2_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4108216, upper bound: 0.4700172
NS_A2_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
NS_A2_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
NS_A2_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
NS_A2_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4560994, upper bound: 0.4486300
NS_A2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4266608, upper bound: 0.4700176
NS_A2_A2_B2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 8.25
Output dim: 7, lower bound: -0.4266608, upper bound: 0.4136123

## BFS NS instance: NS_A1_A1_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0933805, 0.0888441, -0.0948623, 0.0895323, -0.1829129, 0.1837065
1: -0.1156907, 0.0929024, -0.1165938, 0.0960175, -0.2117082, 0.2094962
2: -0.0903364, 0.1691786, -0.0880917, 0.1697058, -0.2600422, 0.2572703
3: -0.0956253, 0.2822672, -0.0939386, 0.2864438, -0.3706265, 0.3649832
4: -0.0931082, 0.1074474, -0.0939795, 0.1089960, -0.2021042, 0.2014270
5: -0.0884628, 0.1209092, -0.0904814, 0.1225272, -0.2109900, 0.2113906
6: -0.1067639, 0.1190558, -0.1076167, 0.1199760, -0.2267400, 0.2266724
7: 0.5152664, 1.1878800, 0.5104500, 1.1843321, -0.6690657, 0.6701528
8: -0.1307035, 0.1554419, -0.1283156, 0.1564311, -0.2719671, 0.2688452
9: -0.1171892, 0.1396630, -0.1183814, 0.1411869, -0.2583761, 0.2580443

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4503920, upper bound: 0.4503495
time: 1.45 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4699189, upper bound: 0.4740602
time: 2.31 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0933805, 0.0888441, -0.0982765, 0.0930481, -0.1864286, 0.1871206
1: -0.1156907, 0.0929024, -0.1203287, 0.1017889, -0.2174795, 0.2132310
2: -0.0903364, 0.1691786, -0.0990608, 0.1768362, -0.2671726, 0.2682394
3: -0.0956253, 0.2822672, -0.1014158, 0.2888208, -0.3730990, 0.3725687
4: -0.0931082, 0.1074474, -0.0988561, 0.1130983, -0.2062065, 0.2063035
5: -0.0884628, 0.1209092, -0.0946136, 0.1273666, -0.2158294, 0.2155228
6: -0.1067639, 0.1190558, -0.1126643, 0.1272558, -0.2340197, 0.2317201
7: 0.5152664, 1.1878800, 0.5075647, 1.2035705, -0.6883041, 0.6803154
8: -0.1307035, 0.1554419, -0.1398074, 0.1619295, -0.2779350, 0.2808362
9: -0.1171892, 0.1396630, -0.1234010, 0.1481843, -0.2653735, 0.2630640

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4423826, upper bound: 0.4551691
time: 1.47 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4699189, upper bound: 0.4740602
time: 1.68 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0933805, 0.0888441, -0.0774087, 0.0804739, -0.1738544, 0.1662529
1: -0.1156907, 0.0929024, -0.1046625, 0.0770577, -0.1927484, 0.1975648
2: -0.0903364, 0.1691786, -0.1116873, 0.1763447, -0.2666811, 0.2808660
3: -0.0956253, 0.2822672, -0.1093048, 0.2481245, -0.3325249, 0.3805428
4: -0.0931082, 0.1074474, -0.0851865, 0.0940043, -0.1871126, 0.1926339
5: -0.0884628, 0.1209092, -0.0730343, 0.1033174, -0.1917802, 0.1939435
6: -0.1067639, 0.1190558, -0.1052018, 0.1109788, -0.2177427, 0.2242576
7: 0.5152664, 1.1878800, 0.5543543, 1.2265166, -0.7112502, 0.6335257
8: -0.1307035, 0.1554419, -0.1520776, 0.1485535, -0.2647644, 0.2938832
9: -0.1171892, 0.1396630, -0.1069157, 0.1279713, -0.2451605, 0.2465787

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4395216, upper bound: 0.4470516
time: 1.09 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4434272, upper bound: 0.4520226
time: 1.32 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0933805, 0.0888441, -0.0804270, 0.0838744, -0.1772549, 0.1692712
1: -0.1156907, 0.0929024, -0.1077663, 0.0796108, -0.1953014, 0.2006687
2: -0.0903364, 0.1691786, -0.1231652, 0.1872807, -0.2776171, 0.2923438
3: -0.0956253, 0.2822672, -0.1173119, 0.2487652, -0.3332403, 0.3886244
4: -0.0931082, 0.1074474, -0.0889005, 0.0973040, -0.1904122, 0.1963479
5: -0.0884628, 0.1209092, -0.0759048, 0.1072853, -0.1957481, 0.1968140
6: -0.1067639, 0.1190558, -0.1131017, 0.1203427, -0.2271066, 0.2321575
7: 0.5152664, 1.1878800, 0.5534323, 1.2461121, -0.7308457, 0.6344478
8: -0.1307035, 0.1554419, -0.1641337, 0.1526609, -0.2691849, 0.3063026
9: -0.1171892, 0.1396630, -0.1110217, 0.1336619, -0.2508511, 0.2506846

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4395216, upper bound: 0.4470516
time: 1.64 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4434272, upper bound: 0.4520226
time: 1.98 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0931887, 0.0897850, -0.1636396, 0.1704989
1: -0.1015633, 0.0750528, -0.1159294, 0.0912257, -0.1927890, 0.1909823
2: -0.1146101, 0.1790866, -0.0986205, 0.1718774, -0.2864875, 0.2777071
3: -0.1112695, 0.2397249, -0.1012428, 0.2741829, -0.3743625, 0.3297787
4: -0.0827738, 0.0911205, -0.0942313, 0.1072208, -0.1899946, 0.1853518
5: -0.0691687, 0.1004435, -0.0877777, 0.1211190, -0.1902877, 0.1882212
6: -0.1069577, 0.1124067, -0.1076128, 0.1213651, -0.2283227, 0.2200195
7: 0.5678793, 1.2305298, 0.5254408, 1.2026211, -0.6347418, 0.7050890
8: -0.1547139, 0.1452890, -0.1393657, 0.1567398, -0.2982110, 0.2704070
9: -0.1051957, 0.1236046, -0.1177539, 0.1412229, -0.2464187, 0.2413585

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4500933, upper bound: 0.4373905
time: 1.35 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4378344, upper bound: 0.4365030
time: 1.70 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0763952, 0.0803513, -0.1542059, 0.1537054
1: -0.1015633, 0.0750528, -0.1039884, 0.0772771, -0.1788404, 0.1790413
2: -0.1146101, 0.1790866, -0.1235119, 0.1875489, -0.3021590, 0.3025985
3: -0.1112695, 0.2397249, -0.1176290, 0.2396694, -0.3386157, 0.3449369
4: -0.0827738, 0.0911205, -0.0857780, 0.0940800, -0.1768538, 0.1768985
5: -0.0691687, 0.1004435, -0.0718796, 0.1032298, -0.1723986, 0.1723231
6: -0.1069577, 0.1124067, -0.1130481, 0.1199156, -0.2268732, 0.2254548
7: 0.5678793, 1.2305298, 0.5672836, 1.2453914, -0.6713743, 0.6632462
8: -0.1547139, 0.1452890, -0.1641706, 0.1487945, -0.2880094, 0.2937406
9: -0.1051957, 0.1236046, -0.1105170, 0.1286713, -0.2338670, 0.2341216

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675073, upper bound: 0.4672882
time: 1.17 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4675073, upper bound: 0.4672882
time: 3.12 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0912091, 0.0889064, -0.1627611, 0.1685193
1: -0.1015633, 0.0750528, -0.1144602, 0.0879484, -0.1895117, 0.1895131
2: -0.1146101, 0.1790866, -0.1023752, 0.1713613, -0.2859714, 0.2814618
3: -0.1112695, 0.2397249, -0.1039093, 0.2663953, -0.3665916, 0.3325413
4: -0.0827738, 0.0911205, -0.0934475, 0.1049757, -0.1877494, 0.1845680
5: -0.0691687, 0.1004435, -0.0854578, 0.1189628, -0.1881316, 0.1859013
6: -0.1069577, 0.1124067, -0.1064869, 0.1201712, -0.2271289, 0.2188936
7: 0.5678793, 1.2305298, 0.5347328, 1.2094069, -0.6415276, 0.6957970
8: -0.1547139, 0.1452890, -0.1433238, 0.1560226, -0.2974309, 0.2746592
9: -0.1051957, 0.1236046, -0.1160308, 0.1401076, -0.2453033, 0.2396354

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4473094, upper bound: 0.4334002
time: 1.37 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4349402, upper bound: 0.4324784
time: 1.18 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0759430, 0.0797878, -0.1536424, 0.1532532
1: -0.1015633, 0.0750528, -0.1032179, 0.0770516, -0.1786150, 0.1782708
2: -0.1146101, 0.1790866, -0.1283904, 0.1922093, -0.3068194, 0.3074770
3: -0.1112695, 0.2397249, -0.1211443, 0.2354487, -0.3344164, 0.3485676
4: -0.0827738, 0.0911205, -0.0855687, 0.0936499, -0.1764237, 0.1766892
5: -0.0691687, 0.1004435, -0.0708648, 0.1027419, -0.1719107, 0.1713083
6: -0.1069577, 0.1124067, -0.1161955, 0.1236392, -0.2305969, 0.2286022
7: 0.5678793, 1.2305298, 0.5740471, 1.2531933, -0.6810856, 0.6564826
8: -0.1547139, 0.1452890, -0.1691534, 0.1483116, -0.2875899, 0.2991033
9: -0.1051957, 0.1236046, -0.1129898, 0.1285024, -0.2336981, 0.2365944

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4670966, upper bound: 0.4671675
time: 1.62 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4670966, upper bound: 0.4671675
time: 1.61 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0882377, 0.0859186, -0.0764527, 0.0798426, -0.1680804, 0.1623714
1: -0.1115054, 0.0841621, -0.1040135, 0.0767050, -0.1882104, 0.1881756
2: -0.0940903, 0.1657078, -0.1152034, 0.1798107, -0.2739010, 0.2809112
3: -0.0979359, 0.2645837, -0.1120417, 0.2448600, -0.3317240, 0.3655393
4: -0.0900618, 0.1013495, -0.0849470, 0.0934772, -0.1835390, 0.1862965
5: -0.0823496, 0.1149474, -0.0720161, 0.1027371, -0.1850867, 0.1869636
6: -0.1028849, 0.1141544, -0.1076145, 0.1135762, -0.2164612, 0.2217688
7: 0.5365034, 1.1949518, 0.5594605, 1.2314520, -0.6949486, 0.6354913
8: -0.1343664, 0.1524535, -0.1557337, 0.1480022, -0.2684057, 0.2946583
9: -0.1118737, 0.1352020, -0.1067124, 0.1273035, -0.2391772, 0.2419144

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4333769, upper bound: 0.4446267
time: 1.39 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4322512, upper bound: 0.4322013
time: 1.08 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0733313, 0.0771207, -0.0764527, 0.0798426, -0.1531739, 0.1535734
1: -0.1010609, 0.0750547, -0.1040135, 0.0767050, -0.1777659, 0.1790683
2: -0.1196885, 0.1839125, -0.1152034, 0.1798107, -0.2994992, 0.2991160
3: -0.1149338, 0.2351997, -0.1120417, 0.2448600, -0.3475155, 0.3349388
4: -0.0829049, 0.0910457, -0.0849470, 0.0934772, -0.1763821, 0.1759927
5: -0.0685286, 0.1003120, -0.0720161, 0.1027371, -0.1712657, 0.1723281
6: -0.1102879, 0.1162934, -0.1076145, 0.1135762, -0.2238641, 0.2239079
7: 0.5748564, 1.2385222, 0.5594605, 1.2314520, -0.6559575, 0.6776593
8: -0.1599115, 0.1451830, -0.1557337, 0.1480022, -0.2924412, 0.2850965
9: -0.1078284, 0.1238441, -0.1067124, 0.1273035, -0.2351319, 0.2305565

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4670001
time: 1.14 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4670001
time: 1.11 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0960689, 0.0917158, -0.0933805, 0.0888441, -0.1849130, 0.1850963
1: -0.1184357, 0.0978159, -0.1156907, 0.0929024, -0.2113380, 0.2135065
2: -0.0985660, 0.1746778, -0.0903364, 0.1691786, -0.2677447, 0.2650141
3: -0.1009334, 0.2837050, -0.0956253, 0.2822672, -0.3720801, 0.3682005
4: -0.0967473, 0.1108283, -0.0931082, 0.1074474, -0.2041947, 0.2039365
5: -0.0919634, 0.1246278, -0.0884628, 0.1209092, -0.2128726, 0.2130906
6: -0.1104989, 0.1248560, -0.1067639, 0.1190558, -0.2295547, 0.2316199
7: 0.5135758, 1.2031047, 0.5152664, 1.1878800, -0.6743042, 0.6878383
8: -0.1392483, 0.1596952, -0.1307035, 0.1554419, -0.2802677, 0.2760624
9: -0.1212771, 0.1451643, -0.1171892, 0.1396630, -0.2609400, 0.2623535

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
time: 1.24 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
time: 1.27 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0960689, 0.0917158, -0.0916335, 0.0881791, -0.1842480, 0.1833492
1: -0.1184357, 0.0978159, -0.1145110, 0.0897422, -0.2081779, 0.2123269
2: -0.0985660, 0.1746778, -0.0941827, 0.1689090, -0.2674751, 0.2688605
3: -0.1009334, 0.2837050, -0.0982640, 0.2750747, -0.3648802, 0.3709030
4: -0.0967473, 0.1108283, -0.0924690, 0.1055274, -0.2022747, 0.2032973
5: -0.0919634, 0.1246278, -0.0863134, 0.1190509, -0.2110143, 0.2109412
6: -0.1104989, 0.1248560, -0.1059136, 0.1182695, -0.2287684, 0.2307696
7: 0.5135758, 1.2031047, 0.5239456, 1.1947829, -0.6812071, 0.6791592
8: -0.1392483, 0.1596952, -0.1346984, 0.1548561, -0.2797244, 0.2803120
9: -0.1212771, 0.1451643, -0.1158145, 0.1386088, -0.2598858, 0.2609788

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
time: 1.48 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
time: 1.25 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0786245, 0.0824976, -0.0933805, 0.0888441, -0.1674687, 0.1758782
1: -0.1061348, 0.0787401, -0.1156907, 0.0929024, -0.1990371, 0.1944308
2: -0.1228928, 0.1868956, -0.0903364, 0.1691786, -0.2920715, 0.2772320
3: -0.1169206, 0.2456895, -0.0956253, 0.2822672, -0.3882353, 0.3303201
4: -0.0874815, 0.0960312, -0.0931082, 0.1074474, -0.1949289, 0.1891394
5: -0.0743530, 0.1051416, -0.0884628, 0.1209092, -0.1952622, 0.1936044
6: -0.1127464, 0.1197582, -0.1067639, 0.1190558, -0.2318022, 0.2265221
7: 0.5576000, 1.2456722, 0.5152664, 1.1878800, -0.6302801, 0.7304058
8: -0.1636541, 0.1512056, -0.1307035, 0.1554419, -0.3058310, 0.2680221
9: -0.1105810, 0.1316433, -0.1171892, 0.1396630, -0.2502440, 0.2488325

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4471090, upper bound: 0.4423423
time: 1.77 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4529862, upper bound: 0.4468223
time: 1.21 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0786245, 0.0824976, -0.0916335, 0.0881791, -0.1668036, 0.1741311
1: -0.1061348, 0.0787401, -0.1145110, 0.0897422, -0.1958770, 0.1932511
2: -0.1228928, 0.1868956, -0.0941827, 0.1689090, -0.2918019, 0.2810784
3: -0.1169206, 0.2456895, -0.0982640, 0.2750747, -0.3810354, 0.3330226
4: -0.0874815, 0.0960312, -0.0924690, 0.1055274, -0.1930089, 0.1885002
5: -0.0743530, 0.1051416, -0.0863134, 0.1190509, -0.1934039, 0.1914549
6: -0.1127464, 0.1197582, -0.1059136, 0.1182695, -0.2310160, 0.2256718
7: 0.5576000, 1.2456722, 0.5239456, 1.1947829, -0.6371829, 0.7217267
8: -0.1636541, 0.1512056, -0.1346984, 0.1548561, -0.3052878, 0.2722718
9: -0.1105810, 0.1316433, -0.1158145, 0.1386088, -0.2491898, 0.2474578

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4471090, upper bound: 0.4423423
time: 1.32 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4529862, upper bound: 0.4468223
time: 1.23 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0933672, 0.0902404, -0.0779800, 0.0813531, -0.1747203, 0.1682204
1: -0.1163365, 0.0916511, -0.1055406, 0.0777137, -0.1940502, 0.1971917
2: -0.1012237, 0.1730527, -0.1155450, 0.1802684, -0.2814921, 0.2885977
3: -0.1028749, 0.2742260, -0.1125347, 0.2485853, -0.3403654, 0.3756781
4: -0.0947768, 0.1076471, -0.0862538, 0.0948797, -0.1896566, 0.1939009
5: -0.0880362, 0.1215180, -0.0737139, 0.1040866, -0.1921228, 0.1952319
6: -0.1082956, 0.1225069, -0.1080149, 0.1143332, -0.2226288, 0.2305219
7: 0.5252254, 1.2075171, 0.5533967, 1.2319946, -0.7067692, 0.6541204
8: -0.1420560, 0.1574366, -0.1563694, 0.1496641, -0.2778380, 0.3005592
9: -0.1185009, 0.1420260, -0.1078933, 0.1295430, -0.2480439, 0.2499193

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4701790
time: 1.32 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4702707
time: 1.54 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0765134, 0.0806988, -0.0779800, 0.0813531, -0.1578664, 0.1586788
1: -0.1042710, 0.0776040, -0.1055406, 0.0777137, -0.1819847, 0.1831445
2: -0.1261088, 0.1899618, -0.1155450, 0.1802684, -0.3063772, 0.3055068
3: -0.1193209, 0.2396717, -0.1125347, 0.2485853, -0.3555641, 0.3398884
4: -0.0861943, 0.0944226, -0.0862538, 0.0948797, -0.1810741, 0.1806764
5: -0.0720512, 0.1035180, -0.0737139, 0.1040866, -0.1761377, 0.1772319
6: -0.1147562, 0.1219075, -0.1080149, 0.1143332, -0.2290894, 0.2299224
7: 0.5671665, 1.2500479, 0.5533967, 1.2319946, -0.6648281, 0.6966512
8: -0.1668166, 0.1493322, -0.1563694, 0.1496641, -0.3010705, 0.2902114
9: -0.1119679, 0.1294102, -0.1078933, 0.1295430, -0.2415109, 0.2373036

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4701790
time: 1.79 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4702707
time: 1.46 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.0933805, 0.0888441, -0.1829937, 0.1842557
1: -0.1170394, 0.0935416, -0.1156907, 0.0929024, -0.2099417, 0.2092322
2: -0.1021949, 0.1741340, -0.0903364, 0.1691786, -0.2713735, 0.2644704
3: -0.1034629, 0.2760351, -0.0956253, 0.2822672, -0.3746488, 0.3605248
4: -0.0956366, 0.1086472, -0.0931082, 0.1074474, -0.2030841, 0.2017554
5: -0.0892088, 0.1225371, -0.0884628, 0.1209092, -0.2101180, 0.2109999
6: -0.1092287, 0.1237412, -0.1067639, 0.1190558, -0.2282845, 0.2305051
7: 0.5229961, 1.2097473, 0.5152664, 1.1878800, -0.6648840, 0.6944809
8: -0.1430700, 0.1584855, -0.1307035, 0.1554419, -0.2842072, 0.2747303
9: -0.1196212, 0.1433626, -0.1171892, 0.1396630, -0.2592842, 0.2605518

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
time: 1.87 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
time: 1.63 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.0916335, 0.0881791, -0.1823286, 0.1825086
1: -0.1170394, 0.0935416, -0.1145110, 0.0897422, -0.2067816, 0.2080526
2: -0.1021949, 0.1741340, -0.0941827, 0.1689090, -0.2711039, 0.2683168
3: -0.1034629, 0.2760351, -0.0982640, 0.2750747, -0.3672720, 0.3630604
4: -0.0956366, 0.1086472, -0.0924690, 0.1055274, -0.2011640, 0.2011161
5: -0.0892088, 0.1225371, -0.0863134, 0.1190509, -0.2082597, 0.2088505
6: -0.1092287, 0.1237412, -0.1059136, 0.1182695, -0.2274983, 0.2296548
7: 0.5229961, 1.2097473, 0.5239456, 1.1947829, -0.6717868, 0.6858017
8: -0.1430700, 0.1584855, -0.1346984, 0.1548561, -0.2833355, 0.2787140
9: -0.1196212, 0.1433626, -0.1158145, 0.1386088, -0.2582300, 0.2591771

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
time: 2.21 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
time: 1.62 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0777754, 0.0820198, -0.0933805, 0.0888441, -0.1666196, 0.1754003
1: -0.1054629, 0.0785578, -0.1156907, 0.0929024, -0.1983653, 0.1942484
2: -0.1275831, 0.1913313, -0.0903364, 0.1691786, -0.2967617, 0.2816676
3: -0.1202425, 0.2416290, -0.0956253, 0.2822672, -0.3916293, 0.3263193
4: -0.0873437, 0.0956680, -0.0931082, 0.1074474, -0.1947910, 0.1887763
5: -0.0734341, 0.1047147, -0.0884628, 0.1209092, -0.1943433, 0.1931775
6: -0.1157950, 0.1232789, -0.1067639, 0.1190558, -0.2348507, 0.2300428
7: 0.5640168, 1.2531587, 0.5152664, 1.1878800, -0.6238632, 0.7378923
8: -0.1684172, 0.1508236, -0.1307035, 0.1554419, -0.3108745, 0.2678077
9: -0.1129952, 0.1313578, -0.1171892, 0.1396630, -0.2526582, 0.2485471

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 159

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4473356, upper bound: 0.4423061
time: 1.60 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4533808, upper bound: 0.4468048
time: 1.31 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0777754, 0.0820198, -0.0916335, 0.0881791, -0.1659545, 0.1736532
1: -0.1054629, 0.0785578, -0.1145110, 0.0897422, -0.1952052, 0.1930688
2: -0.1275831, 0.1913313, -0.0941827, 0.1689090, -0.2964921, 0.2855140
3: -0.1202425, 0.2416290, -0.0982640, 0.2750747, -0.3842309, 0.3287972
4: -0.0873437, 0.0956680, -0.0924690, 0.1055274, -0.1928710, 0.1881370
5: -0.0734341, 0.1047147, -0.0863134, 0.1190509, -0.1924850, 0.1910281
6: -0.1157950, 0.1232789, -0.1059136, 0.1182695, -0.2340645, 0.2291925
7: 0.5640168, 1.2531587, 0.5239456, 1.1947829, -0.6307660, 0.7292131
8: -0.1684172, 0.1508236, -0.1346984, 0.1548561, -0.3099271, 0.2716412
9: -0.1129952, 0.1313578, -0.1158145, 0.1386088, -0.2516040, 0.2471724

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4473356, upper bound: 0.4423061
time: 1.46 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4533808, upper bound: 0.4468048
time: 1.44 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0905257, 0.0887121, -0.0738546, 0.0773102, -0.1678359, 0.1625667
1: -0.1140689, 0.0870910, -0.1015633, 0.0750528, -0.1891217, 0.1886544
2: -0.1045961, 0.1715959, -0.1146101, 0.1790866, -0.2836826, 0.2862060
3: -0.1052302, 0.2642792, -0.1112695, 0.2397249, -0.3346884, 0.3645398
4: -0.0933045, 0.1043664, -0.0827738, 0.0911205, -0.1844250, 0.1871402
5: -0.0847376, 0.1183122, -0.0691687, 0.1004435, -0.1851812, 0.1874810
6: -0.1063513, 0.1201283, -0.1069577, 0.1124067, -0.2187580, 0.2270860
7: 0.5371861, 1.2138150, 0.5678793, 1.2305298, -0.6933437, 0.6459357
8: -0.1455738, 0.1560097, -0.1547139, 0.1452890, -0.2783190, 0.2975410
9: -0.1157812, 0.1399214, -0.1051957, 0.1236046, -0.2393858, 0.2451172

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
time: 6.50 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
time: 1.15 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0905257, 0.0887121, -0.0733313, 0.0771207, -0.1676464, 0.1620434
1: -0.1140689, 0.0870910, -0.1010609, 0.0750547, -0.1891236, 0.1881519
2: -0.1045961, 0.1715959, -0.1196885, 0.1839125, -0.2885086, 0.2912844
3: -0.1052302, 0.2642792, -0.1149338, 0.2351997, -0.3300223, 0.3681273
4: -0.0933045, 0.1043664, -0.0829049, 0.0910457, -0.1843502, 0.1872713
5: -0.0847376, 0.1183122, -0.0685286, 0.1003120, -0.1850497, 0.1868409
6: -0.1063513, 0.1201283, -0.1102879, 0.1162934, -0.2226447, 0.2304161
7: 0.5371861, 1.2138150, 0.5748564, 1.2385222, -0.7013361, 0.6389586
8: -0.1455738, 0.1560097, -0.1599115, 0.1451830, -0.2778907, 0.3027632
9: -0.1157812, 0.1399214, -0.1078284, 0.1238441, -0.2396253, 0.2477498

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
time: 1.39 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
time: 1.51 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.0966685, 0.0924504, -0.1866000, 0.1875436
1: -0.1170394, 0.0935416, -0.1192740, 0.0989732, -0.2160126, 0.2128155
2: -0.1021949, 0.1741340, -0.1013751, 0.1762089, -0.2784038, 0.2755092
3: -0.1034629, 0.2760351, -0.1032154, 0.2845033, -0.3758141, 0.3669609
4: -0.0956366, 0.1086472, -0.0977903, 0.1117142, -0.2073508, 0.2064375
5: -0.0892088, 0.1225371, -0.0927949, 0.1255148, -0.2147235, 0.2153320
6: -0.1092287, 0.1237412, -0.1116103, 0.1265226, -0.2357513, 0.2353515
7: 0.5229961, 1.2097473, 0.5124739, 1.2072618, -0.6832944, 0.6951087
8: -0.1430700, 0.1584855, -0.1423288, 0.1607487, -0.2874336, 0.2842447
9: -0.1196212, 0.1433626, -0.1223439, 0.1465364, -0.2661577, 0.2657065

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
time: 1.85 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
time: 3.30 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.0947789, 0.0916212, -0.1857708, 0.1856540
1: -0.1170394, 0.0935416, -0.1178854, 0.0947583, -0.2117977, 0.2114270
2: -0.1021949, 0.1741340, -0.1049845, 0.1757116, -0.2779065, 0.2791186
3: -0.1034629, 0.2760351, -0.1057764, 0.2768768, -0.3679913, 0.3694510
4: -0.0956366, 0.1086472, -0.0966958, 0.1095582, -0.2051948, 0.2053429
5: -0.0892088, 0.1225371, -0.0900759, 0.1234546, -0.2126634, 0.2126130
6: -0.1092287, 0.1237412, -0.1103428, 0.1254216, -0.2346504, 0.2340840
7: 0.5229961, 1.2097473, 0.5218607, 1.2138948, -0.6870569, 0.6859559
8: -0.1430700, 0.1584855, -0.1461802, 0.1595520, -0.2859210, 0.2880391
9: -0.1196212, 0.1433626, -0.1206999, 0.1447951, -0.2644163, 0.2640624

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
time: 1.53 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
time: 1.55 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0777754, 0.0820198, -0.0966685, 0.0924504, -0.1702259, 0.1786883
1: -0.1054629, 0.0785578, -0.1192740, 0.0989732, -0.2044361, 0.1978317
2: -0.1275831, 0.1913313, -0.1013751, 0.1762089, -0.3037920, 0.2927064
3: -0.1202425, 0.2416290, -0.1032154, 0.2845033, -0.3928734, 0.3328421
4: -0.0873437, 0.0956680, -0.0977903, 0.1117142, -0.1990578, 0.1934584
5: -0.0734341, 0.1047147, -0.0927949, 0.1255148, -0.1989489, 0.1975096
6: -0.1157950, 0.1232789, -0.1116103, 0.1265226, -0.2423176, 0.2348892
7: 0.5640168, 1.2531587, 0.5124739, 1.2072618, -0.6432450, 0.7401500
8: -0.1684172, 0.1508236, -0.1423288, 0.1607487, -0.3142674, 0.2774640
9: -0.1129952, 0.1313578, -0.1223439, 0.1465364, -0.2595316, 0.2537018

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 147

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4481262, upper bound: 0.4425916
time: 3.59 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4539521, upper bound: 0.4468406
time: 1.68 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0777754, 0.0820198, -0.0947789, 0.0916212, -0.1693966, 0.1767987
1: -0.1054629, 0.0785578, -0.1178854, 0.0947583, -0.2002212, 0.1964432
2: -0.1275831, 0.1913313, -0.1049845, 0.1757116, -0.3032947, 0.2963158
3: -0.1202425, 0.2416290, -0.1057764, 0.2768768, -0.3850156, 0.3352737
4: -0.0873437, 0.0956680, -0.0966958, 0.1095582, -0.1969018, 0.1923638
5: -0.0734341, 0.1047147, -0.0900759, 0.1234546, -0.1968887, 0.1947905
6: -0.1157950, 0.1232789, -0.1103428, 0.1254216, -0.2412166, 0.2336216
7: 0.5640168, 1.2531587, 0.5218607, 1.2138948, -0.6498780, 0.7304501
8: -0.1684172, 0.1508236, -0.1461802, 0.1595520, -0.3126790, 0.2811105
9: -0.1129952, 0.1313578, -0.1206999, 0.1447951, -0.2577903, 0.2520577

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 68

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4481262, upper bound: 0.4425916
time: 1.71 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4539521, upper bound: 0.4468406
time: 1.44 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0905257, 0.0887121, -0.0766528, 0.0806988, -0.1712245, 0.1653649
1: -0.1140689, 0.0870910, -0.1043003, 0.0776040, -0.1916729, 0.1913913
2: -0.1045961, 0.1715959, -0.1261318, 0.1899618, -0.2945579, 0.2977278
3: -0.1052302, 0.2642792, -0.1193209, 0.2396741, -0.3335519, 0.3715193
4: -0.0933045, 0.1043664, -0.0862360, 0.0944226, -0.1877271, 0.1906023
5: -0.0847376, 0.1183122, -0.0720967, 0.1035180, -0.1882557, 0.1904089
6: -0.1063513, 0.1201283, -0.1147829, 0.1219075, -0.2282588, 0.2349111
7: 0.5371861, 1.2138150, 0.5671665, 1.2501078, -0.7129122, 0.6466485
8: -0.1455738, 0.1560097, -0.1668166, 0.1494556, -0.2804084, 0.3076356
9: -0.1157812, 0.1399214, -0.1119679, 0.1294231, -0.2452043, 0.2518894

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4674207, upper bound: 0.4700176
time: 1.75 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4674207, upper bound: 0.4700176
time: 1.36 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.74 seconds
NS_A1_A1_A1_B2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4503920, upper bound: 0.4503495
NS_A1_A1_A1_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4699189, upper bound: 0.4740602
NS_A1_A1_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4423826, upper bound: 0.4551691
NS_A1_A1_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4699189, upper bound: 0.4740602
NS_A1_A1_A1_B2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4395216, upper bound: 0.4470516
NS_A1_A1_A1_B2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4434272, upper bound: 0.4520226
NS_A1_A1_A1_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4395216, upper bound: 0.4470516
NS_A1_A1_A1_B2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4434272, upper bound: 0.4520226
NS_A1_A1_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4500933, upper bound: 0.4373905
NS_A1_A1_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4378344, upper bound: 0.4365030
NS_A1_A1_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4675073, upper bound: 0.4672882
NS_A1_A1_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4675073, upper bound: 0.4672882
NS_A1_A1_A2_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4473094, upper bound: 0.4334002
NS_A1_A1_A2_B2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4349402, upper bound: 0.4324784
NS_A1_A1_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4670966, upper bound: 0.4671675
NS_A1_A1_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4670966, upper bound: 0.4671675
NS_A1_A2_B2_A2_B2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4333769, upper bound: 0.4446267
NS_A1_A2_B2_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4322512, upper bound: 0.4322013
NS_A1_A2_B2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4670001
NS_A1_A2_B2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4673267, upper bound: 0.4670001
NS_A2_A1_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
NS_A2_A1_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
NS_A2_A1_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
NS_A2_A1_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4782412, upper bound: 0.4778469
NS_A2_A1_B1_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4471090, upper bound: 0.4423423
NS_A2_A1_B1_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4529862, upper bound: 0.4468223
NS_A2_A1_B1_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4471090, upper bound: 0.4423423
NS_A2_A1_B1_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4529862, upper bound: 0.4468223
NS_A2_A1_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4701790
NS_A2_A1_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4702707
NS_A2_A1_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4701790
NS_A2_A1_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4669953, upper bound: 0.4702707
NS_A2_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
NS_A2_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
NS_A2_A2_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
NS_A2_A2_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4784820, upper bound: 0.4777508
NS_A2_A2_B1_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4473356, upper bound: 0.4423061
NS_A2_A2_B1_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4533808, upper bound: 0.4468048
NS_A2_A2_B1_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4473356, upper bound: 0.4423061
NS_A2_A2_B1_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4533808, upper bound: 0.4468048
NS_A2_A2_B1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
NS_A2_A2_B1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
NS_A2_A2_B1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
NS_A2_A2_B1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4672139, upper bound: 0.4700172
NS_A2_A2_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
NS_A2_A2_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
NS_A2_A2_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
NS_A2_A2_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4786174, upper bound: 0.4777615
NS_A2_A2_B2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4481262, upper bound: 0.4425916
NS_A2_A2_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4539521, upper bound: 0.4468406
NS_A2_A2_B2_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4481262, upper bound: 0.4425916
NS_A2_A2_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4539521, upper bound: 0.4468406
NS_A2_A2_B2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4674207, upper bound: 0.4700176
NS_A2_A2_B2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.74
Output dim: 7, lower bound: -0.4674207, upper bound: 0.4700176

## BFS NS instance: NS_A1_A1_A1_B2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0928927, 0.0885000, -0.0901996, 0.0862572, -0.1791499, 0.1786996
1: -0.1152545, 0.0919012, -0.1124400, 0.0876729, -0.2029275, 0.2043412
2: -0.0901425, 0.1686607, -0.0863244, 0.1647755, -0.2549179, 0.2549851
3: -0.0954089, 0.2809903, -0.0918923, 0.2743174, -0.3580238, 0.3616633
4: -0.0926412, 0.1068623, -0.0899744, 0.1034055, -0.1960467, 0.1968367
5: -0.0877885, 0.1203142, -0.0846489, 0.1168422, -0.2046307, 0.2049630
6: -0.1062734, 0.1184174, -0.1031764, 0.1138918, -0.2201652, 0.2215938
7: 0.5168281, 1.1876862, 0.5249248, 1.1825004, -0.6646947, 0.6438198
8: -0.1304619, 0.1549470, -0.1260055, 0.1523693, -0.2668677, 0.2659509
9: -0.1166348, 0.1389717, -0.1130964, 0.1353991, -0.2520339, 0.2520681

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B1_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4423826, upper bound: 0.4554513
time: 1.08 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B1_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4423826, upper bound: 0.4744610
time: 1.44 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0744400, 0.0749974, -0.0889883, 0.0866748, -0.1611148, 0.1639857
1: -0.0985391, 0.0727022, -0.1119624, 0.0850520, -0.1835911, 0.1846646
2: -0.0868062, 0.1511265, -0.0957086, 0.1670357, -0.2538419, 0.2468351
3: -0.0891292, 0.2371198, -0.0976448, 0.2644820, -0.3422355, 0.3238216
4: -0.0784491, 0.0887929, -0.0906323, 0.1021878, -0.1806369, 0.1794252
5: -0.0689120, 0.0987962, -0.0830558, 0.1159537, -0.1848657, 0.1818521
6: -0.0925665, 0.0951651, -0.1035741, 0.1154812, -0.2080477, 0.1987392
7: 0.5730337, 1.1871074, 0.5368385, 1.1999857, -0.6205090, 0.6408434
8: -0.1243238, 0.1408291, -0.1354301, 0.1534496, -0.2621754, 0.2614083
9: -0.0996210, 0.1183813, -0.1130335, 0.1363829, -0.2360039, 0.2314148

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4446884, upper bound: 0.4551691
time: 3.11 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B2_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4446884, upper bound: 0.4551691
time: 1.24 seconds

## BFS NS instance: NS_A1_A1_A1_B2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0887426, 0.0855821, -0.0976646, 0.0927187, -0.1814613, 0.1832467
1: -0.1115392, 0.0855189, -0.1197817, 0.1008068, -0.2123460, 0.2053006
2: -0.0886109, 0.1642486, -0.0988721, 0.1762329, -0.2648439, 0.2631207
3: -0.0936302, 0.2701294, -0.1012050, 0.2873561, -0.3696437, 0.3600076
4: -0.0894293, 0.1018749, -0.0982456, 0.1125338, -0.2019631, 0.2001205
5: -0.0830742, 0.1152516, -0.0939671, 0.1265824, -0.2096566, 0.2092187
6: -0.1024909, 0.1130352, -0.1120316, 0.1266414, -0.2291324, 0.2250668
7: 0.5296254, 1.1860354, 0.5092846, 1.2033811, -0.6674283, 0.6767508
8: -0.1284084, 0.1518393, -0.1395712, 0.1612896, -0.2749181, 0.2762685
9: -0.1119074, 0.1344571, -0.1228652, 0.1473570, -0.2592643, 0.2573223

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B2_A2_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732051, upper bound: 0.4740602
time: 1.30 seconds

## Relational analysis of NS_A1_A1_A1_B2_B1_B1_B2_A2_B2

### Relational analysis result of NS_A1_A1_A1_B2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4732051, upper bound: 0.4740602
time: 1.27 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0738546, 0.0773102, -0.1511648, 0.1511648
1: -0.1015633, 0.0750528, -0.1015633, 0.0750528, -0.1766162, 0.1766162
2: -0.1146101, 0.1790866, -0.1146101, 0.1790866, -0.2936967, 0.2936967
3: -0.1112695, 0.2397249, -0.1112695, 0.2397249, -0.3385642, 0.3385642
4: -0.0827738, 0.0911205, -0.0827738, 0.0911205, -0.1738943, 0.1738943
5: -0.0691687, 0.1004435, -0.0691687, 0.1004435, -0.1696123, 0.1696123
6: -0.1069577, 0.1124067, -0.1069577, 0.1124067, -0.2193644, 0.2193644
7: 0.5678793, 1.2305298, 0.5678793, 1.2305298, -0.6565687, 0.6565687
8: -0.1547139, 0.1452890, -0.1547139, 0.1452890, -0.2840523, 0.2840523
9: -0.1051957, 0.1236046, -0.1051957, 0.1236046, -0.2288003, 0.2288003

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4609201, upper bound: 0.4597274
time: 1.32 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4649642, upper bound: 0.4647307
time: 1.41 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0765134, 0.0806988, -0.1545534, 0.1538236
1: -0.1015633, 0.0750528, -0.1042710, 0.0776040, -0.1791673, 0.1793239
2: -0.1146101, 0.1790866, -0.1261088, 0.1899618, -0.3045719, 0.3051954
3: -0.1112695, 0.2397249, -0.1193209, 0.2396717, -0.3386167, 0.3467192
4: -0.0827738, 0.0911205, -0.0861943, 0.0944226, -0.1771964, 0.1773148
5: -0.0691687, 0.1004435, -0.0720512, 0.1035180, -0.1726868, 0.1724947
6: -0.1069577, 0.1124067, -0.1147562, 0.1219075, -0.2288652, 0.2271629
7: 0.5678793, 1.2305298, 0.5671665, 1.2500479, -0.6801296, 0.6633633
8: -0.1547139, 0.1452890, -0.1668166, 0.1493322, -0.2885819, 0.2966247
9: -0.1051957, 0.1236046, -0.1119679, 0.1294102, -0.2346060, 0.2355725

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4609201, upper bound: 0.4597274
time: 1.34 seconds

## Relational analysis of NS_A1_A1_A2_B2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4649642, upper bound: 0.4647307
time: 1.43 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0733313, 0.0771207, -0.1509753, 0.1506415
1: -0.1015633, 0.0750528, -0.1010609, 0.0750547, -0.1766181, 0.1761138
2: -0.1146101, 0.1790866, -0.1196885, 0.1839125, -0.2985226, 0.2987750
3: -0.1112695, 0.2397249, -0.1149338, 0.2351997, -0.3341170, 0.3423506
4: -0.0827738, 0.0911205, -0.0829049, 0.0910457, -0.1738195, 0.1740254
5: -0.0691687, 0.1004435, -0.0685286, 0.1003120, -0.1694807, 0.1689722
6: -0.1069577, 0.1124067, -0.1102879, 0.1162934, -0.2232511, 0.2226946
7: 0.5678793, 1.2305298, 0.5748564, 1.2385222, -0.6670717, 0.6545362
8: -0.1547139, 0.1452890, -0.1599115, 0.1451830, -0.2840568, 0.2896739
9: -0.1051957, 0.1236046, -0.1078284, 0.1238441, -0.2290398, 0.2314330

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4599858, upper bound: 0.4590136
time: 1.45 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4645439, upper bound: 0.4646428
time: 1.44 seconds

## BFS NS instance: NS_A1_A1_A2_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0738546, 0.0773102, -0.0767534, 0.0801174, -0.1539720, 0.1540636
1: -0.1015633, 0.0750528, -0.1034755, 0.0773535, -0.1789168, 0.1785283
2: -0.1146101, 0.1790866, -0.1308629, 0.1944807, -0.3090908, 0.3099495
3: -0.1112695, 0.2397249, -0.1227197, 0.2354522, -0.3344195, 0.3501539
4: -0.0827738, 0.0911205, -0.0859531, 0.0939702, -0.1767440, 0.1770737
5: -0.0691687, 0.1004435, -0.0710238, 0.1030086, -0.1721773, 0.1714674
6: -0.1069577, 0.1124067, -0.1178055, 0.1255320, -0.2324897, 0.2302122
7: 0.5678793, 1.2305298, 0.5739439, 1.2575608, -0.6885122, 0.6565859
8: -0.1547139, 0.1452890, -0.1716722, 0.1488049, -0.2881234, 0.3017196
9: -0.1051957, 0.1236046, -0.1143432, 0.1292798, -0.2344755, 0.2379479

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4599858, upper bound: 0.4590136
time: 2.84 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4645439, upper bound: 0.4646428
time: 4.45 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0733313, 0.0771207, -0.0738546, 0.0773102, -0.1506415, 0.1509753
1: -0.1010609, 0.0750547, -0.1015633, 0.0750528, -0.1761138, 0.1766181
2: -0.1196885, 0.1839125, -0.1146101, 0.1790866, -0.2987750, 0.2985226
3: -0.1149338, 0.2351997, -0.1112695, 0.2397249, -0.3423506, 0.3341170
4: -0.0829049, 0.0910457, -0.0827738, 0.0911205, -0.1740254, 0.1738195
5: -0.0685286, 0.1003120, -0.0691687, 0.1004435, -0.1689722, 0.1694807
6: -0.1102879, 0.1162934, -0.1069577, 0.1124067, -0.2226946, 0.2232511
7: 0.5748564, 1.2385222, 0.5678793, 1.2305298, -0.6545362, 0.6670717
8: -0.1599115, 0.1451830, -0.1547139, 0.1452890, -0.2896739, 0.2840568
9: -0.1078284, 0.1238441, -0.1051957, 0.1236046, -0.2314330, 0.2290398

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4588076, upper bound: 0.4598275
time: 1.70 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4648010, upper bound: 0.4644453
time: 1.49 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0733313, 0.0771207, -0.0733313, 0.0771207, -0.1504519, 0.1504519
1: -0.1010609, 0.0750547, -0.1010609, 0.0750547, -0.1761156, 0.1761156
2: -0.1196885, 0.1839125, -0.1196885, 0.1839125, -0.3036010, 0.3036010
3: -0.1149338, 0.2351997, -0.1149338, 0.2351997, -0.3376065, 0.3376064
4: -0.0829049, 0.0910457, -0.0829049, 0.0910457, -0.1739506, 0.1739506
5: -0.0685286, 0.1003120, -0.0685286, 0.1003120, -0.1688406, 0.1688406
6: -0.1102879, 0.1162934, -0.1102879, 0.1162934, -0.2265813, 0.2265813
7: 0.5748564, 1.2385222, 0.5748564, 1.2385222, -0.6588382, 0.6588380
8: -0.1599115, 0.1451830, -0.1599115, 0.1451830, -0.2890988, 0.2890988
9: -0.1078284, 0.1238441, -0.1078284, 0.1238441, -0.2316724, 0.2316724

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4588076, upper bound: 0.4598275
time: 1.29 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4648010, upper bound: 0.4644453
time: 1.64 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0960689, 0.0917158, -0.0927781, 0.0880993, -0.1841682, 0.1844938
1: -0.1184357, 0.0978159, -0.1148335, 0.0917476, -0.2101833, 0.2126493
2: -0.0985660, 0.1746778, -0.0875636, 0.1676165, -0.2661825, 0.2622413
3: -0.1009334, 0.2837050, -0.0933850, 0.2814656, -0.3712761, 0.3659773
4: -0.0967473, 0.1108283, -0.0920466, 0.1065542, -0.2033015, 0.2028748
5: -0.0919634, 0.1246278, -0.0876247, 0.1200163, -0.2119797, 0.2122525
6: -0.1104989, 0.1248560, -0.1056239, 0.1173958, -0.2278948, 0.2304798
7: 0.5135758, 1.2031047, 0.5163724, 1.1838391, -0.6702633, 0.6867323
8: -0.1392483, 0.1596952, -0.1276929, 0.1543718, -0.2791533, 0.2729272
9: -0.1212771, 0.1451643, -0.1161135, 0.1382731, -0.2595502, 0.2612778

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4618662, upper bound: 0.4624129
time: 1.91 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4812906, upper bound: 0.4809555
time: 1.47 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0960689, 0.0917158, -0.1021767, 0.0946125, -0.1906814, 0.1938925
1: -0.1184357, 0.0978159, -0.1234617, 0.1127429, -0.2311785, 0.2212776
2: -0.0985660, 0.1746778, -0.0903870, 0.1775085, -0.2760746, 0.2650648
3: -0.1009334, 0.2837050, -0.0967468, 0.3130607, -0.4028599, 0.3694347
4: -0.0967473, 0.1108283, -0.1007553, 0.1184218, -0.2151691, 0.2115836
5: -0.0919634, 0.1246278, -0.1009983, 0.1315683, -0.2235317, 0.2256261
6: -0.1104989, 0.1248560, -0.1154428, 0.1297908, -0.2402898, 0.2402988
7: 0.5135758, 1.2031047, 0.4769261, 1.1875384, -0.6739626, 0.7261786
8: -0.1392483, 0.1596952, -0.1315937, 0.1638805, -0.2890100, 0.2768013
9: -0.1212771, 0.1451643, -0.1276376, 0.1511954, -0.2724725, 0.2728019

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4618662, upper bound: 0.4624129
time: 1.31 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4812906, upper bound: 0.4809555
time: 1.67 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0960689, 0.0917158, -0.0909959, 0.0874200, -0.1834889, 0.1827117
1: -0.1184357, 0.0978159, -0.1136414, 0.0887799, -0.2072156, 0.2114573
2: -0.0985660, 0.1746778, -0.0914362, 0.1673460, -0.2659121, 0.2661139
3: -0.1009334, 0.2837050, -0.0959936, 0.2742180, -0.3640196, 0.3686498
4: -0.0967473, 0.1108283, -0.0914812, 0.1046031, -0.2013504, 0.2023095
5: -0.0919634, 0.1246278, -0.0855570, 0.1181203, -0.2100837, 0.2101848
6: -0.1104989, 0.1248560, -0.1048153, 0.1165666, -0.2270655, 0.2296713
7: 0.5135758, 1.2031047, 0.5250458, 1.1907685, -0.6771927, 0.6780589
8: -0.1392483, 0.1596952, -0.1316625, 0.1539013, -0.2787229, 0.2771288
9: -0.1212771, 0.1451643, -0.1147247, 0.1373469, -0.2586240, 0.2598890

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4437311, upper bound: 0.4310138
time: 1.46 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4776581, upper bound: 0.4772541
time: 2.76 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0960689, 0.0917158, -0.1004303, 0.0939714, -0.1900403, 0.1921460
1: -0.1184357, 0.0978159, -0.1224555, 0.1051617, -0.2235973, 0.2202713
2: -0.0985660, 0.1746778, -0.0940950, 0.1774152, -0.2759812, 0.2687727
3: -0.1009334, 0.2837050, -0.0991734, 0.3066658, -0.3964776, 0.3719096
4: -0.0967473, 0.1108283, -0.0986185, 0.1166569, -0.2134042, 0.2094468
5: -0.0919634, 0.1246278, -0.0968309, 0.1297436, -0.2217070, 0.2214587
6: -0.1104989, 0.1248560, -0.1139853, 0.1291374, -0.2396363, 0.2388413
7: 0.5135758, 1.2031047, 0.4853118, 1.1943352, -0.6807594, 0.7177930
8: -0.1392483, 0.1596952, -0.1355497, 0.1611184, -0.2864004, 0.2810194
9: -0.1212771, 0.1451643, -0.1264681, 0.1473794, -0.2686565, 0.2716324

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4488341, upper bound: 0.4497104
time: 1.69 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4776581, upper bound: 0.4772541
time: 1.33 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0933672, 0.0902404, -0.0738546, 0.0773102, -0.1706775, 0.1640950
1: -0.1163365, 0.0916511, -0.1015633, 0.0750528, -0.1913893, 0.1932145
2: -0.1012237, 0.1730527, -0.1146101, 0.1790866, -0.2803103, 0.2876628
3: -0.1028749, 0.2742260, -0.1112695, 0.2397249, -0.3315156, 0.3744070
4: -0.0947768, 0.1076471, -0.0827738, 0.0911205, -0.1858974, 0.1904208
5: -0.0880362, 0.1215180, -0.0691687, 0.1004435, -0.1884798, 0.1906868
6: -0.1082956, 0.1225069, -0.1069577, 0.1124067, -0.2207023, 0.2294646
7: 0.5252254, 1.2075171, 0.5678793, 1.2305298, -0.7053044, 0.6396379
8: -0.1420560, 0.1574366, -0.1547139, 0.1452890, -0.2733405, 0.2989271
9: -0.1185009, 0.1420260, -0.1051957, 0.1236046, -0.2421055, 0.2472217

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 147

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4633381, upper bound: 0.4682562
time: 1.45 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669837, upper bound: 0.4728373
time: 1.41 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0933672, 0.0902404, -0.0733313, 0.0771207, -0.1704879, 0.1635717
1: -0.1163365, 0.0916511, -0.1010609, 0.0750547, -0.1913912, 0.1927121
2: -0.1012237, 0.1730527, -0.1196885, 0.1839125, -0.2851362, 0.2927412
3: -0.1028749, 0.2742260, -0.1149338, 0.2351997, -0.3271133, 0.3782627
4: -0.0947768, 0.1076471, -0.0829049, 0.0910457, -0.1858225, 0.1905520
5: -0.0880362, 0.1215180, -0.0685286, 0.1003120, -0.1883482, 0.1900467
6: -0.1082956, 0.1225069, -0.1102879, 0.1162934, -0.2245890, 0.2327948
7: 0.5252254, 1.2075171, 0.5748564, 1.2385222, -0.7132968, 0.6326607
8: -0.1420560, 0.1574366, -0.1599115, 0.1451830, -0.2735582, 0.3046832
9: -0.1185009, 0.1420260, -0.1078284, 0.1238441, -0.2423449, 0.2498544

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 147

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 68

### Candidate
type: A, layer: 1, pos: 52

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4633381, upper bound: 0.4682778
time: 1.39 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4669837, upper bound: 0.4730496
time: 1.43 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0765134, 0.0806988, -0.0738546, 0.0773102, -0.1538236, 0.1545534
1: -0.1042710, 0.0776040, -0.1015633, 0.0750528, -0.1793239, 0.1791673
2: -0.1261088, 0.1899618, -0.1146101, 0.1790866, -0.3051954, 0.3045719
3: -0.1193209, 0.2396717, -0.1112695, 0.2397249, -0.3467193, 0.3386167
4: -0.0861943, 0.0944226, -0.0827738, 0.0911205, -0.1773148, 0.1771964
5: -0.0720512, 0.1035180, -0.0691687, 0.1004435, -0.1724947, 0.1726868
6: -0.1147562, 0.1219075, -0.1069577, 0.1124067, -0.2271629, 0.2288652
7: 0.5671665, 1.2500479, 0.5678793, 1.2305298, -0.6633633, 0.6801297
8: -0.1668166, 0.1493322, -0.1547139, 0.1452890, -0.2966248, 0.2885819
9: -0.1119679, 0.1294102, -0.1051957, 0.1236046, -0.2355725, 0.2346060

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4585725, upper bound: 0.4630194
time: 1.34 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4644528, upper bound: 0.4676411
time: 1.54 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0765134, 0.0806988, -0.0733313, 0.0771207, -0.1536341, 0.1540301
1: -0.1042710, 0.0776040, -0.1010609, 0.0750547, -0.1793257, 0.1786649
2: -0.1261088, 0.1899618, -0.1196885, 0.1839125, -0.3100213, 0.3096502
3: -0.1193209, 0.2396717, -0.1149338, 0.2351997, -0.3422721, 0.3424031
4: -0.0861943, 0.0944226, -0.0829049, 0.0910457, -0.1772400, 0.1773275
5: -0.0720512, 0.1035180, -0.0685286, 0.1003120, -0.1723632, 0.1720467
6: -0.1147562, 0.1219075, -0.1102879, 0.1162934, -0.2310496, 0.2321953
7: 0.5671665, 1.2500479, 0.5748564, 1.2385222, -0.6713557, 0.6751915
8: -0.1668166, 0.1493322, -0.1599115, 0.1451830, -0.2966293, 0.2942035
9: -0.1119679, 0.1294102, -0.1078284, 0.1238441, -0.2358120, 0.2372386

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4585725, upper bound: 0.4630194
time: 1.30 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4644528, upper bound: 0.4677718
time: 1.64 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.0927781, 0.0880993, -0.1822489, 0.1836532
1: -0.1170394, 0.0935416, -0.1148335, 0.0917476, -0.2087870, 0.2083750
2: -0.1021949, 0.1741340, -0.0875636, 0.1676165, -0.2698114, 0.2616976
3: -0.1034629, 0.2760351, -0.0933850, 0.2814656, -0.3738449, 0.3583016
4: -0.0956366, 0.1086472, -0.0920466, 0.1065542, -0.2021908, 0.2006937
5: -0.0892088, 0.1225371, -0.0876247, 0.1200163, -0.2092251, 0.2101618
6: -0.1092287, 0.1237412, -0.1056239, 0.1173958, -0.2266246, 0.2293651
7: 0.5229961, 1.2097473, 0.5163724, 1.1838391, -0.6608430, 0.6933749
8: -0.1430700, 0.1584855, -0.1276929, 0.1543718, -0.2830929, 0.2715951
9: -0.1196212, 0.1433626, -0.1161135, 0.1382731, -0.2578943, 0.2594760

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4460003, upper bound: 0.4474592
time: 1.43 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4783633, upper bound: 0.4776239
time: 1.42 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.1021767, 0.0946125, -0.1887620, 0.1930519
1: -0.1170394, 0.0935416, -0.1234617, 0.1127429, -0.2297822, 0.2170033
2: -0.1021949, 0.1741340, -0.0903870, 0.1775085, -0.2797034, 0.2645211
3: -0.1034629, 0.2760351, -0.0967468, 0.3130607, -0.4054288, 0.3617590
4: -0.0956366, 0.1086472, -0.1007553, 0.1184218, -0.2140585, 0.2094025
5: -0.0892088, 0.1225371, -0.1009983, 0.1315683, -0.2207771, 0.2235354
6: -0.1092287, 0.1237412, -0.1154428, 0.1297908, -0.2390196, 0.2391840
7: 0.5229961, 1.2097473, 0.4769261, 1.1875384, -0.6645423, 0.7328212
8: -0.1430700, 0.1584855, -0.1315937, 0.1638805, -0.2929497, 0.2754692
9: -0.1196212, 0.1433626, -0.1276376, 0.1511954, -0.2708167, 0.2710001

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4460003, upper bound: 0.4474592
time: 1.48 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4783633, upper bound: 0.4776239
time: 1.31 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.0909959, 0.0874200, -0.1815695, 0.1818710
1: -0.1170394, 0.0935416, -0.1136414, 0.0887799, -0.2058193, 0.2071830
2: -0.1021949, 0.1741340, -0.0914362, 0.1673460, -0.2695409, 0.2655702
3: -0.1034629, 0.2760351, -0.0959936, 0.2742180, -0.3664116, 0.3608081
4: -0.0956366, 0.1086472, -0.0914812, 0.1046031, -0.2002397, 0.2001284
5: -0.0892088, 0.1225371, -0.0855570, 0.1181203, -0.2073291, 0.2080941
6: -0.1092287, 0.1237412, -0.1048153, 0.1165666, -0.2257954, 0.2285565
7: 0.5229961, 1.2097473, 0.5250458, 1.1907685, -0.6677724, 0.6847015
8: -0.1430700, 0.1584855, -0.1316625, 0.1539013, -0.2823350, 0.2755306
9: -0.1196212, 0.1433626, -0.1147247, 0.1373469, -0.2569681, 0.2580872

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4446799, upper bound: 0.4465184
time: 1.33 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4778931, upper bound: 0.4771651
time: 5.19 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0941496, 0.0908751, -0.1004303, 0.0939714, -0.1881209, 0.1913054
1: -0.1170394, 0.0935416, -0.1224555, 0.1051617, -0.2222010, 0.2159970
2: -0.1021949, 0.1741340, -0.0940950, 0.1774152, -0.2796101, 0.2682290
3: -0.1034629, 0.2760351, -0.0991734, 0.3066658, -0.3988644, 0.3640668
4: -0.0956366, 0.1086472, -0.0986185, 0.1166569, -0.2122935, 0.2072656
5: -0.0892088, 0.1225371, -0.0968309, 0.1297436, -0.2189524, 0.2193680
6: -0.1092287, 0.1237412, -0.1139853, 0.1291374, -0.2383662, 0.2377265
7: 0.5229961, 1.2097473, 0.4853118, 1.1943352, -0.6713392, 0.7244356
8: -0.1430700, 0.1584855, -0.1355497, 0.1611184, -0.2899854, 0.2794091
9: -0.1196212, 0.1433626, -0.1264681, 0.1473794, -0.2670006, 0.2698307

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 196

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 196

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4446799, upper bound: 0.4465184
time: 1.64 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4778931, upper bound: 0.4771651
time: 1.95 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0913760, 0.0893271, -0.0738546, 0.0773102, -0.1686862, 0.1631817
1: -0.1148317, 0.0883436, -0.1015633, 0.0750528, -0.1898846, 0.1899069
2: -0.1048530, 0.1724762, -0.1146101, 0.1790866, -0.2839395, 0.2870863
3: -0.1054905, 0.2664358, -0.1112695, 0.2397249, -0.3341868, 0.3666328
4: -0.0939565, 0.1053667, -0.0827738, 0.0911205, -0.1850770, 0.1881405
5: -0.0856952, 0.1193326, -0.0691687, 0.1004435, -0.1861387, 0.1885013
6: -0.1071092, 0.1212395, -0.1069577, 0.1124067, -0.2195159, 0.2281971
7: 0.5345422, 1.2141119, 0.5678793, 1.2305298, -0.6959876, 0.6462327
8: -0.1459057, 0.1566664, -0.1547139, 0.1452890, -0.2773731, 0.2981001
9: -0.1167279, 0.1408582, -0.1051957, 0.1236046, -0.2403325, 0.2460539

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 69

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4334986, upper bound: 0.4473388
time: 1.35 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4325030, upper bound: 0.4348224
time: 1.54 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0767534, 0.0801174, -0.0738546, 0.0773102, -0.1540636, 0.1539720
1: -0.1034755, 0.0773535, -0.1015633, 0.0750528, -0.1785283, 0.1789168
2: -0.1308629, 0.1944807, -0.1146101, 0.1790866, -0.3099495, 0.3090908
3: -0.1227197, 0.2354522, -0.1112695, 0.2397249, -0.3501539, 0.3344195
4: -0.0859531, 0.0939702, -0.0827738, 0.0911205, -0.1770737, 0.1767440
5: -0.0710238, 0.1030086, -0.0691687, 0.1004435, -0.1714674, 0.1721773
6: -0.1178055, 0.1255320, -0.1069577, 0.1124067, -0.2302122, 0.2324897
7: 0.5739439, 1.2575608, 0.5678793, 1.2305298, -0.6565859, 0.6885121
8: -0.1716722, 0.1488049, -0.1547139, 0.1452890, -0.3017196, 0.2881235
9: -0.1143432, 0.1292798, -0.1051957, 0.1236046, -0.2379479, 0.2344755

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 52

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4591034, upper bound: 0.4629688
time: 1.69 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4647027, upper bound: 0.4675005
time: 1.31 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0913760, 0.0893271, -0.0733313, 0.0771207, -0.1684967, 0.1626583
1: -0.1148317, 0.0883436, -0.1010609, 0.0750547, -0.1898864, 0.1894045
2: -0.1048530, 0.1724762, -0.1196885, 0.1839125, -0.2887655, 0.2921646
3: -0.1054905, 0.2664358, -0.1149338, 0.2351997, -0.3295607, 0.3702342
4: -0.0939565, 0.1053667, -0.0829049, 0.0910457, -0.1850022, 0.1882716
5: -0.0856952, 0.1193326, -0.0685286, 0.1003120, -0.1860072, 0.1878612
6: -0.1071092, 0.1212395, -0.1102879, 0.1162934, -0.2234026, 0.2315273
7: 0.5345422, 1.2141119, 0.5748564, 1.2385222, -0.7039800, 0.6392555
8: -0.1459057, 0.1566664, -0.1599115, 0.1451830, -0.2771060, 0.3033666
9: -0.1167279, 0.1408582, -0.1078284, 0.1238441, -0.2405720, 0.2486866

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 213

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 86

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4333873, upper bound: 0.4471938
time: 1.17 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.4322512, upper bound: 0.4337401
time: 1.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0767534, 0.0801174, -0.0733313, 0.0771207, -0.1538741, 0.1534486
1: -0.1034755, 0.0773535, -0.1010609, 0.0750547, -0.1785302, 0.1784144
2: -0.1308629, 0.1944807, -0.1196885, 0.1839125, -0.3147754, 0.3141692
3: -0.1227197, 0.2354522, -0.1149338, 0.2351997, -0.3454767, 0.3379366
4: -0.0859531, 0.0939702, -0.0829049, 0.0910457, -0.1769988, 0.1768751
5: -0.0710238, 0.1030086, -0.0685286, 0.1003120, -0.1713358, 0.1715372
6: -0.1178055, 0.1255320, -0.1102879, 0.1162934, -0.2340989, 0.2358199
7: 0.5739439, 1.2575608, 0.5748564, 1.2385222, -0.6645783, 0.6819004
8: -0.1716722, 0.1488049, -0.1599115, 0.1451830, -0.3012603, 0.2933149
9: -0.1143432, 0.1292798, -0.1078284, 0.1238441, -0.2381873, 0.2371081

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 69

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 86

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.14 + 595.99 = 600.13 seconds
