## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06674455


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1392890, 0.1122234, -0.1392890, 0.1122234, -0.2515125, 0.2515125)
1: (-0.0842915, 0.0526762, -0.0842915, 0.0526762, -0.1369677, 0.1369677)
2: (-0.1213749, 0.1386176, -0.1213749, 0.1386176, -0.2599924, 0.2599924)
3: (0.9367827, 1.0235401, 0.9367827, 1.0235401, -0.0867574, 0.0867574)
4: (-0.0926538, 0.1204631, -0.0926538, 0.1204631, -0.2131169, 0.2131169)
5: (-0.0723801, 0.1693253, -0.0723801, 0.1693253, -0.2417054, 0.2417054)
6: (-0.1307721, 0.1265659, -0.1307721, 0.1265659, -0.2573380, 0.2573380)
7: (-0.1173507, 0.0593616, -0.1173507, 0.0593616, -0.1767123, 0.1767123)
8: (-0.0587243, 0.1434318, -0.0587243, 0.1434318, -0.2021561, 0.2021561)
9: (-0.1124728, 0.0843993, -0.1124728, 0.0843993, -0.1968721, 0.1968721)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 3.10 = 4.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0785230, upper bound: 0.0785227

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772780, upper bound: 0.0780093
time: 2.41 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0785101, upper bound: 0.0785101
time: 2.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.60
Output dim: 3, lower bound: -0.0772780, upper bound: 0.0780093
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.60
Output dim: 3, lower bound: -0.0785101, upper bound: 0.0785101

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.1487151, 0.1203682, -0.1385171, 0.1116192, -0.2603344, 0.2588852
1: -0.0909111, 0.0578570, -0.0838459, 0.0522411, -0.1431521, 0.1417030
2: -0.1302593, 0.1486376, -0.1206159, 0.1378590, -0.2681184, 0.2692536
3: 0.9284079, 1.0267935, 0.9377324, 1.0233098, -0.0949019, 0.0890611
4: -0.1005238, 0.1291509, -0.0920401, 0.1198054, -0.2203292, 0.2211910
5: -0.0791567, 0.1788217, -0.0718116, 0.1685486, -0.2477053, 0.2506332
6: -0.1398455, 0.1353667, -0.1300416, 0.1258154, -0.2656609, 0.2654083
7: -0.1240009, 0.0692202, -0.1167974, 0.0586849, -0.1826858, 0.1860176
8: -0.0658315, 0.1538345, -0.0583177, 0.1425551, -0.2083865, 0.2121522
9: -0.1203924, 0.0891905, -0.1118558, 0.0839195, -0.2043119, 0.2010463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0761942, upper bound: 0.0761829
time: 1.38 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772774, upper bound: 0.0779986
time: 2.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.1386419, 0.1117204, -0.1392266, 0.1121748, -0.2508166, 0.2509470
1: -0.0839203, 0.0523124, -0.0842557, 0.0526412, -0.1365615, 0.1365680
2: -0.1207412, 0.1379796, -0.1213138, 0.1385558, -0.2592971, 0.2592934
3: 0.9375139, 1.0233499, 0.9368535, 1.0235215, -0.0860076, 0.0864964
4: -0.0921365, 0.1199132, -0.0926040, 0.1204100, -0.2125465, 0.2125171
5: -0.0719050, 0.1686847, -0.0723342, 0.1692634, -0.2411684, 0.2410190
6: -0.1301615, 0.1259439, -0.1307131, 0.1265059, -0.2566675, 0.2566571
7: -0.1168876, 0.0588036, -0.1173060, 0.0593076, -0.1761952, 0.1761095
8: -0.0583850, 0.1427064, -0.0586913, 0.1433618, -0.2017468, 0.2013977
9: -0.1119547, 0.0840181, -0.1124228, 0.0843624, -0.1963171, 0.1964409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0780099, upper bound: 0.0772780
time: 1.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0780099, upper bound: 0.0785097
time: 2.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 3, lower bound: -0.0761942, upper bound: 0.0761829
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 3, lower bound: -0.0772774, upper bound: 0.0779986
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 3, lower bound: -0.0780099, upper bound: 0.0772780
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.31
Output dim: 3, lower bound: -0.0780099, upper bound: 0.0785097

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.1425323, 0.1152681, -0.1347946, 0.1096616, -0.2521940, 0.2500628
1: -0.0865882, 0.0543855, -0.0820140, 0.0499177, -0.1365059, 0.1363995
2: -0.1240531, 0.1426166, -0.1158215, 0.1358266, -0.2598798, 0.2584381
3: 0.9328135, 1.0242066, 0.9349787, 1.0209588, -0.0881453, 0.0892279
4: -0.0959850, 0.1231474, -0.0908148, 0.1158576, -0.2118427, 0.2139622
5: -0.0745250, 0.1724660, -0.0685010, 0.1649848, -0.2395098, 0.2409669
6: -0.1336394, 0.1294706, -0.1257978, 0.1221362, -0.2557756, 0.2552683
7: -0.1194990, 0.0635058, -0.1137297, 0.0582870, -0.1777861, 0.1772355
8: -0.0608670, 0.1472855, -0.0557394, 0.1395800, -0.2004469, 0.2030249
9: -0.1154097, 0.0859432, -0.1095007, 0.0822321, -0.1976417, 0.1954439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0615442, upper bound: 0.0599693
time: 2.48 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0707847, upper bound: 0.0696166
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685682, upper bound: 0.0689175
time: 1.22 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.1487151, 0.1203682, -0.1372785, 0.1107122, -0.2594274, 0.2576467
1: -0.0909111, 0.0578570, -0.0831549, 0.0515428, -0.1424538, 0.1410119
2: -0.1302593, 0.1486376, -0.1193657, 0.1367172, -0.2669765, 0.2680033
3: 0.9284079, 1.0267935, 0.9386145, 1.0228939, -0.0944860, 0.0881790
4: -0.1005238, 0.1291509, -0.0911364, 0.1187203, -0.2192441, 0.2202873
5: -0.0791567, 0.1788217, -0.0708817, 0.1673537, -0.2465104, 0.2497034
6: -0.1398455, 0.1353667, -0.1288503, 0.1246310, -0.2644766, 0.2642170
7: -0.1240009, 0.0692202, -0.1158943, 0.0577836, -0.1817845, 0.1851145
8: -0.0658315, 0.1538345, -0.0576456, 0.1412435, -0.2070749, 0.2114801
9: -0.1203924, 0.0891905, -0.1108965, 0.0832627, -0.2036550, 0.2000870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747791, upper bound: 0.0762314
time: 2.18 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747792, upper bound: 0.0779992
time: 1.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.1386419, 0.1117204, -0.1487151, 0.1195610, -0.2582029, 0.2604356
1: -0.0839203, 0.0523124, -0.0896572, 0.0578570, -0.1417773, 0.1419696
2: -0.1207412, 0.1379796, -0.1302593, 0.1482204, -0.2689616, 0.2682390
3: 0.9375139, 1.0233499, 0.9284079, 1.0260173, -0.0885034, 0.0949420
4: -0.0921365, 0.1199132, -0.1005238, 0.1282905, -0.2204271, 0.2204370
5: -0.0719050, 0.1686847, -0.0791567, 0.1782320, -0.2501370, 0.2478414
6: -0.1301615, 0.1259439, -0.1394402, 0.1353667, -0.2655282, 0.2653841
7: -0.1168876, 0.0588036, -0.1240009, 0.0674872, -0.1843749, 0.1828045
8: -0.0583850, 0.1427064, -0.0634984, 0.1538345, -0.2122194, 0.2062048
9: -0.1119547, 0.0840181, -0.1201246, 0.0891905, -0.2011451, 0.2041427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0761829, upper bound: 0.0761937
time: 1.56 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0772774
time: 1.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.1386419, 0.1117204, -0.1386419, 0.1117204, -0.2503623, 0.2503623
1: -0.0839203, 0.0523124, -0.0839203, 0.0523124, -0.1362327, 0.1362327
2: -0.1207412, 0.1379796, -0.1207412, 0.1379796, -0.2587209, 0.2587209
3: 0.9375139, 1.0233499, 0.9375139, 1.0233499, -0.0858359, 0.0858359
4: -0.0921365, 0.1199132, -0.0921365, 0.1199132, -0.2120497, 0.2120497
5: -0.0719050, 0.1686847, -0.0719050, 0.1686847, -0.2405897, 0.2405897
6: -0.1301615, 0.1259439, -0.1301615, 0.1259439, -0.2561055, 0.2561055
7: -0.1168876, 0.0588036, -0.1168876, 0.0588036, -0.1756912, 0.1756912
8: -0.0583850, 0.1427064, -0.0583850, 0.1427064, -0.2010914, 0.2010914
9: -0.1119547, 0.0840181, -0.1119547, 0.0840181, -0.1959728, 0.1959728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0761829, upper bound: 0.0777572
time: 1.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0785040
time: 2.00 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.07 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0707847, upper bound: 0.0696166
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0685682, upper bound: 0.0689175
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0747791, upper bound: 0.0762314
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0747792, upper bound: 0.0779992
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0761829, upper bound: 0.0761937
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0772774
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0761829, upper bound: 0.0777572
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0785040

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1296683, 0.1057695, -0.1334683, 0.1087036, -0.2383720, 0.2392377
1: -0.0791132, 0.0470811, -0.0812834, 0.0491624, -0.1282756, 0.1283645
2: -0.1109134, 0.1307764, -0.1144659, 0.1346161, -0.2455294, 0.2452423
3: 0.9394633, 1.0195211, 0.9356879, 1.0204923, -0.0810291, 0.0838332
4: -0.0867272, 0.1115343, -0.0898613, 0.1146849, -0.2014121, 0.2013956
5: -0.0648016, 0.1600701, -0.0674898, 0.1637280, -0.2285296, 0.2275599
6: -0.1210437, 0.1173086, -0.1245081, 0.1208764, -0.2419201, 0.2418167
7: -0.1100783, 0.0540894, -0.1127585, 0.0573714, -0.1674498, 0.1668479
8: -0.0531128, 0.1339940, -0.0550050, 0.1382054, -0.1913182, 0.1889990
9: -0.1054104, 0.0795408, -0.1084770, 0.0815673, -0.1869776, 0.1880177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0693282, upper bound: 0.0663967
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0693282, upper bound: 0.0663963
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1150171, 0.0952384, -0.1216838, 0.1000637, -0.2150808, 0.2169223
1: -0.0710951, 0.0390063, -0.0747453, 0.0425520, -0.1136471, 0.1137517
2: -0.0965928, 0.1168816, -0.1027358, 0.1235136, -0.2201064, 0.2196174
3: 0.9407060, 1.0151528, 0.9418691, 1.0167489, -0.0760429, 0.0732837
4: -0.0756035, 0.0989477, -0.0809913, 0.1044741, -0.1800776, 0.1799390
5: -0.0541156, 0.1467552, -0.0587437, 0.1526173, -0.2067329, 0.2054989
6: -0.1072916, 0.1039675, -0.1132836, 0.1098333, -0.2171249, 0.2172511
7: -0.0994909, 0.0440233, -0.1042209, 0.0487639, -0.1482548, 0.1482441
8: -0.0454852, 0.1191415, -0.0487197, 0.1258685, -0.1713537, 0.1678612
9: -0.0939244, 0.0734464, -0.0992518, 0.0758140, -0.1697384, 0.1726982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0663960
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0663959
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1454258, 0.1183378, -0.1372785, 0.1107122, -0.2561380, 0.2556162
1: -0.0887180, 0.0557632, -0.0831549, 0.0515428, -0.1402608, 0.1389181
2: -0.1258563, 0.1468174, -0.1193657, 0.1367172, -0.2625735, 0.2661831
3: 0.9253336, 1.0241864, 0.9386145, 1.0228939, -0.0975603, 0.0855719
4: -0.0996217, 0.1251453, -0.0911364, 0.1187203, -0.2183420, 0.2162818
5: -0.0761432, 0.1753820, -0.0708817, 0.1673537, -0.2434969, 0.2462637
6: -0.1357860, 0.1320996, -0.1288503, 0.1246310, -0.2604170, 0.2609499
7: -0.1212357, 0.0682992, -0.1158943, 0.0577836, -0.1790192, 0.1841935
8: -0.0623444, 0.1513208, -0.0576456, 0.1412435, -0.2035878, 0.2089664
9: -0.1182432, 0.0877123, -0.1108965, 0.0832627, -0.2015059, 0.1986088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0759189
time: 1.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0762314
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1475226, 0.1193831, -0.1372785, 0.1107122, -0.2582348, 0.2566615
1: -0.0900766, 0.0571861, -0.0831549, 0.0515428, -0.1416194, 0.1403410
2: -0.1290567, 0.1474789, -0.1193657, 0.1367172, -0.2657738, 0.2668446
3: 0.9292635, 1.0262902, 0.9386145, 1.0228939, -0.0936304, 0.0876757
4: -0.0996514, 0.1279899, -0.0911364, 0.1187203, -0.2183717, 0.2191263
5: -0.0782609, 0.1775912, -0.0708817, 0.1673537, -0.2456146, 0.2484729
6: -0.1386461, 0.1342273, -0.1288503, 0.1246310, -0.2632771, 0.2630776
7: -0.1231311, 0.0681176, -0.1158943, 0.0577836, -0.1809147, 0.1840120
8: -0.0648706, 0.1525695, -0.0576456, 0.1412435, -0.2061141, 0.2102152
9: -0.1194324, 0.0885569, -0.1108965, 0.0832627, -0.2026950, 0.1994534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747178
time: 1.64 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0757923
time: 1.71 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1348894, 0.1097408, -0.1425323, 0.1150233, -0.2499126, 0.2522731
1: -0.0820720, 0.0499711, -0.0862079, 0.0543855, -0.1364575, 0.1361790
2: -0.1159154, 0.1359190, -0.1240531, 0.1424901, -0.2584055, 0.2599722
3: 0.9347730, 1.0209875, 0.9328135, 1.0239716, -0.0891986, 0.0881740
4: -0.0908889, 0.1159386, -0.0959850, 0.1228864, -0.2137753, 0.2119237
5: -0.0685707, 0.1650929, -0.0745250, 0.1722872, -0.2408579, 0.2396180
6: -0.1258874, 0.1222362, -0.1335165, 0.1294706, -0.2553580, 0.2557527
7: -0.1137977, 0.0583848, -0.1194990, 0.0629801, -0.1767779, 0.1778838
8: -0.0557892, 0.1397002, -0.0601594, 0.1472855, -0.2030747, 0.1998597
9: -0.1095756, 0.0823160, -0.1153285, 0.0859432, -0.1955189, 0.1976445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0599688, upper bound: 0.0615449
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0696166, upper bound: 0.0707847
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689174, upper bound: 0.0685683
time: 1.70 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1374049, 0.1108150, -0.1487151, 0.1195610, -0.2569660, 0.2595301
1: -0.0832302, 0.0516150, -0.0896572, 0.0578570, -0.1410872, 0.1412722
2: -0.1194925, 0.1368398, -0.1302593, 0.1482204, -0.2677129, 0.2670991
3: 0.9383941, 1.0229342, 0.9284079, 1.0260173, -0.0876232, 0.0945263
4: -0.0912343, 0.1188298, -0.1005238, 0.1282905, -0.2195248, 0.2193536
5: -0.0709766, 0.1674915, -0.0791567, 0.1782320, -0.2492085, 0.2466482
6: -0.1289717, 0.1247611, -0.1394402, 0.1353667, -0.2643384, 0.2642013
7: -0.1159858, 0.0579038, -0.1240009, 0.0674872, -0.1834730, 0.1819047
8: -0.0577139, 0.1413969, -0.0634984, 0.1538345, -0.2115484, 0.2048953
9: -0.1109969, 0.0833621, -0.1201246, 0.0891905, -0.2001874, 0.2034867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0747790
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0772774
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1348894, 0.1097408, -0.1323732, 0.1071173, -0.2420066, 0.2421140
1: -0.0820720, 0.0499711, -0.0804212, 0.0487821, -0.1308542, 0.1303923
2: -0.1159154, 0.1359190, -0.1144225, 0.1321742, -0.2480896, 0.2503416
3: 0.9347730, 1.0209875, 0.9419975, 1.0212650, -0.0864921, 0.0789900
4: -0.0908889, 0.1159386, -0.0875320, 0.1144306, -0.2053196, 0.2034707
5: -0.0685707, 0.1650929, -0.0672050, 0.1626458, -0.2312165, 0.2322980
6: -0.1258874, 0.1222362, -0.1241387, 0.1199657, -0.2458531, 0.2463749
7: -0.1137977, 0.0583848, -0.1123206, 0.0542152, -0.1680130, 0.1707053
8: -0.0557892, 0.1397002, -0.0549940, 0.1360651, -0.1918543, 0.1946943
9: -0.1095756, 0.0823160, -0.1070914, 0.0807080, -0.1902837, 0.1894075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0652898, upper bound: 0.0692646
time: 1.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0652893, upper bound: 0.0666031
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1374049, 0.1108150, -0.1386419, 0.1117204, -0.2491253, 0.2494569
1: -0.0832302, 0.0516150, -0.0839203, 0.0523124, -0.1355425, 0.1355354
2: -0.1194925, 0.1368398, -0.1207412, 0.1379796, -0.2574721, 0.2575811
3: 0.9383941, 1.0229342, 0.9375139, 1.0233499, -0.0849558, 0.0854203
4: -0.0912343, 0.1188298, -0.0921365, 0.1199132, -0.2111475, 0.2109663
5: -0.0709766, 0.1674915, -0.0719050, 0.1686847, -0.2396613, 0.2393965
6: -0.1289717, 0.1247611, -0.1301615, 0.1259439, -0.2549157, 0.2549226
7: -0.1159858, 0.0579038, -0.1168876, 0.0588036, -0.1747893, 0.1747914
8: -0.0577139, 0.1413969, -0.0583850, 0.1427064, -0.2004204, 0.1997818
9: -0.1109969, 0.0833621, -0.1119547, 0.0840181, -0.1950151, 0.1953167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0777383, upper bound: 0.0770551
time: 1.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0777383, upper bound: 0.0785040
time: 2.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.38 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0693282, upper bound: 0.0663967
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0693282, upper bound: 0.0663963
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0663960
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0663959
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0759189
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0762314
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747178
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0757923
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0696166, upper bound: 0.0707847
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0689174, upper bound: 0.0685683
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0747790
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0772774
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0652898, upper bound: 0.0692646
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0652893, upper bound: 0.0666031
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0777383, upper bound: 0.0770551
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.38
Output dim: 3, lower bound: -0.0777383, upper bound: 0.0785040

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1296683, 0.1057695, -0.1440577, 0.1169359, -0.2466042, 0.2498271
1: -0.0791132, 0.0470811, -0.0873132, 0.0549871, -0.1341003, 0.1343944
2: -0.1109134, 0.1307764, -0.1244562, 0.1453586, -0.2562720, 0.2552326
3: 0.9394633, 1.0195211, 0.9260395, 1.0233092, -0.0838460, 0.0934817
4: -0.0867272, 0.1115343, -0.0986425, 0.1234904, -0.2102176, 0.2101769
5: -0.0648016, 0.1600701, -0.0751096, 0.1737765, -0.2385782, 0.2351796
6: -0.1210437, 0.1173086, -0.1342486, 0.1308073, -0.2518510, 0.2515572
7: -0.1100783, 0.0540894, -0.1202333, 0.0664602, -0.1765385, 0.1743228
8: -0.0531128, 0.1339940, -0.0603825, 0.1499104, -0.2030233, 0.1943765
9: -0.1054104, 0.0795408, -0.1170500, 0.0870342, -0.1924445, 0.1965908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663938
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663973
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1296683, 0.1057695, -0.1335625, 0.1087823, -0.2384506, 0.2393320
1: -0.0791132, 0.0470811, -0.0813411, 0.0492157, -0.1283289, 0.1284222
2: -0.1109134, 0.1307764, -0.1145594, 0.1347081, -0.2456214, 0.2453358
3: 0.9394633, 1.0195211, 0.9354841, 1.0205209, -0.0810577, 0.0840371
4: -0.0867272, 0.1115343, -0.0899353, 0.1147656, -0.2014928, 0.2014696
5: -0.0648016, 0.1600701, -0.0675593, 0.1638356, -0.2286372, 0.2276293
6: -0.1210437, 0.1173086, -0.1245976, 0.1209759, -0.2420196, 0.2419062
7: -0.1100783, 0.0540894, -0.1128262, 0.0574686, -0.1675470, 0.1669157
8: -0.0531128, 0.1339940, -0.0550547, 0.1383252, -0.1914380, 0.1890486
9: -0.1054104, 0.0795408, -0.1085517, 0.0816507, -0.1870611, 0.1880925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0689979
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663972
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1150171, 0.0952384, -0.1327627, 0.1086677, -0.2236848, 0.2280011
1: -0.0710951, 0.0390063, -0.0810552, 0.0486160, -0.1197112, 0.1200615
2: -0.0965928, 0.1168816, -0.1131572, 0.1347550, -0.2313478, 0.2300387
3: 0.9407060, 1.0151528, 0.9318945, 1.0196424, -0.0789363, 0.0832583
4: -0.0756035, 0.0989477, -0.0901711, 0.1136833, -0.1892868, 0.1891188
5: -0.0541156, 0.1467552, -0.0666704, 0.1631496, -0.2172652, 0.2134256
6: -0.1072916, 0.1039675, -0.1234362, 0.1202070, -0.2274987, 0.2274037
7: -0.0994909, 0.0440233, -0.1120494, 0.0582496, -0.1577404, 0.1560726
8: -0.0454852, 0.1191415, -0.0543013, 0.1381115, -0.1835967, 0.1734429
9: -0.0939244, 0.0734464, -0.1082130, 0.0815379, -0.1754624, 0.1816594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0663923
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0663963
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1150171, 0.0952384, -0.1217775, 0.1001411, -0.2151582, 0.2170159
1: -0.0710951, 0.0390063, -0.0748023, 0.0426049, -0.1137000, 0.1138086
2: -0.0965928, 0.1168816, -0.1028283, 0.1236047, -0.2201975, 0.2197099
3: 0.9407060, 1.0151528, 0.9416775, 1.0167776, -0.0760716, 0.0734753
4: -0.0756035, 0.0989477, -0.0810646, 0.1045541, -0.1801576, 0.1800122
5: -0.0541156, 0.1467552, -0.0588128, 0.1527228, -0.2068383, 0.2055680
6: -0.1072916, 0.1039675, -0.1133725, 0.1099313, -0.2172230, 0.2173400
7: -0.0994909, 0.0440233, -0.1042879, 0.0488593, -0.1483501, 0.1483112
8: -0.0454852, 0.1191415, -0.0487691, 0.1259860, -0.1714712, 0.1679106
9: -0.0939244, 0.0734464, -0.0993260, 0.0758941, -0.1698186, 0.1727724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0680074
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0689174
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1454258, 0.1183378, -0.1475226, 0.1186848, -0.2641106, 0.2658603
1: -0.0887180, 0.0557632, -0.0889918, 0.0571861, -0.1459042, 0.1447550
2: -0.1258563, 0.1468174, -0.1290567, 0.1471179, -0.2729743, 0.2758741
3: 0.9253336, 1.0241864, 0.9292635, 1.0256188, -0.1002852, 0.0949229
4: -0.0996217, 0.1251453, -0.0996514, 0.1272454, -0.2268670, 0.2247967
5: -0.0761432, 0.1753820, -0.0782609, 0.1770809, -0.2532241, 0.2536429
6: -0.1357860, 0.1320996, -0.1382954, 0.1342273, -0.2700132, 0.2703949
7: -0.1212357, 0.0682992, -0.1231311, 0.0666182, -0.1878538, 0.1914303
8: -0.0623444, 0.1513208, -0.0628522, 0.1525695, -0.2149139, 0.2141730
9: -0.1182432, 0.0877123, -0.1192007, 0.0885569, -0.2068001, 0.2069130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0702823, upper bound: 0.0687015
time: 1.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663958, upper bound: 0.0678793
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1454258, 0.1183378, -0.1374049, 0.1108150, -0.2562408, 0.2557427
1: -0.0887180, 0.0557632, -0.0832302, 0.0516150, -0.1403331, 0.1389934
2: -0.1258563, 0.1468174, -0.1194925, 0.1368398, -0.2626961, 0.2663099
3: 0.9253336, 1.0241864, 0.9383941, 1.0229342, -0.0976006, 0.0857922
4: -0.0996217, 0.1251453, -0.0912343, 0.1188298, -0.2184514, 0.2163796
5: -0.0761432, 0.1753820, -0.0709766, 0.1674915, -0.2436347, 0.2463585
6: -0.1357860, 0.1320996, -0.1289717, 0.1247611, -0.2605470, 0.2610713
7: -0.1212357, 0.0682992, -0.1159858, 0.0579038, -0.1791394, 0.1842850
8: -0.0623444, 0.1513208, -0.0577139, 0.1413969, -0.2037412, 0.2090347
9: -0.1182432, 0.0877123, -0.1109969, 0.0833621, -0.2016053, 0.1987092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0702823, upper bound: 0.0698919
time: 1.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663958, upper bound: 0.0678801
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1475226, 0.1193831, -0.1475226, 0.1186848, -0.2662074, 0.2669057
1: -0.0900766, 0.0571861, -0.0889918, 0.0571861, -0.1472628, 0.1461779
2: -0.1290567, 0.1474789, -0.1290567, 0.1471179, -0.2761746, 0.2765356
3: 0.9292635, 1.0262902, 0.9292635, 1.0256188, -0.0963553, 0.0970267
4: -0.0996514, 0.1279899, -0.0996514, 0.1272454, -0.2268967, 0.2276413
5: -0.0782609, 0.1775912, -0.0782609, 0.1770809, -0.2553419, 0.2558521
6: -0.1386461, 0.1342273, -0.1382954, 0.1342273, -0.2728733, 0.2725226
7: -0.1231311, 0.0681176, -0.1231311, 0.0666182, -0.1897493, 0.1912488
8: -0.0648706, 0.1525695, -0.0628522, 0.1525695, -0.2174401, 0.2154218
9: -0.1194324, 0.0885569, -0.1192007, 0.0885569, -0.2079893, 0.2077576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671808, upper bound: 0.0630836
time: 1.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0630822, upper bound: 0.0630833
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1475226, 0.1193831, -0.1374049, 0.1108150, -0.2583376, 0.2567880
1: -0.0900766, 0.0571861, -0.0832302, 0.0516150, -0.1416917, 0.1404163
2: -0.1290567, 0.1474789, -0.1194925, 0.1368398, -0.2658965, 0.2669714
3: 0.9292635, 1.0262902, 0.9383941, 1.0229342, -0.0936707, 0.0878960
4: -0.0996514, 0.1279899, -0.0912343, 0.1188298, -0.2184811, 0.2192242
5: -0.0782609, 0.1775912, -0.0709766, 0.1674915, -0.2457525, 0.2485677
6: -0.1386461, 0.1342273, -0.1289717, 0.1247611, -0.2634071, 0.2631990
7: -0.1231311, 0.0681176, -0.1159858, 0.0579038, -0.1810349, 0.1841034
8: -0.0648706, 0.1525695, -0.0577139, 0.1413969, -0.2062675, 0.2102835
9: -0.1194324, 0.0885569, -0.1109969, 0.0833621, -0.2027944, 0.1995538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0671808, upper bound: 0.0666006
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0630822, upper bound: 0.0651802
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1335625, 0.1087823, -0.1296683, 0.1057695, -0.2393320, 0.2384506
1: -0.0813411, 0.0492157, -0.0791132, 0.0470811, -0.1284222, 0.1283289
2: -0.1145594, 0.1347081, -0.1109134, 0.1307764, -0.2453358, 0.2456214
3: 0.9354841, 1.0205209, 0.9394633, 1.0195211, -0.0840371, 0.0810577
4: -0.0899353, 0.1147656, -0.0867272, 0.1115343, -0.2014696, 0.2014928
5: -0.0675593, 0.1638356, -0.0648016, 0.1600701, -0.2276293, 0.2286372
6: -0.1245976, 0.1209759, -0.1210437, 0.1173086, -0.2419062, 0.2420196
7: -0.1128262, 0.0574686, -0.1100783, 0.0540894, -0.1669157, 0.1675470
8: -0.0550547, 0.1383252, -0.0531128, 0.1339940, -0.1890486, 0.1914380
9: -0.1085517, 0.0816507, -0.1054104, 0.0795408, -0.1880925, 0.1870611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689171, upper bound: 0.0685678
time: 1.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689171, upper bound: 0.0685684
time: 1.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1217775, 0.1001411, -0.1150171, 0.0952384, -0.2170159, 0.2151582
1: -0.0748023, 0.0426049, -0.0710951, 0.0390063, -0.1138086, 0.1137000
2: -0.1028283, 0.1236047, -0.0965928, 0.1168816, -0.2197099, 0.2201975
3: 0.9416775, 1.0167776, 0.9407060, 1.0151528, -0.0734753, 0.0760716
4: -0.0810646, 0.1045541, -0.0756035, 0.0989477, -0.1800122, 0.1801576
5: -0.0588128, 0.1527228, -0.0541156, 0.1467552, -0.2055680, 0.2068383
6: -0.1133725, 0.1099313, -0.1072916, 0.1039675, -0.2173400, 0.2172230
7: -0.1042879, 0.0488593, -0.0994909, 0.0440233, -0.1483112, 0.1483501
8: -0.0487691, 0.1259860, -0.0454852, 0.1191415, -0.1679106, 0.1714712
9: -0.0993260, 0.0758941, -0.0939244, 0.0734464, -0.1727724, 0.1698186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689175, upper bound: 0.0685680
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689175, upper bound: 0.0685684
time: 1.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1374049, 0.1108150, -0.1454258, 0.1179186, -0.2553235, 0.2562408
1: -0.0832302, 0.0516150, -0.0880667, 0.0557632, -0.1389934, 0.1396818
2: -0.1194925, 0.1368398, -0.1258563, 0.1466007, -0.2660932, 0.2626961
3: 0.9383941, 1.0229342, 0.9253336, 1.0237833, -0.0853892, 0.0976006
4: -0.0912343, 0.1188298, -0.0996217, 0.1246984, -0.2159327, 0.2184514
5: -0.0709766, 0.1674915, -0.0761432, 0.1750756, -0.2460522, 0.2436347
6: -0.1289717, 0.1247611, -0.1355755, 0.1320996, -0.2610713, 0.2603365
7: -0.1159858, 0.0579038, -0.1212357, 0.0673971, -0.1833829, 0.1791394
8: -0.0577139, 0.1413969, -0.0611325, 0.1513208, -0.2090347, 0.2025294
9: -0.1109969, 0.0833621, -0.1181041, 0.0877123, -0.1987092, 0.2014662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0695470, upper bound: 0.0663989
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0663980
time: 2.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1374049, 0.1108150, -0.1475226, 0.1186848, -0.2560897, 0.2583376
1: -0.0832302, 0.0516150, -0.0889918, 0.0571861, -0.1404163, 0.1406069
2: -0.1194925, 0.1368398, -0.1290567, 0.1471179, -0.2666104, 0.2658965
3: 0.9383941, 1.0229342, 0.9292635, 1.0256188, -0.0872247, 0.0936707
4: -0.0912343, 0.1188298, -0.0996514, 0.1272454, -0.2184796, 0.2184811
5: -0.0709766, 0.1674915, -0.0782609, 0.1770809, -0.2480575, 0.2457525
6: -0.1289717, 0.1247611, -0.1382954, 0.1342273, -0.2631990, 0.2630565
7: -0.1159858, 0.0579038, -0.1231311, 0.0666182, -0.1826040, 0.1810349
8: -0.0577139, 0.1413969, -0.0628522, 0.1525695, -0.2102835, 0.2042491
9: -0.1109969, 0.0833621, -0.1192007, 0.0885569, -0.1995538, 0.2025628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0695470, upper bound: 0.0707998
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0708001
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1348894, 0.1097408, -0.1288828, 0.1044720, -0.2393614, 0.2386236
1: -0.0820720, 0.0499711, -0.0784544, 0.0468313, -0.1289033, 0.1284255
2: -0.1159154, 0.1359190, -0.1110144, 0.1287712, -0.2446866, 0.2469334
3: 0.9347730, 1.0209875, 0.9447684, 1.0202233, -0.0854503, 0.0762191
4: -0.0908889, 0.1159386, -0.0847806, 0.1114569, -0.2023459, 0.2007193
5: -0.0685707, 0.1650929, -0.0646391, 0.1593156, -0.2278862, 0.2297321
6: -0.1258874, 0.1222362, -0.1208508, 0.1166744, -0.2425618, 0.2430871
7: -0.1137977, 0.0583848, -0.1098217, 0.0514191, -0.1652168, 0.1682065
8: -0.0557892, 0.1397002, -0.0531634, 0.1322886, -0.1880779, 0.1928636
9: -0.1095756, 0.0823160, -0.1043152, 0.0789002, -0.1884759, 0.1866312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0652889, upper bound: 0.0666031
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0652889, upper bound: 0.0666029
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1374049, 0.1108150, -0.1348894, 0.1097408, -0.2471457, 0.2457044
1: -0.0832302, 0.0516150, -0.0820720, 0.0499711, -0.1332013, 0.1336871
2: -0.1194925, 0.1368398, -0.1159154, 0.1359190, -0.2554116, 0.2527552
3: 0.9383941, 1.0229342, 0.9347730, 1.0209875, -0.0825934, 0.0881612
4: -0.0912343, 0.1188298, -0.0908889, 0.1159386, -0.2071729, 0.2097187
5: -0.0709766, 0.1674915, -0.0685707, 0.1650929, -0.2360695, 0.2360622
6: -0.1289717, 0.1247611, -0.1258874, 0.1222362, -0.2512079, 0.2506485
7: -0.1159858, 0.0579038, -0.1137977, 0.0583848, -0.1743706, 0.1717015
8: -0.0577139, 0.1413969, -0.0557892, 0.1397002, -0.1974142, 0.1971861
9: -0.1109969, 0.0833621, -0.1095756, 0.0823160, -0.1933130, 0.1929377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0691372, upper bound: 0.0652897
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0666000, upper bound: 0.0652896
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1374049, 0.1108150, -0.1374049, 0.1108150, -0.2482199, 0.2482199
1: -0.0832302, 0.0516150, -0.0832302, 0.0516150, -0.1348452, 0.1348452
2: -0.1194925, 0.1368398, -0.1194925, 0.1368398, -0.2563323, 0.2563323
3: 0.9383941, 1.0229342, 0.9383941, 1.0229342, -0.0845401, 0.0845401
4: -0.0912343, 0.1188298, -0.0912343, 0.1188298, -0.2100640, 0.2100640
5: -0.0709766, 0.1674915, -0.0709766, 0.1674915, -0.2384681, 0.2384681
6: -0.1289717, 0.1247611, -0.1289717, 0.1247611, -0.2537328, 0.2537328
7: -0.1159858, 0.0579038, -0.1159858, 0.0579038, -0.1738896, 0.1738896
8: -0.0577139, 0.1413969, -0.0577139, 0.1413969, -0.1991108, 0.1991108
9: -0.1109969, 0.0833621, -0.1109969, 0.0833621, -0.1943590, 0.1943590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0691372, upper bound: 0.0688556
time: 1.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0666000, upper bound: 0.0688550
time: 1.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.21 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663938
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663973
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0689979
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663972
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0663923
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0663963
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0680074
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0689174
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0702823, upper bound: 0.0687015
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0663958, upper bound: 0.0678793
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0702823, upper bound: 0.0698919
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0663958, upper bound: 0.0678801
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0671808, upper bound: 0.0630836
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0630822, upper bound: 0.0630833
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0671808, upper bound: 0.0666006
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0630822, upper bound: 0.0651802
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0689171, upper bound: 0.0685678
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0689171, upper bound: 0.0685684
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0689175, upper bound: 0.0685680
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0689175, upper bound: 0.0685684
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0695470, upper bound: 0.0663989
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0663980
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0695470, upper bound: 0.0707998
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0708001
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0652889, upper bound: 0.0666031
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0652889, upper bound: 0.0666029
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0691372, upper bound: 0.0652897
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0666000, upper bound: 0.0652896
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0691372, upper bound: 0.0688556
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -0.0666000, upper bound: 0.0688550

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1440577, 0.1169359, -0.2495253, 0.2527323
1: -0.0809906, 0.0484630, -0.0873132, 0.0549871, -0.1359777, 0.1357763
2: -0.1127485, 0.1349086, -0.1244562, 0.1453586, -0.2581071, 0.2593647
3: 0.9321295, 1.0193111, 0.9260395, 1.0233092, -0.0911797, 0.0932716
4: -0.0903902, 0.1133753, -0.0986425, 0.1234904, -0.2138805, 0.2120179
5: -0.0664112, 0.1629083, -0.0751096, 0.1737765, -0.2401877, 0.2380178
6: -0.1231250, 0.1199399, -0.1342486, 0.1308073, -0.2539323, 0.2541885
7: -0.1118449, 0.0585164, -0.1202333, 0.0664602, -0.1783051, 0.1787498
8: -0.0540693, 0.1380443, -0.0603825, 0.1499104, -0.2039797, 0.1984268
9: -0.1081993, 0.0813312, -0.1170500, 0.0870342, -0.1952335, 0.1983812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663932
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663930
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1346416, 0.1094295, -0.1440577, 0.1169359, -0.2515775, 0.2534872
1: -0.0818898, 0.0498854, -0.0873132, 0.0549871, -0.1368769, 0.1371987
2: -0.1159472, 0.1353870, -0.1244562, 0.1453586, -0.2613058, 0.2598432
3: 0.9358367, 1.0211815, 0.9260395, 1.0233092, -0.0874726, 0.0951420
4: -0.0903837, 0.1158983, -0.0986425, 0.1234904, -0.2138741, 0.2145408
5: -0.0685374, 0.1648742, -0.0751096, 0.1737765, -0.2423139, 0.2399838
6: -0.1258359, 0.1220468, -0.1342486, 0.1308073, -0.2566432, 0.2562953
7: -0.1137101, 0.0577350, -0.1202333, 0.0664602, -0.1801703, 0.1779684
8: -0.0558092, 0.1392670, -0.0603825, 0.1499104, -0.2057196, 0.1996495
9: -0.1092722, 0.0821861, -0.1170500, 0.0870342, -0.1963064, 0.1992361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663970
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663971
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1335625, 0.1087823, -0.2413716, 0.2422372
1: -0.0809906, 0.0484630, -0.0813411, 0.0492157, -0.1302063, 0.1298041
2: -0.1127485, 0.1349086, -0.1145594, 0.1347081, -0.2474566, 0.2494680
3: 0.9321295, 1.0193111, 0.9354841, 1.0205209, -0.0883914, 0.0838270
4: -0.0903902, 0.1133753, -0.0899353, 0.1147656, -0.2051558, 0.2033106
5: -0.0664112, 0.1629083, -0.0675593, 0.1638356, -0.2302468, 0.2304675
6: -0.1231250, 0.1199399, -0.1245976, 0.1209759, -0.2441009, 0.2445375
7: -0.1118449, 0.0585164, -0.1128262, 0.0574686, -0.1693136, 0.1713426
8: -0.0540693, 0.1380443, -0.0550547, 0.1383252, -0.1923945, 0.1930989
9: -0.1081993, 0.0813312, -0.1085517, 0.0816507, -0.1898501, 0.1898829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0689979
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0689979
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1346416, 0.1094295, -0.1335625, 0.1087823, -0.2434238, 0.2429921
1: -0.0818898, 0.0498854, -0.0813411, 0.0492157, -0.1311055, 0.1312265
2: -0.1159472, 0.1353870, -0.1145594, 0.1347081, -0.2506553, 0.2499464
3: 0.9358367, 1.0211815, 0.9354841, 1.0205209, -0.0846843, 0.0856974
4: -0.0903837, 0.1158983, -0.0899353, 0.1147656, -0.2051493, 0.2058336
5: -0.0685374, 0.1648742, -0.0675593, 0.1638356, -0.2323730, 0.2324335
6: -0.1258359, 0.1220468, -0.1245976, 0.1209759, -0.2468118, 0.2466443
7: -0.1137101, 0.0577350, -0.1128262, 0.0574686, -0.1711788, 0.1705613
8: -0.0558092, 0.1392670, -0.0550547, 0.1383252, -0.1941344, 0.1943217
9: -0.1092722, 0.0821861, -0.1085517, 0.0816507, -0.1909230, 0.1907378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0696166
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0696166
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1183204, 0.0983966, -0.1217775, 0.1001411, -0.2184615, 0.2201741
1: -0.0731852, 0.0406024, -0.0748023, 0.0426049, -0.1157901, 0.1154047
2: -0.0988272, 0.1213201, -0.1028283, 0.1236047, -0.2224319, 0.2241484
3: 0.9330631, 1.0150847, 0.9416775, 1.0167776, -0.0837145, 0.0734072
4: -0.0794985, 0.1011350, -0.0810646, 0.1045541, -0.1840526, 0.1821996
5: -0.0560150, 0.1499757, -0.0588128, 0.1527228, -0.2087378, 0.2087885
6: -0.1097481, 0.1069696, -0.1133725, 0.1099313, -0.2196795, 0.2203421
7: -0.1015449, 0.0487259, -0.1042879, 0.0488593, -0.1504042, 0.1530138
8: -0.0466571, 0.1235784, -0.0487691, 0.1259860, -0.1726431, 0.1723475
9: -0.0969927, 0.0754595, -0.0993260, 0.0758941, -0.1728868, 0.1747855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680077
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1198855, 0.0988301, -0.1217775, 0.1001411, -0.2200266, 0.2206075
1: -0.0738123, 0.0417495, -0.0748023, 0.0426049, -0.1164172, 0.1165518
2: -0.1015046, 0.1214103, -0.1028283, 0.1236047, -0.2251092, 0.2242386
3: 0.9371079, 1.0167693, 0.9416775, 1.0167776, -0.0796697, 0.0750918
4: -0.0791954, 0.1032070, -0.0810646, 0.1045541, -0.1837495, 0.1842716
5: -0.0577697, 0.1514509, -0.0588128, 0.1527228, -0.2104924, 0.2102637
6: -0.1119747, 0.1086060, -0.1133725, 0.1099313, -0.2219060, 0.2219784
7: -0.1030404, 0.0475872, -0.1042879, 0.0488593, -0.1518996, 0.1518751
8: -0.0481198, 0.1243111, -0.0487691, 0.1259860, -0.1741058, 0.1730801
9: -0.0977124, 0.0760327, -0.0993260, 0.0758941, -0.1736065, 0.1753587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0689169
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0689171
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1461586, 0.1177056, -0.2502949, 0.2548332
1: -0.0809906, 0.0484630, -0.0882413, 0.0564147, -0.1374053, 0.1367043
2: -0.1127485, 0.1349086, -0.1276770, 0.1458737, -0.2586222, 0.2625856
3: 0.9321295, 1.0193111, 0.9299443, 1.0251508, -0.0930213, 0.0893668
4: -0.0903902, 0.1133753, -0.0986722, 0.1260465, -0.2164367, 0.2120475
5: -0.0664112, 0.1629083, -0.0772323, 0.1757931, -0.2422042, 0.2401406
6: -0.1231250, 0.1199399, -0.1369816, 0.1329358, -0.2560608, 0.2569215
7: -0.1118449, 0.0585164, -0.1221348, 0.0656845, -0.1775294, 0.1806512
8: -0.0540693, 0.1380443, -0.0621061, 0.1511623, -0.2052316, 0.2001504
9: -0.1081993, 0.0813312, -0.1181491, 0.0878878, -0.1960872, 0.1994802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663957, upper bound: 0.0678798
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663957, upper bound: 0.0678798
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1183204, 0.0983966, -0.1346938, 0.1093504, -0.2276708, 0.2330904
1: -0.0731852, 0.0406024, -0.0818882, 0.0499607, -0.1231460, 0.1224905
2: -0.0988272, 0.1213201, -0.1161586, 0.1351828, -0.2340101, 0.2374787
3: 0.9330631, 1.0150847, 0.9356807, 1.0214248, -0.0883617, 0.0794040
4: -0.0794985, 0.1011350, -0.0901374, 0.1160593, -0.1955578, 0.1912724
5: -0.0560150, 0.1499757, -0.0686690, 0.1649854, -0.2210005, 0.2186448
6: -0.1097481, 0.1069696, -0.1259882, 0.1221877, -0.2319358, 0.2329579
7: -0.1015449, 0.0487259, -0.1137995, 0.0574316, -0.1589766, 0.1625254
8: -0.0466571, 0.1235784, -0.0559404, 0.1392239, -0.1858809, 0.1795188
9: -0.0969927, 0.0754595, -0.1092096, 0.0823179, -0.1793106, 0.1846691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1360702, 0.1098533, -0.2424427, 0.2447448
1: -0.0809906, 0.0484630, -0.0824939, 0.0508540, -0.1318446, 0.1309570
2: -0.1127485, 0.1349086, -0.1181299, 0.1356232, -0.2483717, 0.2530385
3: 0.9321295, 1.0193111, 0.9390885, 1.0224689, -0.0903394, 0.0802226
4: -0.0903902, 0.1133753, -0.0902714, 0.1176550, -0.2080452, 0.2036467
5: -0.0664112, 0.1629083, -0.0699640, 0.1662279, -0.2326390, 0.2328722
6: -0.1231250, 0.1199399, -0.1276777, 0.1234960, -0.2466210, 0.2476176
7: -0.1118449, 0.0585164, -0.1150118, 0.0569734, -0.1688184, 0.1735282
8: -0.0540693, 0.1380443, -0.0569791, 0.1400152, -0.1940845, 0.1950234
9: -0.1081993, 0.0813312, -0.1099660, 0.0826985, -0.1908978, 0.1912972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663978, upper bound: 0.0685738
time: 1.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663978, upper bound: 0.0685734
time: 1.48 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1183204, 0.0983966, -0.1242423, 0.1012090, -0.2195294, 0.2226389
1: -0.0731852, 0.0406024, -0.0759393, 0.0442138, -0.1173990, 0.1165416
2: -0.0988272, 0.1213201, -0.1063311, 0.1245155, -0.2233427, 0.2276512
3: 0.9330631, 1.0150847, 0.9451058, 1.0187007, -0.0856376, 0.0699790
4: -0.0794985, 0.1011350, -0.0814063, 0.1073877, -0.1868862, 0.1825413
5: -0.0560150, 0.1499757, -0.0611840, 0.1550652, -0.2110802, 0.2111597
6: -0.1097481, 0.1069696, -0.1163867, 0.1124274, -0.2221756, 0.2233563
7: -0.1015449, 0.0487259, -0.1064289, 0.0484080, -0.1499529, 0.1551548
8: -0.0466571, 0.1235784, -0.0506632, 0.1276713, -0.1743283, 0.1742416
9: -0.0969927, 0.0754595, -0.1007180, 0.0769367, -0.1739294, 0.1761775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663980, upper bound: 0.0685739
time: 1.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663980, upper bound: 0.0685734
time: 1.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1440384, 0.1164261, -0.1475226, 0.1186848, -0.2627233, 0.2639486
1: -0.0876346, 0.0552295, -0.0889918, 0.0571861, -0.1448207, 0.1442213
2: -0.1256445, 0.1439024, -0.1290567, 0.1471179, -0.2727624, 0.2729591
3: 0.9320320, 1.0249517, 0.9292635, 1.0256188, -0.0935867, 0.0956882
4: -0.0968789, 0.1246959, -0.0996514, 0.1272454, -0.2241242, 0.2243473
5: -0.0756937, 0.1740491, -0.0782609, 0.1770809, -0.2527747, 0.2523101
6: -0.1351987, 0.1309509, -0.1382954, 0.1342273, -0.2694260, 0.2692462
7: -0.1206374, 0.0646334, -0.1231311, 0.0666182, -0.1872556, 0.1877645
8: -0.0621500, 0.1487966, -0.0628522, 0.1525695, -0.2147196, 0.2116489
9: -0.1165453, 0.0867621, -0.1192007, 0.0885569, -0.2051022, 0.2059628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0630821, upper bound: 0.0630832
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0630821, upper bound: 0.0630835
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1440384, 0.1164261, -0.1374049, 0.1108150, -0.2548534, 0.2538310
1: -0.0876346, 0.0552295, -0.0832302, 0.0516150, -0.1392497, 0.1384597
2: -0.1256445, 0.1439024, -0.1194925, 0.1368398, -0.2624843, 0.2633949
3: 0.9320320, 1.0249517, 0.9383941, 1.0229342, -0.0909021, 0.0865576
4: -0.0968789, 0.1246959, -0.0912343, 0.1188298, -0.2157086, 0.2159302
5: -0.0756937, 0.1740491, -0.0709766, 0.1674915, -0.2431853, 0.2450257
6: -0.1351987, 0.1309509, -0.1289717, 0.1247611, -0.2599598, 0.2599226
7: -0.1206374, 0.0646334, -0.1159858, 0.0579038, -0.1785412, 0.1806192
8: -0.0621500, 0.1487966, -0.0577139, 0.1413969, -0.2035469, 0.2065106
9: -0.1165453, 0.0867621, -0.1109969, 0.0833621, -0.1999074, 0.1977590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0630844, upper bound: 0.0651806
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0630844, upper bound: 0.0651804
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1224673, 0.1007717, -0.1296683, 0.1057695, -0.2282367, 0.2304400
1: -0.0752174, 0.0428783, -0.0791132, 0.0470811, -0.1222986, 0.1219915
2: -0.1031720, 0.1246065, -0.1109134, 0.1307764, -0.2339485, 0.2355199
3: 0.9415862, 1.0166223, 0.9394633, 1.0195211, -0.0779349, 0.0771590
4: -0.0819490, 0.1049587, -0.0867272, 0.1115343, -0.1934833, 0.1916859
5: -0.0591207, 0.1532771, -0.0648016, 0.1600701, -0.2191907, 0.2180787
6: -0.1137879, 0.1104516, -0.1210437, 0.1173086, -0.2310966, 0.2314953
7: -0.1046996, 0.0497274, -0.1100783, 0.0540894, -0.1587891, 0.1598058
8: -0.0489237, 0.1268218, -0.0531128, 0.1339940, -0.1829177, 0.1799346
9: -0.0999834, 0.0760543, -0.1054104, 0.0795408, -0.1795241, 0.1814646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700890
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700886
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1070720, 0.0897472, -0.1296683, 0.1057695, -0.2128414, 0.2194155
1: -0.0667697, 0.0345466, -0.0791132, 0.0470811, -0.1138509, 0.1136598
2: -0.0884381, 0.1098502, -0.1109134, 0.1307764, -0.2192145, 0.2207636
3: 0.9429960, 1.0122710, 0.9394633, 1.0195211, -0.0765251, 0.0728078
4: -0.0700960, 0.0918550, -0.0867272, 0.1115343, -0.1816303, 0.1785822
5: -0.0480115, 0.1393077, -0.0648016, 0.1600701, -0.2080816, 0.2041093
6: -0.0995537, 0.0965336, -0.1210437, 0.1173086, -0.2168624, 0.2175773
7: -0.0936450, 0.0390376, -0.1100783, 0.0540894, -0.1477345, 0.1491159
8: -0.0412567, 0.1111409, -0.0531128, 0.1339940, -0.1752507, 0.1642537
9: -0.0879000, 0.0697001, -0.1054104, 0.0795408, -0.1674408, 0.1751105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700890
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700888
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1224673, 0.1007717, -0.1150171, 0.0952384, -0.2177057, 0.2157889
1: -0.0752174, 0.0428783, -0.0710951, 0.0390063, -0.1142238, 0.1139735
2: -0.1031720, 0.1246065, -0.0965928, 0.1168816, -0.2200536, 0.2211993
3: 0.9415862, 1.0166223, 0.9407060, 1.0151528, -0.0735666, 0.0759163
4: -0.0819490, 0.1049587, -0.0756035, 0.0989477, -0.1808967, 0.1805622
5: -0.0591207, 0.1532771, -0.0541156, 0.1467552, -0.2058759, 0.2073926
6: -0.1137879, 0.1104516, -0.1072916, 0.1039675, -0.2177554, 0.2177433
7: -0.1046996, 0.0497274, -0.0994909, 0.0440233, -0.1487229, 0.1492183
8: -0.0489237, 0.1268218, -0.0454852, 0.1191415, -0.1680653, 0.1723070
9: -0.0999834, 0.0760543, -0.0939244, 0.0734464, -0.1734298, 0.1699787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663941
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0685683
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1070720, 0.0897472, -0.1150171, 0.0952384, -0.2023104, 0.2047643
1: -0.0667697, 0.0345466, -0.0710951, 0.0390063, -0.1057761, 0.1056417
2: -0.0884381, 0.1098502, -0.0965928, 0.1168816, -0.2053197, 0.2064430
3: 0.9429960, 1.0122710, 0.9407060, 1.0151528, -0.0721568, 0.0715650
4: -0.0700960, 0.0918550, -0.0756035, 0.0989477, -0.1690437, 0.1674585
5: -0.0480115, 0.1393077, -0.0541156, 0.1467552, -0.1947668, 0.1934232
6: -0.0995537, 0.0965336, -0.1072916, 0.1039675, -0.2035213, 0.2038253
7: -0.0936450, 0.0390376, -0.0994909, 0.0440233, -0.1376683, 0.1385285
8: -0.0412567, 0.1111409, -0.0454852, 0.1191415, -0.1603983, 0.1566261
9: -0.0879000, 0.0697001, -0.0939244, 0.0734464, -0.1613464, 0.1636246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663941
time: 1.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0685683
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1249191, 0.1018048, -0.1440577, 0.1169359, -0.2418550, 0.2458625
1: -0.0763374, 0.0444955, -0.0873132, 0.0549871, -0.1313244, 0.1318088
2: -0.1067018, 0.1254580, -0.1244562, 0.1453586, -0.2520604, 0.2499141
3: 0.9449932, 1.0185899, 0.9260395, 1.0233092, -0.0783160, 0.0925504
4: -0.0822238, 0.1078117, -0.0986425, 0.1234904, -0.2057141, 0.2064543
5: -0.0615130, 0.1556173, -0.0751096, 0.1737765, -0.2352896, 0.2307269
6: -0.1168360, 0.1129417, -0.1342486, 0.1308073, -0.2476433, 0.2471903
7: -0.1068466, 0.0491899, -0.1202333, 0.0664602, -0.1733067, 0.1694232
8: -0.0508476, 0.1284621, -0.0603825, 0.1499104, -0.2007580, 0.1888447
9: -0.1013522, 0.0771057, -0.1170500, 0.0870342, -0.1883864, 0.1941557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685736, upper bound: 0.0663984
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685736, upper bound: 0.0663980
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1092550, 0.0905294, -0.1327627, 0.1086677, -0.2179227, 0.2232921
1: -0.0677641, 0.0358849, -0.0810552, 0.0486160, -0.1163801, 0.1169401
2: -0.0914630, 0.1105535, -0.1131572, 0.1347550, -0.2262181, 0.2237107
3: 0.9465500, 1.0139796, 0.9318945, 1.0196424, -0.0730924, 0.0820850
4: -0.0702998, 0.0943808, -0.0901711, 0.1136833, -0.1839831, 0.1845520
5: -0.0501155, 0.1413912, -0.0666704, 0.1631496, -0.2132651, 0.2080616
6: -0.1021801, 0.0986729, -0.1234362, 0.1202070, -0.2223871, 0.2221091
7: -0.0955421, 0.0384157, -0.1120494, 0.0582496, -0.1537917, 0.1504651
8: -0.0427246, 0.1125568, -0.0543013, 0.1381115, -0.1808361, 0.1668582
9: -0.0890630, 0.0705892, -0.1082130, 0.0815379, -0.1706010, 0.1788022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0663985
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0663979
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1249191, 0.1018048, -0.1461586, 0.1177056, -0.2426247, 0.2479634
1: -0.0763374, 0.0444955, -0.0882413, 0.0564147, -0.1327521, 0.1327368
2: -0.1067018, 0.1254580, -0.1276770, 0.1458737, -0.2525755, 0.2531349
3: 0.9449932, 1.0185899, 0.9299443, 1.0251508, -0.0801576, 0.0886456
4: -0.0822238, 0.1078117, -0.0986722, 0.1260465, -0.2082703, 0.2064839
5: -0.0615130, 0.1556173, -0.0772323, 0.1757931, -0.2373061, 0.2328496
6: -0.1168360, 0.1129417, -0.1369816, 0.1329358, -0.2497719, 0.2499232
7: -0.1068466, 0.0491899, -0.1221348, 0.0656845, -0.1725310, 0.1713247
8: -0.0508476, 0.1284621, -0.0621061, 0.1511623, -0.2020099, 0.1905683
9: -0.1013522, 0.0771057, -0.1181491, 0.0878878, -0.1892401, 0.1952547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0602180, upper bound: 0.0615437
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0708001
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0707999
time: 1.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1092550, 0.0905294, -0.1346938, 0.1093504, -0.2186054, 0.2252232
1: -0.0677641, 0.0358849, -0.0818882, 0.0499607, -0.1177248, 0.1177730
2: -0.0914630, 0.1105535, -0.1161586, 0.1351828, -0.2266459, 0.2267121
3: 0.9465500, 1.0139796, 0.9356807, 1.0214248, -0.0748748, 0.0782988
4: -0.0702998, 0.0943808, -0.0901374, 0.1160593, -0.1863591, 0.1845183
5: -0.0501155, 0.1413912, -0.0686690, 0.1649854, -0.2151009, 0.2100603
6: -0.1021801, 0.0986729, -0.1259882, 0.1221877, -0.2243678, 0.2246611
7: -0.0955421, 0.0384157, -0.1137995, 0.0574316, -0.1529738, 0.1522152
8: -0.0427246, 0.1125568, -0.0559404, 0.1392239, -0.1819485, 0.1684972
9: -0.0890630, 0.0705892, -0.1092096, 0.0823179, -0.1713809, 0.1797988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0590111, upper bound: 0.0610495
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 172

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0609879, upper bound: 0.0620246
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0708000
time: 2.10 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0708003
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1339569, 0.1081996, -0.1348894, 0.1097408, -0.2436976, 0.2430890
1: -0.0812862, 0.0496853, -0.0820720, 0.0499711, -0.1312573, 0.1317574
2: -0.1161190, 0.1334740, -0.1159154, 0.1359190, -0.2520381, 0.2493894
3: 0.9411452, 1.0219066, 0.9347730, 1.0209875, -0.0798423, 0.0871336
4: -0.0885088, 0.1158945, -0.0908889, 0.1159386, -0.2044474, 0.2067835
5: -0.0684431, 0.1642000, -0.0685707, 0.1650929, -0.2335360, 0.2327707
6: -0.1257216, 0.1215128, -0.1258874, 0.1222362, -0.2479579, 0.2474002
7: -0.1135172, 0.0551319, -0.1137977, 0.0583848, -0.1719019, 0.1689296
8: -0.0559075, 0.1376643, -0.0557892, 0.1397002, -0.1956077, 0.1934536
9: -0.1082506, 0.0815768, -0.1095756, 0.0823160, -0.1905666, 0.1911524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0665998, upper bound: 0.0652893
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0665998, upper bound: 0.0652898
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1339569, 0.1081996, -0.1374049, 0.1108150, -0.2447719, 0.2456045
1: -0.0812862, 0.0496853, -0.0832302, 0.0516150, -0.1329012, 0.1329155
2: -0.1161190, 0.1334740, -0.1194925, 0.1368398, -0.2529588, 0.2529665
3: 0.9411452, 1.0219066, 0.9383941, 1.0229342, -0.0817890, 0.0835125
4: -0.0885088, 0.1158945, -0.0912343, 0.1188298, -0.2073385, 0.2071288
5: -0.0684431, 0.1642000, -0.0709766, 0.1674915, -0.2359346, 0.2351766
6: -0.1257216, 0.1215128, -0.1289717, 0.1247611, -0.2504827, 0.2504846
7: -0.1135172, 0.0551319, -0.1159858, 0.0579038, -0.1714209, 0.1711177
8: -0.0559075, 0.1376643, -0.0577139, 0.1413969, -0.1973043, 0.1953783
9: -0.1082506, 0.0815768, -0.1109969, 0.0833621, -0.1916126, 0.1925738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0688533, upper bound: 0.0688547
time: 1.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0688533, upper bound: 0.0688545
time: 1.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1257056, 0.1018357, -0.1299546, 0.1051673, -0.2308729, 0.2317903
1: -0.0766055, 0.0452098, -0.0790292, 0.0474826, -0.1240881, 0.1242390
2: -0.1084328, 0.1250741, -0.1122753, 0.1295428, -0.2379756, 0.2373494
3: 0.9467922, 1.0198317, 0.9440725, 1.0207751, -0.0739828, 0.0757591
4: -0.0816185, 0.1090821, -0.0853266, 0.1125100, -0.1941284, 0.1944087
5: -0.0625937, 0.1565054, -0.0655406, 0.1604113, -0.2230049, 0.2220460
6: -0.1182320, 0.1138820, -0.1220062, 0.1177639, -0.2359958, 0.2358882
7: -0.1077053, 0.0481695, -0.1106587, 0.0519149, -0.1596202, 0.1588282
8: -0.0518097, 0.1286576, -0.0538491, 0.1333410, -0.1851507, 0.1825068
9: -0.1015778, 0.0776070, -0.1050650, 0.0795599, -0.1811377, 0.1826720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0635911, upper bound: 0.0658688
time: 1.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0635905, upper bound: 0.0635912
time: 1.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.12 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663932
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663930
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663970
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663971
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0689979
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0689979
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0696166
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0700890, upper bound: 0.0696166
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680077
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0689169
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0689171
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663957, upper bound: 0.0678798
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663957, upper bound: 0.0678798
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663978, upper bound: 0.0685738
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663978, upper bound: 0.0685734
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663980, upper bound: 0.0685739
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0663980, upper bound: 0.0685734
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0630821, upper bound: 0.0630832
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0630821, upper bound: 0.0630835
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0630844, upper bound: 0.0651806
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0630844, upper bound: 0.0651804
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700890
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700886
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700890
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700888
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663941
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0685683
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663941
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0685683
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0685736, upper bound: 0.0663984
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0685736, upper bound: 0.0663980
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0663985
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0685739, upper bound: 0.0663979
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0708001
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0707999
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0708000
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0722205, upper bound: 0.0708003
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0665998, upper bound: 0.0652893
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0665998, upper bound: 0.0652898
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0688533, upper bound: 0.0688547
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0688533, upper bound: 0.0688545
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0635911, upper bound: 0.0658688
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.12
Output dim: 3, lower bound: -0.0635905, upper bound: 0.0635912

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1325893, 0.1086746, -0.2412639, 0.2412639
1: -0.0809906, 0.0484630, -0.0809906, 0.0484630, -0.1294536, 0.1294536
2: -0.1127485, 0.1349086, -0.1127485, 0.1349086, -0.2476571, 0.2476571
3: 0.9321295, 1.0193111, 0.9321295, 1.0193111, -0.0871816, 0.0871816
4: -0.0903902, 0.1133753, -0.0903902, 0.1133753, -0.2037655, 0.2037655
5: -0.0664112, 0.1629083, -0.0664112, 0.1629083, -0.2293194, 0.2293194
6: -0.1231250, 0.1199399, -0.1231250, 0.1199399, -0.2430649, 0.2430649
7: -0.1118449, 0.0585164, -0.1118449, 0.0585164, -0.1703614, 0.1703614
8: -0.0540693, 0.1380443, -0.0540693, 0.1380443, -0.1921135, 0.1921135
9: -0.1081993, 0.0813312, -0.1081993, 0.0813312, -0.1895305, 0.1895305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.03 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0624671, upper bound: 0.0627869
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0661731, upper bound: 0.0637482
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1183204, 0.0983966, -0.2309860, 0.2269950
1: -0.0809906, 0.0484630, -0.0731852, 0.0406024, -0.1215930, 0.1216483
2: -0.1127485, 0.1349086, -0.0988272, 0.1213201, -0.2340686, 0.2337358
3: 0.9321295, 1.0193111, 0.9330631, 1.0150847, -0.0829552, 0.0862480
4: -0.0903902, 0.1133753, -0.0794985, 0.1011350, -0.1915252, 0.1928738
5: -0.0664112, 0.1629083, -0.0560150, 0.1499757, -0.2163869, 0.2189233
6: -0.1231250, 0.1199399, -0.1097481, 0.1069696, -0.2300947, 0.2296880
7: -0.1118449, 0.0585164, -0.1015449, 0.0487259, -0.1605708, 0.1600614
8: -0.0540693, 0.1380443, -0.0466571, 0.1235784, -0.1776477, 0.1847013
9: -0.1081993, 0.0813312, -0.0969927, 0.0754595, -0.1836588, 0.1783239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0624671, upper bound: 0.0627876
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0661731, upper bound: 0.0637482
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1346416, 0.1094295, -0.1325893, 0.1086746, -0.2433162, 0.2420188
1: -0.0818898, 0.0498854, -0.0809906, 0.0484630, -0.1303528, 0.1308760
2: -0.1159472, 0.1353870, -0.1127485, 0.1349086, -0.2508558, 0.2481355
3: 0.9358367, 1.0211815, 0.9321295, 1.0193111, -0.0834744, 0.0890520
4: -0.0903837, 0.1158983, -0.0903902, 0.1133753, -0.2037590, 0.2062885
5: -0.0685374, 0.1648742, -0.0664112, 0.1629083, -0.2314456, 0.2312854
6: -0.1258359, 0.1220468, -0.1231250, 0.1199399, -0.2457758, 0.2451718
7: -0.1137101, 0.0577350, -0.1118449, 0.0585164, -0.1722265, 0.1695800
8: -0.0558092, 0.1392670, -0.0540693, 0.1380443, -0.1938534, 0.1933363
9: -0.1092722, 0.0821861, -0.1081993, 0.0813312, -0.1906034, 0.1903854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.32 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0643419, upper bound: 0.0637514
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0665796, upper bound: 0.0637515
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1346416, 0.1094295, -0.1183204, 0.0983966, -0.2330382, 0.2277499
1: -0.0818898, 0.0498854, -0.0731852, 0.0406024, -0.1224921, 0.1230706
2: -0.1159472, 0.1353870, -0.0988272, 0.1213201, -0.2372673, 0.2342142
3: 0.9358367, 1.0211815, 0.9330631, 1.0150847, -0.0792481, 0.0881184
4: -0.0903837, 0.1158983, -0.0794985, 0.1011350, -0.1915187, 0.1953968
5: -0.0685374, 0.1648742, -0.0560150, 0.1499757, -0.2185131, 0.2208892
6: -0.1258359, 0.1220468, -0.1097481, 0.1069696, -0.2328055, 0.2317949
7: -0.1137101, 0.0577350, -0.1015449, 0.0487259, -0.1624360, 0.1592800
8: -0.0558092, 0.1392670, -0.0466571, 0.1235784, -0.1793876, 0.1859241
9: -0.1092722, 0.0821861, -0.0969927, 0.0754595, -0.1847317, 0.1791788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0643419, upper bound: 0.0637516
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0665796, upper bound: 0.0637517
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1224673, 0.1007717, -0.2333611, 0.2311419
1: -0.0809906, 0.0484630, -0.0752174, 0.0428783, -0.1238689, 0.1236805
2: -0.1127485, 0.1349086, -0.1031720, 0.1246065, -0.2373551, 0.2380806
3: 0.9321295, 1.0193111, 0.9415862, 1.0166223, -0.0844928, 0.0777249
4: -0.0903902, 0.1133753, -0.0819490, 0.1049587, -0.1953489, 0.1953243
5: -0.0664112, 0.1629083, -0.0591207, 0.1532771, -0.2196882, 0.2220289
6: -0.1231250, 0.1199399, -0.1137879, 0.1104516, -0.2335766, 0.2337278
7: -0.1118449, 0.0585164, -0.1046996, 0.0497274, -0.1615724, 0.1632161
8: -0.0540693, 0.1380443, -0.0489237, 0.1268218, -0.1808911, 0.1869680
9: -0.1081993, 0.0813312, -0.0999834, 0.0760543, -0.1842536, 0.1813145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.49 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0639771, upper bound: 0.0645034
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0675237, upper bound: 0.0665001
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1070720, 0.0897472, -0.2223365, 0.2157466
1: -0.0809906, 0.0484630, -0.0667697, 0.0345466, -0.1155372, 0.1152328
2: -0.1127485, 0.1349086, -0.0884381, 0.1098502, -0.2225987, 0.2233467
3: 0.9321295, 1.0193111, 0.9429960, 1.0122710, -0.0801415, 0.0763150
4: -0.0903902, 0.1133753, -0.0700960, 0.0918550, -0.1822452, 0.1834713
5: -0.0664112, 0.1629083, -0.0480115, 0.1393077, -0.2057188, 0.2109198
6: -0.1231250, 0.1199399, -0.0995537, 0.0965336, -0.2196586, 0.2194936
7: -0.1118449, 0.0585164, -0.0936450, 0.0390376, -0.1508825, 0.1521614
8: -0.0540693, 0.1380443, -0.0412567, 0.1111409, -0.1652102, 0.1793010
9: -0.1081993, 0.0813312, -0.0879000, 0.0697001, -0.1778995, 0.1692312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.39 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0639771, upper bound: 0.0645034
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0675237, upper bound: 0.0665001
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1346416, 0.1094295, -0.1224673, 0.1007717, -0.2354133, 0.2318968
1: -0.0818898, 0.0498854, -0.0752174, 0.0428783, -0.1247681, 0.1251028
2: -0.1159472, 0.1353870, -0.1031720, 0.1246065, -0.2405538, 0.2385591
3: 0.9358367, 1.0211815, 0.9415862, 1.0166223, -0.0807856, 0.0795953
4: -0.0903837, 0.1158983, -0.0819490, 0.1049587, -0.1953424, 0.1978473
5: -0.0685374, 0.1648742, -0.0591207, 0.1532771, -0.2218144, 0.2239949
6: -0.1258359, 0.1220468, -0.1137879, 0.1104516, -0.2362875, 0.2358347
7: -0.1137101, 0.0577350, -0.1046996, 0.0497274, -0.1634376, 0.1624347
8: -0.0558092, 0.1392670, -0.0489237, 0.1268218, -0.1826310, 0.1881907
9: -0.1092722, 0.0821861, -0.0999834, 0.0760543, -0.1853265, 0.1821694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.39 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0658602, upper bound: 0.0667565
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680562, upper bound: 0.0670944
time: 2.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1346416, 0.1094295, -0.1070720, 0.0897472, -0.2243888, 0.2165015
1: -0.0818898, 0.0498854, -0.0667697, 0.0345466, -0.1164364, 0.1166551
2: -0.1159472, 0.1353870, -0.0884381, 0.1098502, -0.2257974, 0.2238251
3: 0.9358367, 1.0211815, 0.9429960, 1.0122710, -0.0764344, 0.0781854
4: -0.0903837, 0.1158983, -0.0700960, 0.0918550, -0.1822388, 0.1859943
5: -0.0685374, 0.1648742, -0.0480115, 0.1393077, -0.2078450, 0.2128857
6: -0.1258359, 0.1220468, -0.0995537, 0.0965336, -0.2223695, 0.2216005
7: -0.1137101, 0.0577350, -0.0936450, 0.0390376, -0.1527477, 0.1513801
8: -0.0558092, 0.1392670, -0.0412567, 0.1111409, -0.1669500, 0.1805237
9: -0.1092722, 0.0821861, -0.0879000, 0.0697001, -0.1789724, 0.1700861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.34 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0658602, upper bound: 0.0667564
time: 1.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680562, upper bound: 0.0670944
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1183204, 0.0983966, -0.1224673, 0.1007717, -0.2190921, 0.2208639
1: -0.0731852, 0.0406024, -0.0752174, 0.0428783, -0.1160636, 0.1158198
2: -0.0988272, 0.1213201, -0.1031720, 0.1246065, -0.2234338, 0.2244922
3: 0.9330631, 1.0150847, 0.9415862, 1.0166223, -0.0835592, 0.0734985
4: -0.0794985, 0.1011350, -0.0819490, 0.1049587, -0.1844572, 0.1830840
5: -0.0560150, 0.1499757, -0.0591207, 0.1532771, -0.2092921, 0.2090964
6: -0.1097481, 0.1069696, -0.1137879, 0.1104516, -0.2201997, 0.2207576
7: -0.1015449, 0.0487259, -0.1046996, 0.0497274, -0.1512724, 0.1534255
8: -0.0466571, 0.1235784, -0.0489237, 0.1268218, -0.1734788, 0.1725021
9: -0.0969927, 0.0754595, -0.0999834, 0.0760543, -0.1730470, 0.1754429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 253

Time for candidate selection: 16.29 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0591989, upper bound: 0.0625884
time: 1.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0637487, upper bound: 0.0654488
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1183204, 0.0983966, -0.1070720, 0.0897472, -0.2080676, 0.2054686
1: -0.0731852, 0.0406024, -0.0667697, 0.0345466, -0.1077318, 0.1073721
2: -0.0988272, 0.1213201, -0.0884381, 0.1098502, -0.2086775, 0.2097582
3: 0.9330631, 1.0150847, 0.9429960, 1.0122710, -0.0792080, 0.0720887
4: -0.0794985, 0.1011350, -0.0700960, 0.0918550, -0.1713535, 0.1712310
5: -0.0560150, 0.1499757, -0.0480115, 0.1393077, -0.1953227, 0.1979872
6: -0.1097481, 0.1069696, -0.0995537, 0.0965336, -0.2062817, 0.2065234
7: -0.1015449, 0.0487259, -0.0936450, 0.0390376, -0.1405825, 0.1423709
8: -0.0466571, 0.1235784, -0.0412567, 0.1111409, -0.1577979, 0.1648351
9: -0.0969927, 0.0754595, -0.0879000, 0.0697001, -0.1666928, 0.1633595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 253

Time for candidate selection: 16.45 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0591989, upper bound: 0.0625882
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0637487, upper bound: 0.0654488
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1198855, 0.0988301, -0.1224673, 0.1007717, -0.2206573, 0.2212973
1: -0.0738123, 0.0417495, -0.0752174, 0.0428783, -0.1166906, 0.1169669
2: -0.1015046, 0.1214103, -0.1031720, 0.1246065, -0.2261111, 0.2245824
3: 0.9371079, 1.0167693, 0.9415862, 1.0166223, -0.0795144, 0.0751831
4: -0.0791954, 0.1032070, -0.0819490, 0.1049587, -0.1841541, 0.1851560
5: -0.0577697, 0.1514509, -0.0591207, 0.1532771, -0.2110467, 0.2105716
6: -0.1119747, 0.1086060, -0.1137879, 0.1104516, -0.2224263, 0.2223939
7: -0.1030404, 0.0475872, -0.1046996, 0.0497274, -0.1527678, 0.1522868
8: -0.0481198, 0.1243111, -0.0489237, 0.1268218, -0.1749416, 0.1732348
9: -0.0977124, 0.0760327, -0.0999834, 0.0760543, -0.1737667, 0.1760161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.81 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0636783, upper bound: 0.0659511
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0657522, upper bound: 0.0663650
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1198855, 0.0988301, -0.1070720, 0.0897472, -0.2096327, 0.2059021
1: -0.0738123, 0.0417495, -0.0667697, 0.0345466, -0.1083589, 0.1085192
2: -0.1015046, 0.1214103, -0.0884381, 0.1098502, -0.2113548, 0.2098484
3: 0.9371079, 1.0167693, 0.9429960, 1.0122710, -0.0751631, 0.0737733
4: -0.0791954, 0.1032070, -0.0700960, 0.0918550, -0.1710504, 0.1733030
5: -0.0577697, 0.1514509, -0.0480115, 0.1393077, -0.1970773, 0.1994624
6: -0.1119747, 0.1086060, -0.0995537, 0.0965336, -0.2085083, 0.2081597
7: -0.1030404, 0.0475872, -0.0936450, 0.0390376, -0.1420780, 0.1412323
8: -0.0481198, 0.1243111, -0.0412567, 0.1111409, -0.1592607, 0.1655678
9: -0.0977124, 0.0760327, -0.0879000, 0.0697001, -0.1674126, 0.1639328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.41 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0636783, upper bound: 0.0659511
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0657522, upper bound: 0.0663648
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1346416, 0.1094295, -0.2420188, 0.2433162
1: -0.0809906, 0.0484630, -0.0818898, 0.0498854, -0.1308760, 0.1303528
2: -0.1127485, 0.1349086, -0.1159472, 0.1353870, -0.2481355, 0.2508558
3: 0.9321295, 1.0193111, 0.9358367, 1.0211815, -0.0890520, 0.0834744
4: -0.0903902, 0.1133753, -0.0903837, 0.1158983, -0.2062885, 0.2037590
5: -0.0664112, 0.1629083, -0.0685374, 0.1648742, -0.2312854, 0.2314456
6: -0.1231250, 0.1199399, -0.1258359, 0.1220468, -0.2451718, 0.2457758
7: -0.1118449, 0.0585164, -0.1137101, 0.0577350, -0.1695800, 0.1722265
8: -0.0540693, 0.1380443, -0.0558092, 0.1392670, -0.1933363, 0.1938534
9: -0.1081993, 0.0813312, -0.1092722, 0.0821861, -0.1903854, 0.1906034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 185
type: A, layer: 3, pos: 225

Time for candidate selection: 16.50 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0640284, upper bound: 0.0639415
time: 1.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0677764, upper bound: 0.0658921
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1325893, 0.1086746, -0.1198855, 0.0988301, -0.2314194, 0.2285601
1: -0.0809906, 0.0484630, -0.0738123, 0.0417495, -0.1227401, 0.1222753
2: -0.1127485, 0.1349086, -0.1015046, 0.1214103, -0.2341588, 0.2364132
3: 0.9321295, 1.0193111, 0.9371079, 1.0167693, -0.0846398, 0.0822031
4: -0.0903902, 0.1133753, -0.0791954, 0.1032070, -0.1935972, 0.1925707
5: -0.0664112, 0.1629083, -0.0577697, 0.1514509, -0.2178621, 0.2206779
6: -0.1231250, 0.1199399, -0.1119747, 0.1086060, -0.2317310, 0.2319146
7: -0.1118449, 0.0585164, -0.1030404, 0.0475872, -0.1594322, 0.1615568
8: -0.0540693, 0.1380443, -0.0481198, 0.1243111, -0.1783803, 0.1861641
9: -0.1081993, 0.0813312, -0.0977124, 0.0760327, -0.1842321, 0.1790436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 172

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.37 + 596.36 = 600.73 seconds
