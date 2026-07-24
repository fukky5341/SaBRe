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
execution time: IAR + RelationalAnalysis = 1.24 + 3.09 = 4.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0785230, upper bound: 0.0785227

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 57

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772780, upper bound: 0.0780093
time: 2.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0785101, upper bound: 0.0785101
time: 2.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 4.63
Output dim: 3, lower bound: -0.0772780, upper bound: 0.0780093
IS_A2, status: Status.UNKNOWN, split count: 1, time: 4.63
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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0761942, upper bound: 0.0761829
time: 1.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772774, upper bound: 0.0779986
time: 2.24 seconds

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0777572, upper bound: 0.0770551
time: 1.50 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0785043, upper bound: 0.0785040
time: 2.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.23
Output dim: 3, lower bound: -0.0761942, upper bound: 0.0761829
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.23
Output dim: 3, lower bound: -0.0772774, upper bound: 0.0779986
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.23
Output dim: 3, lower bound: -0.0777572, upper bound: 0.0770551
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.23
Output dim: 3, lower bound: -0.0785043, upper bound: 0.0785040

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

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747527, upper bound: 0.0757918
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747527, upper bound: 0.0761829
time: 1.95 seconds

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
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

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
0: -0.1323732, 0.1071173, -0.1353763, 0.1101262, -0.2424994, 0.2424936
1: -0.0804212, 0.0487821, -0.0823548, 0.0502417, -0.1306629, 0.1311369
2: -0.1144225, 0.1321742, -0.1163846, 0.1364059, -0.2508285, 0.2485588
3: 0.9419975, 1.0212650, 0.9341533, 1.0211234, -0.0791259, 0.0871118
4: -0.0875320, 0.1144306, -0.0912854, 0.1163481, -0.2038801, 0.2057160
5: -0.0672050, 0.1626458, -0.0689235, 0.1655810, -0.2327861, 0.2315693
6: -0.1241387, 0.1199657, -0.1263406, 0.1227073, -0.2468460, 0.2463064
7: -0.1123206, 0.0542152, -0.1141441, 0.0588242, -0.1711447, 0.1683594
8: -0.0549940, 0.1360651, -0.0560395, 0.1402579, -0.1952519, 0.1921046
9: -0.1070914, 0.0807080, -0.1099672, 0.0826156, -0.1897070, 0.1906752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0747790
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0770550
time: 1.58 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.1386419, 0.1117204, -0.1379820, 0.1112633, -0.2499052, 0.2497024
1: -0.0839203, 0.0523124, -0.0835610, 0.0519393, -0.1358596, 0.1358733
2: -0.1207412, 0.1379796, -0.1200568, 0.1374090, -0.2581502, 0.2580364
3: 0.9375139, 1.0233499, 0.9377388, 1.0231041, -0.0855901, 0.0856110
4: -0.0921365, 0.1199132, -0.0916952, 0.1193198, -0.2114563, 0.2116084
5: -0.0719050, 0.1686847, -0.0713998, 0.1680626, -0.2399676, 0.2400845
6: -0.1301615, 0.1259439, -0.1295154, 0.1253161, -0.2554776, 0.2554594
7: -0.1168876, 0.0588036, -0.1163985, 0.0584015, -0.1752891, 0.1752021
8: -0.0583850, 0.1427064, -0.0580165, 0.1420440, -0.2004290, 0.2007229
9: -0.1119547, 0.0840181, -0.1114588, 0.0837022, -0.1956569, 0.1954769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0772774
time: 1.82 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0785040
time: 1.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.00 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0747527, upper bound: 0.0757918
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0747527, upper bound: 0.0761829
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0747791, upper bound: 0.0762314
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0747792, upper bound: 0.0779992
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0747790
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0762319, upper bound: 0.0770550
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0772774
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.00
Output dim: 3, lower bound: -0.0779992, upper bound: 0.0785040

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1454258, 0.1183378, -0.1347946, 0.1096616, -0.2550874, 0.2531324
1: -0.0887180, 0.0557632, -0.0820140, 0.0499177, -0.1386357, 0.1377772
2: -0.1258563, 0.1468174, -0.1158215, 0.1358266, -0.2616830, 0.2626389
3: 0.9253336, 1.0241864, 0.9349787, 1.0209588, -0.0956252, 0.0892076
4: -0.0996217, 0.1251453, -0.0908148, 0.1158576, -0.2154793, 0.2159601
5: -0.0761432, 0.1753820, -0.0685010, 0.1649848, -0.2411279, 0.2438830
6: -0.1357860, 0.1320996, -0.1257978, 0.1221362, -0.2579222, 0.2578973
7: -0.1212357, 0.0682992, -0.1137297, 0.0582870, -0.1795227, 0.1820289
8: -0.0623444, 0.1513208, -0.0557394, 0.1395800, -0.2019243, 0.2070602
9: -0.1182432, 0.0877123, -0.1095007, 0.0822321, -0.2004753, 0.1972130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747179
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0757918
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1475226, 0.1193831, -0.1347946, 0.1096616, -0.2571842, 0.2541777
1: -0.0900766, 0.0571861, -0.0820140, 0.0499177, -0.1399943, 0.1392001
2: -0.1290567, 0.1474789, -0.1158215, 0.1358266, -0.2648833, 0.2633004
3: 0.9292635, 1.0262902, 0.9349787, 1.0209588, -0.0916953, 0.0913115
4: -0.0996514, 0.1279899, -0.0908148, 0.1158576, -0.2155090, 0.2188047
5: -0.0782609, 0.1775912, -0.0685010, 0.1649848, -0.2432457, 0.2460922
6: -0.1386461, 0.1342273, -0.1257978, 0.1221362, -0.2607823, 0.2600250
7: -0.1231311, 0.0681176, -0.1137297, 0.0582870, -0.1814181, 0.1818473
8: -0.0648706, 0.1525695, -0.0557394, 0.1395800, -0.2044506, 0.2083089
9: -0.1194324, 0.0885569, -0.1095007, 0.0822321, -0.2016645, 0.1980576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747773
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0761824
time: 1.58 seconds

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0759189
time: 1.67 seconds

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 57

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747178
time: 1.66 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0757923
time: 1.72 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1323732, 0.1071173, -0.1454258, 0.1179186, -0.2502918, 0.2525431
1: -0.0804212, 0.0487821, -0.0880667, 0.0557632, -0.1361844, 0.1368489
2: -0.1144225, 0.1321742, -0.1258563, 0.1466007, -0.2610232, 0.2580305
3: 0.9419975, 1.0212650, 0.9253336, 1.0237833, -0.0817858, 0.0959314
4: -0.0875320, 0.1144306, -0.0996217, 0.1246984, -0.2122304, 0.2140523
5: -0.0672050, 0.1626458, -0.0761432, 0.1750756, -0.2422807, 0.2387889
6: -0.1241387, 0.1199657, -0.1355755, 0.1320996, -0.2562382, 0.2555412
7: -0.1123206, 0.0542152, -0.1212357, 0.0673971, -0.1797177, 0.1754509
8: -0.0549940, 0.1360651, -0.0611325, 0.1513208, -0.2063148, 0.1971976
9: -0.1070914, 0.0807080, -0.1181041, 0.0877123, -0.1948037, 0.1988122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0747527
time: 1.53 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0747789
time: 1.50 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1323732, 0.1071173, -0.1348894, 0.1097408, -0.2421140, 0.2420066
1: -0.0804212, 0.0487821, -0.0820720, 0.0499711, -0.1303923, 0.1308542
2: -0.1144225, 0.1321742, -0.1159154, 0.1359190, -0.2503416, 0.2480896
3: 0.9419975, 1.0212650, 0.9347730, 1.0209875, -0.0789900, 0.0864921
4: -0.0875320, 0.1144306, -0.0908889, 0.1159386, -0.2034707, 0.2053196
5: -0.0672050, 0.1626458, -0.0685707, 0.1650929, -0.2322980, 0.2312165
6: -0.1241387, 0.1199657, -0.1258874, 0.1222362, -0.2463749, 0.2458531
7: -0.1123206, 0.0542152, -0.1137977, 0.0583848, -0.1707053, 0.1680130
8: -0.0549940, 0.1360651, -0.0557892, 0.1397002, -0.1946943, 0.1918543
9: -0.1070914, 0.0807080, -0.1095756, 0.0823160, -0.1894075, 0.1902837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0747527
time: 1.64 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0770551
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1386419, 0.1117204, -0.1475226, 0.1186848, -0.2573267, 0.2592430
1: -0.0839203, 0.0523124, -0.0889918, 0.0571861, -0.1411065, 0.1413042
2: -0.1207412, 0.1379796, -0.1290567, 0.1471179, -0.2678592, 0.2670363
3: 0.9375139, 1.0233499, 0.9292635, 1.0256188, -0.0881048, 0.0940864
4: -0.0921365, 0.1199132, -0.0996514, 0.1272454, -0.2193819, 0.2195646
5: -0.0719050, 0.1686847, -0.0782609, 0.1770809, -0.2489859, 0.2469457
6: -0.1301615, 0.1259439, -0.1382954, 0.1342273, -0.2643888, 0.2642393
7: -0.1168876, 0.0588036, -0.1231311, 0.0666182, -0.1835058, 0.1819347
8: -0.0583850, 0.1427064, -0.0628522, 0.1525695, -0.2109545, 0.2055587
9: -0.1119547, 0.0840181, -0.1192007, 0.0885569, -0.2005115, 0.2032188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0761937
time: 1.40 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0772774
time: 2.11 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1386419, 0.1117204, -0.1374049, 0.1108150, -0.2494569, 0.2491253
1: -0.0839203, 0.0523124, -0.0832302, 0.0516150, -0.1355354, 0.1355425
2: -0.1207412, 0.1379796, -0.1194925, 0.1368398, -0.2575811, 0.2574721
3: 0.9375139, 1.0233499, 0.9383941, 1.0229342, -0.0854203, 0.0849558
4: -0.0921365, 0.1199132, -0.0912343, 0.1188298, -0.2109663, 0.2111475
5: -0.0719050, 0.1686847, -0.0709766, 0.1674915, -0.2393965, 0.2396613
6: -0.1301615, 0.1259439, -0.1289717, 0.1247611, -0.2549226, 0.2549157
7: -0.1168876, 0.0588036, -0.1159858, 0.0579038, -0.1747914, 0.1747893
8: -0.0583850, 0.1427064, -0.0577139, 0.1413969, -0.1997818, 0.2004204
9: -0.1119547, 0.0840181, -0.1109969, 0.0833621, -0.1953167, 0.1950151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0777567
time: 1.61 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0785040
time: 1.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.63 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747179
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0757918
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747773
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0761824
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0759189
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0762314
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0747178
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0747178, upper bound: 0.0757923
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0747527
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0747789
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0747527
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0770551
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0761937
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0772774
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0777567
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.63
Output dim: 3, lower bound: -0.0757923, upper bound: 0.0785040

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1454258, 0.1183378, -0.1454258, 0.1179186, -0.2633444, 0.2637635
1: -0.0887180, 0.0557632, -0.0880667, 0.0557632, -0.1444812, 0.1438299
2: -0.1258563, 0.1468174, -0.1258563, 0.1466007, -0.2724570, 0.2726738
3: 0.9253336, 1.0241864, 0.9253336, 1.0237833, -0.0984497, 0.0988528
4: -0.0996217, 0.1251453, -0.0996217, 0.1246984, -0.2243201, 0.2247670
5: -0.0761432, 0.1753820, -0.0761432, 0.1750756, -0.2512188, 0.2515251
6: -0.1357860, 0.1320996, -0.1355755, 0.1320996, -0.2678856, 0.2676750
7: -0.1212357, 0.0682992, -0.1212357, 0.0673971, -0.1886327, 0.1895348
8: -0.0623444, 0.1513208, -0.0611325, 0.1513208, -0.2136651, 0.2124533
9: -0.1182432, 0.0877123, -0.1181041, 0.0877123, -0.2059555, 0.2058164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663932, upper bound: 0.0687812
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0663923
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1454258, 0.1183378, -0.1348894, 0.1097408, -0.2551666, 0.2532271
1: -0.0887180, 0.0557632, -0.0820720, 0.0499711, -0.1386892, 0.1378352
2: -0.1258563, 0.1468174, -0.1159154, 0.1359190, -0.2617754, 0.2627329
3: 0.9253336, 1.0241864, 0.9347730, 1.0209875, -0.0956539, 0.0894134
4: -0.0996217, 0.1251453, -0.0908889, 0.1159386, -0.2155603, 0.2160343
5: -0.0761432, 0.1753820, -0.0685707, 0.1650929, -0.2412361, 0.2439526
6: -0.1357860, 0.1320996, -0.1258874, 0.1222362, -0.2580222, 0.2579870
7: -0.1212357, 0.0682992, -0.1137977, 0.0583848, -0.1796204, 0.1820969
8: -0.0623444, 0.1513208, -0.0557892, 0.1397002, -0.2020446, 0.2071100
9: -0.1182432, 0.0877123, -0.1095756, 0.0823160, -0.2005592, 0.1972879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663938
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0680078
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1475226, 0.1193831, -0.1454258, 0.1179186, -0.2654412, 0.2648089
1: -0.0900766, 0.0571861, -0.0880667, 0.0557632, -0.1458398, 0.1452529
2: -0.1290567, 0.1474789, -0.1258563, 0.1466007, -0.2756574, 0.2733352
3: 0.9292635, 1.0262902, 0.9253336, 1.0237833, -0.0945199, 0.1009566
4: -0.0996514, 0.1279899, -0.0996217, 0.1246984, -0.2243498, 0.2276116
5: -0.0782609, 0.1775912, -0.0761432, 0.1750756, -0.2533365, 0.2537344
6: -0.1386461, 0.1342273, -0.1355755, 0.1320996, -0.2707456, 0.2698027
7: -0.1231311, 0.0681176, -0.1212357, 0.0673971, -0.1905282, 0.1893533
8: -0.0648706, 0.1525695, -0.0611325, 0.1513208, -0.2161914, 0.2137021
9: -0.1194324, 0.0885569, -0.1181041, 0.0877123, -0.2071447, 0.2066610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687015, upper bound: 0.0702826
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0663958
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1475226, 0.1193831, -0.1348894, 0.1097408, -0.2572634, 0.2542724
1: -0.0900766, 0.0571861, -0.0820720, 0.0499711, -0.1400478, 0.1392581
2: -0.1290567, 0.1474789, -0.1159154, 0.1359190, -0.2649757, 0.2633943
3: 0.9292635, 1.0262902, 0.9347730, 1.0209875, -0.0917240, 0.0915172
4: -0.0996514, 0.1279899, -0.0908889, 0.1159386, -0.2155900, 0.2188789
5: -0.0782609, 0.1775912, -0.0685707, 0.1650929, -0.2433539, 0.2461619
6: -0.1386461, 0.1342273, -0.1258874, 0.1222362, -0.2608823, 0.2601147
7: -0.1231311, 0.0681176, -0.1137977, 0.0583848, -0.1815159, 0.1819154
8: -0.0648706, 0.1525695, -0.0557892, 0.1397002, -0.2045708, 0.2083588
9: -0.1194324, 0.0885569, -0.1095756, 0.0823160, -0.2017484, 0.1981325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0687015, upper bound: 0.0702828
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0689171
time: 1.36 seconds

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0702823, upper bound: 0.0687015
time: 1.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663958, upper bound: 0.0678793
time: 1.48 seconds

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

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

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0697534, upper bound: 0.0722877
time: 2.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772192, upper bound: 0.0772390
time: 1.63 seconds

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0722580, upper bound: 0.0715879
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772192, upper bound: 0.0779935
time: 1.61 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1348894, 0.1097408, -0.1454258, 0.1179186, -0.2528079, 0.2551666
1: -0.0820720, 0.0499711, -0.0880667, 0.0557632, -0.1378352, 0.1380379
2: -0.1159154, 0.1359190, -0.1258563, 0.1466007, -0.2625161, 0.2617754
3: 0.9347730, 1.0209875, 0.9253336, 1.0237833, -0.0890104, 0.0956539
4: -0.0908889, 0.1159386, -0.0996217, 0.1246984, -0.2155874, 0.2155603
5: -0.0685707, 0.1650929, -0.0761432, 0.1750756, -0.2436463, 0.2412361
6: -0.1258874, 0.1222362, -0.1355755, 0.1320996, -0.2579870, 0.2578117
7: -0.1137977, 0.0583848, -0.1212357, 0.0673971, -0.1811948, 0.1796204
8: -0.0557892, 0.1397002, -0.0611325, 0.1513208, -0.2071100, 0.2008328
9: -0.1095756, 0.0823160, -0.1181041, 0.0877123, -0.1972879, 0.2004202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700890
time: 1.18 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0663938
time: 2.06 seconds

## BFS IS instance: IS_A2_B1_B1_A2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0712056
time: 1.30 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0663983
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1348894, 0.1097408, -0.1348894, 0.1097408, -0.2446301, 0.2446301
1: -0.0820720, 0.0499711, -0.0820720, 0.0499711, -0.1320432, 0.1320432
2: -0.1159154, 0.1359190, -0.1159154, 0.1359190, -0.2518345, 0.2518345
3: 0.9347730, 1.0209875, 0.9347730, 1.0209875, -0.0862145, 0.0862145
4: -0.0908889, 0.1159386, -0.0908889, 0.1159386, -0.2068276, 0.2068276
5: -0.0685707, 0.1650929, -0.0685707, 0.1650929, -0.2336636, 0.2336636
6: -0.1258874, 0.1222362, -0.1258874, 0.1222362, -0.2481236, 0.2481236
7: -0.1137977, 0.0583848, -0.1137977, 0.0583848, -0.1721825, 0.1721825
8: -0.0557892, 0.1397002, -0.0557892, 0.1397002, -0.1954895, 0.1954895
9: -0.1095756, 0.0823160, -0.1095756, 0.0823160, -0.1918916, 0.1918916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0694328
time: 1.39 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0769830
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_B2_A2

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0694074, upper bound: 0.0663973
time: 1.40 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0770510
time: 1.89 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1348894, 0.1097408, -0.1475226, 0.1186848, -0.2535742, 0.2572634
1: -0.0820720, 0.0499711, -0.0889918, 0.0571861, -0.1392581, 0.1389629
2: -0.1159154, 0.1359190, -0.1290567, 0.1471179, -0.2630334, 0.2649757
3: 0.9347730, 1.0209875, 0.9292635, 1.0256188, -0.0908458, 0.0917240
4: -0.0908889, 0.1159386, -0.0996514, 0.1272454, -0.2181343, 0.2155900
5: -0.0685707, 0.1650929, -0.0782609, 0.1770809, -0.2456516, 0.2433539
6: -0.1258874, 0.1222362, -0.1382954, 0.1342273, -0.2601147, 0.2605316
7: -0.1137977, 0.0583848, -0.1231311, 0.0666182, -0.1804159, 0.1815159
8: -0.0557892, 0.1397002, -0.0628522, 0.1525695, -0.2083588, 0.2025525
9: -0.1095756, 0.0823160, -0.1192007, 0.0885569, -0.1981325, 0.2015167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0691930, upper bound: 0.0690838
time: 1.36 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0685683
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_B1_A2

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
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0740804
time: 2.95 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0707999
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1348894, 0.1097408, -0.1374049, 0.1108150, -0.2457044, 0.2471457
1: -0.0820720, 0.0499711, -0.0832302, 0.0516150, -0.1336871, 0.1332013
2: -0.1159154, 0.1359190, -0.1194925, 0.1368398, -0.2527552, 0.2554116
3: 0.9347730, 1.0209875, 0.9383941, 1.0229342, -0.0881612, 0.0825934
4: -0.0908889, 0.1159386, -0.0912343, 0.1188298, -0.2097187, 0.2071729
5: -0.0685707, 0.1650929, -0.0709766, 0.1674915, -0.2360622, 0.2360695
6: -0.1258874, 0.1222362, -0.1289717, 0.1247611, -0.2506485, 0.2512079
7: -0.1137977, 0.0583848, -0.1159858, 0.0579038, -0.1717015, 0.1743706
8: -0.0557892, 0.1397002, -0.0577139, 0.1413969, -0.1971861, 0.1974142
9: -0.1095756, 0.0823160, -0.1109969, 0.0833621, -0.1929377, 0.1933130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0698467
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0777504
time: 2.05 seconds

## BFS IS instance: IS_A2_B2_B2_A2

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

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0748720
time: 1.46 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0784992
time: 2.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.80 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663932, upper bound: 0.0687812
IS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0663923
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0687814, upper bound: 0.0663938
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663919, upper bound: 0.0680078
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0687015, upper bound: 0.0702826
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0663958
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0687015, upper bound: 0.0702828
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0678798, upper bound: 0.0689171
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0702823, upper bound: 0.0687015
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663958, upper bound: 0.0678793
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0702823, upper bound: 0.0698919
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663958, upper bound: 0.0678801
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0697534, upper bound: 0.0722877
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0772192, upper bound: 0.0772390
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0722580, upper bound: 0.0715879
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0772192, upper bound: 0.0779935
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0700890
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0663938
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0712056
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0663983
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0694328
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0769830
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0694074, upper bound: 0.0663973
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0770510
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0691930, upper bound: 0.0690838
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0685683
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0740804
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0707999
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0698467
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0777504
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0748720
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.80
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0784992

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1440577, 0.1172293, -0.1325893, 0.1086746, -0.2527323, 0.2498186
1: -0.0877690, 0.0549871, -0.0809906, 0.0484630, -0.1362320, 0.1359777
2: -0.1244562, 0.1455102, -0.1127485, 0.1349086, -0.2593647, 0.2582587
3: 0.9260395, 1.0235910, 0.9321295, 1.0193111, -0.0932716, 0.0914615
4: -0.0986425, 0.1238031, -0.0903902, 0.1133753, -0.2120179, 0.2141933
5: -0.0751096, 0.1739909, -0.0664112, 0.1629083, -0.2380178, 0.2404020
6: -0.1343960, 0.1308073, -0.1231250, 0.1199399, -0.2543359, 0.2539323
7: -0.1202333, 0.0670915, -0.1118449, 0.0585164, -0.1787498, 0.1789365
8: -0.0612304, 0.1499104, -0.0540693, 0.1380443, -0.1992747, 0.2039797
9: -0.1171473, 0.0870342, -0.1081993, 0.0813312, -0.1984785, 0.1952335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0663920, upper bound: 0.0663922
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0663920, upper bound: 0.0663925
time: 0.97 seconds

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680075
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1461586, 0.1182791, -0.1325893, 0.1086746, -0.2548332, 0.2508684
1: -0.0891323, 0.0564147, -0.0809906, 0.0484630, -0.1375953, 0.1374053
2: -0.1276770, 0.1461701, -0.1127485, 0.1349086, -0.2625856, 0.2589186
3: 0.9299443, 1.0257020, 0.9321295, 1.0193111, -0.0893668, 0.0935725
4: -0.0986722, 0.1266581, -0.0903902, 0.1133753, -0.2120475, 0.2170482
5: -0.0772323, 0.1762123, -0.0664112, 0.1629083, -0.2401406, 0.2426234
6: -0.1372696, 0.1329358, -0.1231250, 0.1199399, -0.2572095, 0.2560608
7: -0.1221348, 0.0669163, -0.1118449, 0.0585164, -0.1806512, 0.1787613
8: -0.0637638, 0.1511623, -0.0540693, 0.1380443, -0.2018080, 0.2052316
9: -0.1183393, 0.0878878, -0.1081993, 0.0813312, -0.1996705, 0.1960872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663964
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663960
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1346938, 0.1093504, -0.1183204, 0.0983966, -0.2330904, 0.2276708
1: -0.0818882, 0.0499607, -0.0731852, 0.0406024, -0.1224905, 0.1231460
2: -0.1161586, 0.1351828, -0.0988272, 0.1213201, -0.2374787, 0.2340101
3: 0.9356807, 1.0214248, 0.9330631, 1.0150847, -0.0794040, 0.0883617
4: -0.0901374, 0.1160593, -0.0794985, 0.1011350, -0.1912724, 0.1955578
5: -0.0686690, 0.1649854, -0.0560150, 0.1499757, -0.2186448, 0.2210005
6: -0.1259882, 0.1221877, -0.1097481, 0.1069696, -0.2329579, 0.2319358
7: -0.1137995, 0.0574316, -0.1015449, 0.0487259, -0.1625254, 0.1589766
8: -0.0559404, 0.1392239, -0.0466571, 0.1235784, -0.1795188, 0.1858809
9: -0.1092096, 0.0823179, -0.0969927, 0.0754595, -0.1846691, 0.1793106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663955
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663959
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1461586, 0.1182791, -0.1224673, 0.1007717, -0.2469303, 0.2407463
1: -0.0891323, 0.0564147, -0.0752174, 0.0428783, -0.1320106, 0.1316321
2: -0.1276770, 0.1461701, -0.1031720, 0.1246065, -0.2522835, 0.2493422
3: 0.9299443, 1.0257020, 0.9415862, 1.0166223, -0.0866780, 0.0841158
4: -0.0986722, 0.1266581, -0.0819490, 0.1049587, -0.2036309, 0.2086071
5: -0.0772323, 0.1762123, -0.0591207, 0.1532771, -0.2305094, 0.2353329
6: -0.1372696, 0.1329358, -0.1137879, 0.1104516, -0.2477212, 0.2467238
7: -0.1221348, 0.0669163, -0.1046996, 0.0497274, -0.1718622, 0.1716159
8: -0.0637638, 0.1511623, -0.0489237, 0.1268218, -0.1905856, 0.2000861
9: -0.1183393, 0.0878878, -0.0999834, 0.0760543, -0.1943936, 0.1878712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685679, upper bound: 0.0689173
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685679, upper bound: 0.0689171
time: 2.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1346938, 0.1093504, -0.1070720, 0.0897472, -0.2244410, 0.2164224
1: -0.0818882, 0.0499607, -0.0667697, 0.0345466, -0.1164348, 0.1167304
2: -0.1161586, 0.1351828, -0.0884381, 0.1098502, -0.2260088, 0.2236209
3: 0.9356807, 1.0214248, 0.9429960, 1.0122710, -0.0765903, 0.0784287
4: -0.0901374, 0.1160593, -0.0700960, 0.0918550, -0.1819925, 0.1861553
5: -0.0686690, 0.1649854, -0.0480115, 0.1393077, -0.2079767, 0.2129969
6: -0.1259882, 0.1221877, -0.0995537, 0.0965336, -0.2225219, 0.2217414
7: -0.1137995, 0.0574316, -0.0936450, 0.0390376, -0.1528371, 0.1510766
8: -0.0559404, 0.1392239, -0.0412567, 0.1111409, -0.1670812, 0.1804806
9: -0.1092096, 0.0823179, -0.0879000, 0.0697001, -0.1789098, 0.1702179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685683, upper bound: 0.0689174
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0685683, upper bound: 0.0689173
time: 1.38 seconds

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

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
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663957, upper bound: 0.0678798
time: 1.13 seconds

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
time: 1.34 seconds

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663978, upper bound: 0.0685738
time: 1.90 seconds

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
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

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
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0663980, upper bound: 0.0685734
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1503326, 0.1219204, -0.1446143, 0.1163805, -0.2667130, 0.2665347
1: -0.0920786, 0.0588718, -0.0873182, 0.0555866, -0.1476653, 0.1461899
2: -0.1319203, 0.1504090, -0.1263232, 0.1441238, -0.2760440, 0.2767323
3: 0.9237472, 1.0275388, 0.9324523, 1.0248680, -0.1011208, 0.0950865
4: -0.1019335, 0.1306586, -0.0971905, 0.1248420, -0.2267755, 0.2278491
5: -0.0804787, 0.1806210, -0.0761771, 0.1742780, -0.2547566, 0.2567980
6: -0.1415397, 0.1370740, -0.1356268, 0.1314833, -0.2730231, 0.2727008
7: -0.1251205, 0.0713215, -0.1210857, 0.0639972, -0.1891177, 0.1924072
8: -0.0671755, 0.1558934, -0.0613885, 0.1492878, -0.2164632, 0.2172819
9: -0.1218003, 0.0904897, -0.1168285, 0.0869657, -0.2087660, 0.2073183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0618728, upper bound: 0.0613659
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 154

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0697505, upper bound: 0.0697510
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0697505, upper bound: 0.0722877
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1468821, 0.1188262, -0.1475226, 0.1186848, -0.2655669, 0.2663488
1: -0.0896243, 0.0568319, -0.0889918, 0.0571861, -0.1468105, 0.1458237
2: -0.1284454, 0.1468029, -0.1290567, 0.1471179, -0.2755633, 0.2758596
3: 0.9299229, 1.0260621, 0.9292635, 1.0256188, -0.0956959, 0.0967987
4: -0.0991228, 0.1273957, -0.0996514, 0.1272454, -0.2263681, 0.2270471
5: -0.0777972, 0.1769331, -0.0782609, 0.1770809, -0.2548782, 0.2551941
6: -0.1380250, 0.1336211, -0.1382954, 0.1342273, -0.2722523, 0.2719164
7: -0.1226776, 0.0674407, -0.1231311, 0.0666182, -0.1892958, 0.1905719
8: -0.0643835, 0.1518541, -0.0628522, 0.1525695, -0.2169530, 0.2147064
9: -0.1188962, 0.0882072, -0.1192007, 0.0885569, -0.2074531, 0.2074080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680438, upper bound: 0.0645179
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772192, upper bound: 0.0772390
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1446143, 0.1168309, -0.1399981, 0.1128858, -0.2575002, 0.2568290
1: -0.0880180, 0.0555866, -0.0847431, 0.0531847, -0.1412026, 0.1403297
2: -0.1263232, 0.1443565, -0.1222440, 0.1392874, -0.2656106, 0.2666005
3: 0.9324523, 1.0253009, 0.9331056, 1.0239480, -0.0914956, 0.0921953
4: -0.0971905, 0.1253221, -0.0932104, 0.1211197, -0.2183102, 0.2185326
5: -0.0761771, 0.1746071, -0.0730576, 0.1701963, -0.2463733, 0.2476647
6: -0.1358530, 0.1314833, -0.1315956, 0.1274239, -0.2632769, 0.2630789
7: -0.1210857, 0.0649650, -0.1178580, 0.0602986, -0.1813843, 0.1828230
8: -0.0626906, 0.1492878, -0.0592426, 0.1444353, -0.2071258, 0.2085304
9: -0.1169780, 0.0869657, -0.1130593, 0.0852285, -0.2022064, 0.2000250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0620512, upper bound: 0.0602145
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0727630, upper bound: 0.0715618
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1475226, 0.1193831, -0.1367555, 0.1103090, -0.2578316, 0.2561386
1: -0.0900766, 0.0571861, -0.0828580, 0.0512561, -0.1413328, 0.1400442
2: -0.1290567, 0.1474789, -0.1188719, 0.1361872, -0.2652439, 0.2663508
3: 0.9292635, 1.0262902, 0.9390615, 1.0227575, -0.0934941, 0.0872287
4: -0.0996514, 0.1279899, -0.0907005, 0.1182864, -0.2179378, 0.2186905
5: -0.0782609, 0.1775912, -0.0705079, 0.1668631, -0.2451241, 0.2480991
6: -0.1386461, 0.1342273, -0.1283700, 0.1241473, -0.2627933, 0.2625972
7: -0.1231311, 0.0681176, -0.1155253, 0.0573379, -0.1804690, 0.1836430
8: -0.0648706, 0.1525695, -0.0573829, 0.1406729, -0.2055435, 0.2099524
9: -0.1194324, 0.0885569, -0.1104733, 0.0830067, -0.2024391, 0.1990302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 172

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0645188, upper bound: 0.0682523
time: 1.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0772537, upper bound: 0.0779935
time: 2.01 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1335625, 0.1087823, -0.1325893, 0.1086746, -0.2422372, 0.2413716
1: -0.0813411, 0.0492157, -0.0809906, 0.0484630, -0.1298041, 0.1302063
2: -0.1145594, 0.1347081, -0.1127485, 0.1349086, -0.2494680, 0.2474566
3: 0.9354841, 1.0205209, 0.9321295, 1.0193111, -0.0838270, 0.0883914
4: -0.0899353, 0.1147656, -0.0903902, 0.1133753, -0.2033106, 0.2051558
5: -0.0675593, 0.1638356, -0.0664112, 0.1629083, -0.2304675, 0.2302468
6: -0.1245976, 0.1209759, -0.1231250, 0.1199399, -0.2445375, 0.2441009
7: -0.1128262, 0.0574686, -0.1118449, 0.0585164, -0.1713426, 0.1693136
8: -0.0550547, 0.1383252, -0.0540693, 0.1380443, -0.1930989, 0.1923945
9: -0.1085517, 0.0816507, -0.1081993, 0.0813312, -0.1898829, 0.1898501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 154
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663941
time: 2.48 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663942
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1217775, 0.1001411, -0.1183204, 0.0983966, -0.2201741, 0.2184615
1: -0.0748023, 0.0426049, -0.0731852, 0.0406024, -0.1154047, 0.1157901
2: -0.1028283, 0.1236047, -0.0988272, 0.1213201, -0.2241484, 0.2224319
3: 0.9416775, 1.0167776, 0.9330631, 1.0150847, -0.0734072, 0.0837145
4: -0.0810646, 0.1045541, -0.0794985, 0.1011350, -0.1821996, 0.1840526
5: -0.0588128, 0.1527228, -0.0560150, 0.1499757, -0.2087885, 0.2087378
6: -0.1133725, 0.1099313, -0.1097481, 0.1069696, -0.2203421, 0.2196795
7: -0.1042879, 0.0488593, -0.1015449, 0.0487259, -0.1530138, 0.1504042
8: -0.0487691, 0.1259860, -0.0466571, 0.1235784, -0.1723475, 0.1726431
9: -0.0993260, 0.0758941, -0.0969927, 0.0754595, -0.1747855, 0.1728868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 154
type: B, layer: 1, pos: 154
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663942
time: 1.31 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663942
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.61 seconds
IS_A1_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663920, upper bound: 0.0663922
IS_A1_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663920, upper bound: 0.0663925
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680075
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663940, upper bound: 0.0680078
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663964
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663960
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663955
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0678797, upper bound: 0.0663959
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0685679, upper bound: 0.0689173
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0685679, upper bound: 0.0689171
IS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0685683, upper bound: 0.0689174
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0685683, upper bound: 0.0689173
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663957, upper bound: 0.0678798
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663957, upper bound: 0.0678798
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663959, upper bound: 0.0678801
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663978, upper bound: 0.0685738
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663978, upper bound: 0.0685734
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663980, upper bound: 0.0685739
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0663980, upper bound: 0.0685734
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0697505, upper bound: 0.0697510
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0697505, upper bound: 0.0722877
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0680438, upper bound: 0.0645179
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0772192, upper bound: 0.0772390
IS_A1_B2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0620512, upper bound: 0.0602145
IS_A1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0727630, upper bound: 0.0715618
IS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0645188, upper bound: 0.0682523
IS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0772537, upper bound: 0.0779935
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663941
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663942
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663942
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 15.61
Output dim: 3, lower bound: -0.0680073, upper bound: 0.0663942
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0712056
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0663983
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0694328
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0769830
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0694074, upper bound: 0.0663973
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0770510
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0691930, upper bound: 0.0690838
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0685683
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0689979, upper bound: 0.0740804
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0680075, upper bound: 0.0707999
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0698467
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0777504
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0663931, upper bound: 0.0748720
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.61
Output dim: 3, lower bound: -0.0769769, upper bound: 0.0784992

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.33 + 596.83 = 601.16 seconds
