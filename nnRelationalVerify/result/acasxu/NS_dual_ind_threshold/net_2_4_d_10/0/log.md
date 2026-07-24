## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 1988.4930817703312


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-423.7106934, 1821.5634766, -423.7106934, 1821.5634766, -2245.2739258, 2245.2739258)
1: (-308.2591858, 1064.1890869, -308.2591858, 1064.1890869, -1372.4482422, 1372.4482422)
2: (-168.3014069, 979.2583618, -168.3014069, 979.2583618, -1147.5595703, 1147.5596924)
3: (-221.9561310, 1465.7564697, -221.9561310, 1465.7564697, -1687.7125244, 1687.7125244)
4: (-312.7283936, 1197.3074951, -312.7283936, 1197.3074951, -1510.0358887, 1510.0358887)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.89 + 1.89 = 3.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5129669

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5059924, upper bound: 1988.5084499
time: 0.56 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5049868, upper bound: 1988.5049868
time: 0.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -1988.5059924, upper bound: 1988.5084499
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -1988.5049868, upper bound: 1988.5049868

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -403.8159180, 1738.0155029, -411.9780273, 1772.3681641, -2176.1840820, 2149.9936523
1: -294.1569519, 1014.0042725, -299.9505615, 1034.5622559, -1328.7192383, 1313.9548340
2: -160.5168152, 932.4663086, -163.7181702, 951.6783447, -1112.1949463, 1096.1839600
3: -211.6352234, 1395.9152832, -215.8756409, 1424.6198730, -1636.2551270, 1611.7908936
4: -298.1479492, 1140.6009521, -304.1505432, 1163.8190918, -1461.9667969, 1444.7514648

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5015597, upper bound: 1988.5036610
time: 0.76 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4994938, upper bound: 1988.5027409
time: 0.73 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -478.9209290, 2032.8426514, -403.7608643, 1734.2376709, -2213.1582031, 2436.6035156
1: -346.6009827, 1216.6601562, -293.1990356, 1013.1207886, -1359.7218018, 1509.8591309
2: -189.4543610, 1125.7376709, -160.2842865, 932.4251099, -1121.8795166, 1286.0219727
3: -247.0906677, 1666.4526367, -210.9848175, 1395.6087646, -1642.6994629, 1877.4375000
4: -350.9189148, 1369.3562012, -297.8642578, 1139.5859375, -1490.5047607, 1667.2204590

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4991109, upper bound: 1988.4985342
time: 0.78 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.45 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1988.5015597, upper bound: 1988.5036610
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1988.4994938, upper bound: 1988.5027409
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1988.4991109, upper bound: 1988.4985342
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -368.6786499, 1589.9730225, -372.4757080, 1606.5086670, -1975.1872559, 1962.4486084
1: -268.2933350, 924.1448975, -272.0793457, 934.0072632, -1202.3005371, 1196.2241211
2: -146.5409088, 849.8211670, -148.3183289, 859.1714478, -1005.7123413, 998.1395264
3: -193.4033508, 1273.3713379, -195.6159210, 1288.1599121, -1481.5632324, 1468.9870605
4: -272.5582275, 1039.4506836, -275.5273743, 1051.3968506, -1323.9550781, 1314.9777832

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004577, upper bound: 1988.5031113
time: 0.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -395.6435852, 1700.4143066, -400.5414124, 1719.6075439, -2115.2512207, 2100.9558105
1: -288.5354309, 992.5007935, -292.0738220, 1004.2655640, -1292.8009033, 1284.5745850
2: -157.3508606, 912.4728394, -159.2739716, 923.2225342, -1080.5731201, 1071.7468262
3: -207.4189606, 1366.0892334, -209.9599762, 1382.5723877, -1589.9913330, 1576.0491943
4: -291.9739685, 1116.3052979, -295.5011597, 1129.6416016, -1421.6152344, 1411.8062744

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.4998192
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.5027409
time: 0.83 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -440.3746033, 1871.2081299, -363.8406067, 1563.7156982, -2004.0903320, 2235.0488281
1: -319.2138977, 1114.7564697, -264.5905457, 910.5772705, -1229.7910156, 1379.3470459
2: -174.5199585, 1030.0721436, -144.5937805, 837.6251831, -1012.1450806, 1174.6658936
3: -227.7361908, 1527.9379883, -190.4566345, 1255.6956787, -1483.4318848, 1718.3946533
4: -323.3014526, 1254.8395996, -268.7847595, 1024.9606934, -1348.2622070, 1523.6242676

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
time: 0.86 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -474.8911743, 2014.5128174, -390.4894104, 1673.8055420, -2148.6965332, 2405.0017090
1: -343.6637268, 1206.2864990, -284.1219482, 977.7543335, -1321.4180908, 1490.4084473
2: -187.8276062, 1116.2609863, -155.1717682, 898.6128540, -1086.4401855, 1271.4326172
3: -244.8308868, 1652.0946045, -204.1697540, 1346.4029541, -1591.2337646, 1856.2642822
4: -347.8052673, 1357.6705322, -287.9757690, 1099.6618652, -1447.4671631, 1645.6462402

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
time: 0.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.98 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.5004577, upper bound: 1988.5031113
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.4998192
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.5027409
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.98
Output dim: 0, lower bound: -1988.4979563, upper bound: 1988.4979563

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -351.9781799, 1513.7548828, -360.2138672, 1551.2678223, -1903.2459717, 1873.9687500
1: -256.4042969, 881.6527710, -263.2996216, 902.6366577, -1159.0406494, 1144.9523926
2: -139.9517975, 810.7362061, -143.4680939, 830.3178711, -970.2696533, 954.2042236
3: -184.5717926, 1215.1569824, -189.1528778, 1244.9495850, -1429.5213623, 1404.3098145
4: -259.9700623, 991.6868896, -266.2778015, 1016.1268311, -1276.0969238, 1257.9647217

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -412.0316467, 1768.0001221, -351.9468384, 1517.1188965, -1929.1503906, 2119.9470215
1: -300.4685059, 1046.7729492, -256.5674744, 883.9263306, -1184.3947754, 1303.3404541
2: -163.5333557, 963.7539673, -139.7291870, 813.3353882, -976.8687134, 1103.4831543
3: -217.1094971, 1438.7313232, -184.6374664, 1219.0483398, -1436.1577148, 1623.3686523
4: -304.5973206, 1175.6849365, -259.8325195, 994.6068726, -1299.2039795, 1435.5174561

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4942457, upper bound: 1988.4986630
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004577, upper bound: 1988.5030986
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -364.3321838, 1573.2664795, -400.5414124, 1719.6075439, -2083.9396973, 1973.8078613
1: -266.4621277, 913.7915039, -292.0738220, 1004.2655640, -1270.7276611, 1205.8653564
2: -145.1979523, 840.4483032, -159.2739716, 923.2225342, -1068.4204102, 999.7221680
3: -191.4476776, 1260.2944336, -209.9599762, 1382.5723877, -1574.0200195, 1470.2543945
4: -269.6809387, 1028.5430908, -295.5011597, 1129.6416016, -1399.3225098, 1324.0441895

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.4998192
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.4998192
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -392.5169678, 1685.5590820, -400.5414124, 1719.6075439, -2112.1245117, 2086.1003418
1: -286.3705139, 984.1212769, -292.0738220, 1004.2655640, -1290.6359863, 1276.1950684
2: -156.1260986, 904.6725464, -159.2739716, 923.2225342, -1079.3485107, 1063.9464111
3: -205.7637482, 1354.7049561, -209.9599762, 1382.5723877, -1588.3361816, 1564.6649170
4: -289.6065979, 1106.8894043, -295.5011597, 1129.6416016, -1419.2481689, 1402.3905029

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.5027409
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.5027409
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -432.6084900, 1839.1585693, -363.8406067, 1563.7156982, -1996.3240967, 2202.9987793
1: -314.6546326, 1094.3081055, -264.5905457, 910.5772705, -1225.2319336, 1358.8986816
2: -171.9007111, 1010.3269043, -144.5937805, 837.6251831, -1009.5258789, 1154.9205322
3: -224.3664856, 1501.0369873, -190.4566345, 1255.6956787, -1480.0621338, 1691.4936523
4: -318.2871094, 1232.0827637, -268.7847595, 1024.9606934, -1343.2478027, 1500.8674316

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4943377, upper bound: 1988.4971533
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4989853, upper bound: 1988.4983201
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -473.0801697, 2005.9555664, -363.8406067, 1563.7156982, -2036.7958984, 2369.7951660
1: -342.3330994, 1201.5083008, -264.5905457, 910.5772705, -1252.9102783, 1466.0988770
2: -187.0977325, 1111.8410645, -144.5937805, 837.6251831, -1024.7229004, 1256.4346924
3: -243.8048096, 1645.5061035, -190.4566345, 1255.6956787, -1499.5004883, 1835.9627686
4: -346.4430542, 1352.2399902, -268.7847595, 1024.9606934, -1371.4038086, 1621.0246582

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4943377, upper bound: 1988.4971533
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4989853, upper bound: 1988.4983201
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -432.6084900, 1839.1585693, -390.4894104, 1673.8055420, -2106.4140625, 2229.6477051
1: -314.6546326, 1094.3081055, -284.1219482, 977.7543335, -1292.4089355, 1378.4300537
2: -171.9007111, 1010.3269043, -155.1717682, 898.6128540, -1070.5135498, 1165.4984131
3: -224.3664856, 1501.0369873, -204.1697540, 1346.4029541, -1570.7694092, 1705.2066650
4: -318.2871094, 1232.0827637, -287.9757690, 1099.6618652, -1417.9489746, 1520.0584717

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4939236, upper bound: 1988.4960666
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4977552, upper bound: 1988.4977552
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -473.0801697, 2005.9555664, -390.4894104, 1673.8055420, -2146.8857422, 2396.4443359
1: -342.3330994, 1201.5083008, -284.1219482, 977.7543335, -1320.0874023, 1485.6302490
2: -187.0977325, 1111.8410645, -155.1717682, 898.6128540, -1085.7102051, 1267.0126953
3: -243.8048096, 1645.5061035, -204.1697540, 1346.4029541, -1590.2077637, 1849.6757812
4: -346.4430542, 1352.2399902, -287.9757690, 1099.6618652, -1446.1047363, 1640.2158203

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4939236, upper bound: 1988.4967277
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4977552, upper bound: 1988.4977552
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.78 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4942457, upper bound: 1988.4986630
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.5004577, upper bound: 1988.5030986
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.4998192
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.4998192
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.5027409
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4986232, upper bound: 1988.5027409
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4943377, upper bound: 1988.4971533
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4989853, upper bound: 1988.4983201
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4943377, upper bound: 1988.4971533
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4989853, upper bound: 1988.4983201
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4939236, upper bound: 1988.4960666
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4977552, upper bound: 1988.4977552
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4939236, upper bound: 1988.4967277
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.78
Output dim: 0, lower bound: -1988.4977552, upper bound: 1988.4977552

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -351.9781799, 1513.7548828, -352.9336548, 1521.8427734, -1873.8208008, 1866.6884766
1: -256.4042969, 881.6527710, -258.3097839, 884.8072510, -1141.2111816, 1139.9625244
2: -139.9517975, 810.7362061, -140.6913605, 813.8074951, -953.7592773, 951.4273682
3: -184.5717926, 1215.1569824, -185.4115143, 1220.3930664, -1404.9648438, 1400.5684814
4: -259.9700623, 991.6868896, -261.0785522, 995.9262695, -1255.8963623, 1252.7652588

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5029942
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -351.9781799, 1513.7548828, -423.8618774, 1800.4079590, -2152.3862305, 1937.6166992
1: -256.4042969, 881.6527710, -308.3460388, 1073.2017822, -1329.6060791, 1189.9987793
2: -139.9517975, 810.7362061, -168.4053345, 991.2348633, -1131.1866455, 979.1414795
3: -184.5717926, 1215.1569824, -219.5643311, 1471.5904541, -1656.1622314, 1434.7213135
4: -259.9700623, 991.6868896, -311.6663208, 1208.2840576, -1468.2541504, 1303.3532715

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5029942
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -402.7874451, 1727.0706787, -391.3357849, 1678.1149902, -2080.9020996, 2118.4064941
1: -293.6874390, 1023.1170044, -285.8509827, 978.5982056, -1272.2856445, 1308.9680176
2: -159.8489227, 942.0859985, -155.9317627, 899.6941528, -1059.5428467, 1098.0178223
3: -212.2712097, 1405.9906006, -206.5194855, 1349.7143555, -1561.9855957, 1612.5101318
4: -297.7755432, 1148.9630127, -290.0588684, 1100.7521973, -1398.5277100, 1439.0218506

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4940495, upper bound: 1988.4983691
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4940495, upper bound: 1988.4986630
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -410.8666382, 1763.0089111, -345.5467224, 1489.0675049, -1899.9339600, 2108.5554199
1: -299.6419373, 1043.9404297, -252.0294342, 867.6687622, -1167.3106689, 1295.9698486
2: -163.0729370, 961.1421509, -137.2048950, 798.4096680, -961.4826050, 1098.3470459
3: -216.4962158, 1434.8363037, -181.2726593, 1196.3142090, -1412.8104248, 1616.1090088
4: -303.7406616, 1172.4832764, -254.9968262, 976.3526001, -1280.0932617, 1427.4799805

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030092
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030986
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -364.3321838, 1573.2664795, -392.5169678, 1685.5590820, -2049.8911133, 1965.7834473
1: -266.4621277, 913.7915039, -286.3705139, 984.1212769, -1250.5833740, 1200.1619873
2: -145.1979523, 840.4483032, -156.1260986, 904.6725464, -1049.8703613, 996.5742798
3: -191.4476776, 1260.2944336, -205.7637482, 1354.7049561, -1546.1525879, 1466.0582275
4: -269.6809387, 1028.5430908, -289.6065979, 1106.8894043, -1376.5703125, 1318.1496582

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4974505, upper bound: 1988.4976398
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4974505, upper bound: 1988.4987908
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -364.3321838, 1573.2664795, -473.0801697, 2005.9555664, -2370.2878418, 2046.3466797
1: -266.4621277, 913.7915039, -342.3330994, 1201.5083008, -1467.9704590, 1256.1246338
2: -145.1979523, 840.4483032, -187.0977325, 1111.8410645, -1257.0389404, 1027.5458984
3: -191.4476776, 1260.2944336, -243.8048096, 1645.5061035, -1836.9537354, 1504.0992432
4: -269.6809387, 1028.5430908, -346.4430542, 1352.2399902, -1621.9208984, 1374.9860840

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4974505, upper bound: 1988.4976398
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4977111, upper bound: 1988.4987908
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -392.5169678, 1685.5590820, -392.5169678, 1685.5590820, -2078.0761719, 2078.0761719
1: -286.3705139, 984.1212769, -286.3705139, 984.1212769, -1270.4915771, 1270.4915771
2: -156.1260986, 904.6725464, -156.1260986, 904.6725464, -1060.7984619, 1060.7984619
3: -205.7637482, 1354.7049561, -205.7637482, 1354.7049561, -1560.4687500, 1560.4687500
4: -289.6065979, 1106.8894043, -289.6065979, 1106.8894043, -1396.4959717, 1396.4959717

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986507, upper bound: 1988.5012944
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -392.5169678, 1685.5590820, -473.0801697, 2005.9555664, -2398.4721680, 2158.6391602
1: -286.3705139, 984.1212769, -342.3330994, 1201.5083008, -1487.8786621, 1326.4543457
2: -156.1260986, 904.6725464, -187.0977325, 1111.8410645, -1267.9671631, 1091.7698975
3: -205.7637482, 1354.7049561, -243.8048096, 1645.5061035, -1851.2698975, 1598.5097656
4: -289.6065979, 1106.8894043, -346.4430542, 1352.2399902, -1641.8465576, 1453.3323975

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4986507, upper bound: 1988.5012944
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -423.5730896, 1798.7497559, -404.2005005, 1728.9105225, -2152.4831543, 2202.9501953
1: -307.9371338, 1070.5532227, -294.6521301, 1007.8382568, -1315.7753906, 1365.2053223
2: -168.2593536, 988.6336060, -161.1680450, 926.6148071, -1094.8741455, 1149.8016357
3: -219.6796417, 1468.0261230, -212.8758698, 1389.8476562, -1609.5270996, 1680.9019775
4: -311.6030273, 1205.1878662, -299.6665344, 1134.1793213, -1445.7823486, 1504.8543701

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4924759, upper bound: 1988.4950928
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4931429, upper bound: 1988.4975991
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -431.4986877, 1834.3636475, -357.5568848, 1536.1671143, -1967.6657715, 2191.9201660
1: -313.8591309, 1091.5969238, -260.1064453, 894.7820435, -1208.6408691, 1351.7033691
2: -171.4603424, 1007.8615112, -142.1008148, 823.1566772, -994.6170044, 1149.9621582
3: -223.7669830, 1497.2625732, -187.1425781, 1233.6350098, -1457.4019775, 1684.4051514
4: -317.4558716, 1229.0292969, -264.0863342, 1007.1880493, -1324.6436768, 1493.1156006

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4977617, upper bound: 1988.4966439
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5009428, upper bound: 1988.5009428
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -464.1206360, 1966.3314209, -404.2005005, 1728.9105225, -2193.0307617, 2370.5319824
1: -335.7048340, 1178.5838623, -294.6521301, 1007.8382568, -1343.5430908, 1473.2358398
2: -183.4922485, 1090.9842529, -161.1680450, 926.6148071, -1110.1068115, 1252.1522217
3: -239.1071167, 1613.7001953, -212.8758698, 1389.8476562, -1628.9548340, 1826.5759277
4: -339.8017883, 1326.4278564, -299.6665344, 1134.1793213, -1473.9810791, 1626.0942383

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4919584, upper bound: 1988.4943754
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4932824, upper bound: 1988.4964004
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4926524, upper bound: 1988.4956101
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -471.9100342, 2000.8161621, -357.5568848, 1536.1671143, -2008.0771484, 2358.3720703
1: -341.4698792, 1198.4521484, -260.1064453, 894.7820435, -1236.2519531, 1458.5585938
2: -186.6303711, 1109.0391846, -142.1008148, 823.1566772, -1009.7869873, 1251.1400146
3: -243.1790771, 1641.3120117, -187.1425781, 1233.6350098, -1476.8139648, 1828.4545898
4: -345.5745239, 1348.8112793, -264.0863342, 1007.1880493, -1352.7623291, 1612.8975830

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4964796, upper bound: 1988.4956654
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4948868, upper bound: 1988.4964626
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -423.5730896, 1798.7497559, -426.7030945, 1818.7218018, -2242.2946777, 2225.4528809
1: -307.9371338, 1070.5532227, -310.3608093, 1070.6892090, -1378.6262207, 1380.9140625
2: -168.2593536, 988.6336060, -169.6423950, 985.7564087, -1154.0156250, 1158.2760010
3: -219.6796417, 1468.0261230, -223.4792938, 1471.3923340, -1691.0717773, 1691.5053711
4: -311.6030273, 1205.1878662, -315.3156738, 1203.5928955, -1515.1959229, 1520.5035400

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4933552, upper bound: 1988.4950840
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -431.4986877, 1834.3636475, -384.8901672, 1649.2077637, -2080.7065430, 2219.2531738
1: -313.8591309, 1091.5969238, -280.0491638, 963.2838135, -1277.1427002, 1371.6461182
2: -171.4603424, 1007.8615112, -152.9328461, 885.3435059, -1056.8038330, 1160.7939453
3: -223.7669830, 1497.2625732, -201.2039032, 1326.3482666, -1550.1152344, 1698.4664307
4: -317.4558716, 1229.0292969, -283.7504578, 1083.4108887, -1400.8665771, 1512.7797852

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969537, upper bound: 1988.4966777
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4956802, upper bound: 1988.4942325
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -464.1206360, 1966.3314209, -426.7030945, 1818.7218018, -2282.8422852, 2393.0344238
1: -335.7048340, 1178.5838623, -310.3608093, 1070.6892090, -1406.3939209, 1488.9447021
2: -183.4922485, 1090.9842529, -169.6423950, 985.7564087, -1169.2481689, 1260.6265869
3: -239.1071167, 1613.7001953, -223.4792938, 1471.3923340, -1710.4995117, 1837.1794434
4: -339.8017883, 1326.4278564, -315.3156738, 1203.5928955, -1543.3946533, 1641.7435303

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4937907, upper bound: 1988.4963621
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -471.9100342, 2000.8161621, -384.8901672, 1649.2077637, -2121.1176758, 2385.7055664
1: -341.4698792, 1198.4521484, -280.0491638, 963.2838135, -1304.7536621, 1478.5012207
2: -186.6303711, 1109.0391846, -152.9328461, 885.3435059, -1071.9736328, 1261.9718018
3: -243.1790771, 1641.3120117, -201.2039032, 1326.3482666, -1569.5273438, 1842.5158691
4: -345.5745239, 1348.8112793, -283.7504578, 1083.4108887, -1428.9853516, 1632.5616455

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969104, upper bound: 1988.4965721
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4964673, upper bound: 1988.4964022
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.84 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5029942
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5029942
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.5004475, upper bound: 1988.5030364
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4940495, upper bound: 1988.4983691
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4940495, upper bound: 1988.4986630
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030092
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030986
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4974505, upper bound: 1988.4976398
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4974505, upper bound: 1988.4987908
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4974505, upper bound: 1988.4976398
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4977111, upper bound: 1988.4987908
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4986507, upper bound: 1988.5012944
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4986507, upper bound: 1988.5012944
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4924759, upper bound: 1988.4950928
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4931429, upper bound: 1988.4975991
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4977617, upper bound: 1988.4966439
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.5009428, upper bound: 1988.5009428
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4932824, upper bound: 1988.4964004
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4926524, upper bound: 1988.4956101
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4948868, upper bound: 1988.4964626
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4933552, upper bound: 1988.4950840
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4969537, upper bound: 1988.4966777
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4956802, upper bound: 1988.4942325
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4937907, upper bound: 1988.4963621
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4969104, upper bound: 1988.4965721
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -1988.4964673, upper bound: 1988.4964022

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -346.8688660, 1494.6314697, -352.9336548, 1521.8427734, -1868.7114258, 1847.5649414
1: -254.0014038, 869.4281616, -258.3097839, 884.8072510, -1138.8083496, 1127.7379150
2: -138.3030243, 799.6902466, -140.6913605, 813.8074951, -952.1105347, 940.3814697
3: -182.1867676, 1199.1530762, -185.4115143, 1220.3930664, -1402.5798340, 1384.5645752
4: -256.4965210, 978.6588135, -261.0785522, 995.9262695, -1252.4227295, 1239.7373047

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4978365, upper bound: 1988.4990012
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5010866, upper bound: 1988.5027350
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -381.6204834, 1638.7261963, -352.9336548, 1521.8427734, -1903.4630127, 1991.6595459
1: -278.5390625, 956.6821289, -258.3097839, 884.8072510, -1163.3461914, 1214.9919434
2: -151.7820435, 879.6323242, -140.6913605, 813.8074951, -965.5895386, 1020.3234863
3: -199.9298401, 1316.8010254, -185.4115143, 1220.3930664, -1420.3228760, 1502.2125244
4: -281.3140869, 1076.1342773, -261.0785522, 995.9262695, -1277.2403564, 1337.2127686

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4978365, upper bound: 1988.5003802
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5010866, upper bound: 1988.5037449
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -346.8688660, 1494.6314697, -423.8618774, 1800.4079590, -2147.2768555, 1918.4934082
1: -254.0014038, 869.4281616, -308.3460388, 1073.2017822, -1327.2031250, 1177.7741699
2: -138.3030243, 799.6902466, -168.4053345, 991.2348633, -1129.5375977, 968.0955811
3: -182.1867676, 1199.1530762, -219.5643311, 1471.5904541, -1653.7772217, 1418.7174072
4: -256.4965210, 978.6588135, -311.6663208, 1208.2840576, -1464.7805176, 1290.3251953

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4995728, upper bound: 1988.5016417
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -381.6204834, 1638.7261963, -423.8618774, 1800.4079590, -2182.0280762, 2062.5881348
1: -278.5390625, 956.6821289, -308.3460388, 1073.2017822, -1351.7408447, 1265.0281982
2: -151.7820435, 879.6323242, -168.4053345, 991.2348633, -1143.0167236, 1048.0374756
3: -199.9298401, 1316.8010254, -219.5643311, 1471.5904541, -1671.5202637, 1536.3653564
4: -281.3140869, 1076.1342773, -311.6663208, 1208.2840576, -1489.5981445, 1387.8005371

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4995728, upper bound: 1988.5030364
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -417.4858704, 1795.6062012, -391.3357849, 1678.1149902, -2095.6005859, 2186.9418945
1: -306.2102661, 1059.4573975, -285.8509827, 978.5982056, -1284.8084717, 1345.3083496
2: -166.3286896, 973.9873657, -155.9317627, 899.6941528, -1066.0228271, 1129.9191895
3: -220.9653320, 1459.1175537, -206.5194855, 1349.7143555, -1570.6796875, 1665.6370850
4: -309.4956665, 1190.3154297, -290.0588684, 1100.7521973, -1410.2478027, 1480.3742676

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -431.4193115, 1847.3771973, -391.3357849, 1678.1149902, -2109.5341797, 2238.7128906
1: -314.3407593, 1100.2963867, -285.8509827, 978.5982056, -1292.9389648, 1386.1473389
2: -171.0572815, 1014.2250366, -155.9317627, 899.6941528, -1070.7510986, 1170.1567383
3: -226.8275299, 1511.4803467, -206.5194855, 1349.7143555, -1576.5418701, 1717.9998779
4: -318.6476135, 1235.6940918, -290.0588684, 1100.7521973, -1419.3997803, 1525.7529297

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -426.4330750, 1835.2335205, -345.5467224, 1489.0675049, -1915.5003662, 2180.7800293
1: -312.8148193, 1082.8702393, -252.0294342, 867.6687622, -1180.4835205, 1334.8996582
2: -169.9058380, 995.4548340, -137.2048950, 798.4096680, -968.3154297, 1132.6596680
3: -225.5690765, 1491.4764404, -181.2726593, 1196.3142090, -1421.8833008, 1672.7491455
4: -316.0650330, 1216.7205811, -254.9968262, 976.3526001, -1292.4176025, 1471.7174072

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030092
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030092
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -441.9566040, 1894.5395508, -345.5467224, 1489.0675049, -1931.0240479, 2240.0859375
1: -322.2830200, 1127.2900391, -252.0294342, 867.6687622, -1189.9516602, 1379.3194580
2: -175.3214264, 1038.9511719, -137.2048950, 798.4096680, -973.7310791, 1176.1560059
3: -232.3989868, 1548.8479004, -181.2726593, 1196.3142090, -1428.7131348, 1730.1206055
4: -326.4559631, 1266.2268066, -254.9968262, 976.3526001, -1302.8085938, 1521.2236328

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030986
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030986
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -355.7401123, 1536.4284668, -388.5358582, 1668.6934814, -2024.4332275, 1924.9641113
1: -259.8900757, 891.9407349, -283.3814392, 973.7002563, -1233.5903320, 1175.3221436
2: -141.6579132, 820.5383301, -154.5082550, 895.0847168, -1036.7425537, 975.0465698
3: -186.8862457, 1229.8178711, -203.6874390, 1340.4265137, -1527.3127441, 1433.5053711
4: -263.1836548, 1003.9241333, -286.6103210, 1095.2443848, -1358.4279785, 1290.5344238

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5008270, upper bound: 1988.5002610
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5051688, upper bound: 1988.5050261
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -354.9854431, 1530.8850098, -383.0639954, 1643.8892822, -1998.8745117, 1913.9489746
1: -258.9927063, 887.3123169, -279.2291260, 959.0688477, -1218.0615234, 1166.5413818
2: -141.2203369, 816.1882324, -152.2661438, 881.7045898, -1022.9248047, 968.4543457
3: -186.4590759, 1224.3547363, -200.7766113, 1320.4748535, -1506.9339600, 1425.1313477
4: -262.4097290, 999.0028687, -282.4956360, 1078.7170410, -1341.1267090, 1281.4985352

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5033513, upper bound: 1988.5007513
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5045801, upper bound: 1988.5021544
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -355.7401123, 1536.4284668, -468.7881470, 1987.3642578, -2343.1044922, 2005.2164307
1: -259.8900757, 891.9407349, -339.1356201, 1190.1069336, -1449.9969482, 1231.0764160
2: -141.6579132, 820.5383301, -185.3605957, 1101.2701416, -1242.9281006, 1005.8989258
3: -186.8862457, 1229.8178711, -241.5397797, 1629.9978027, -1816.8840332, 1471.3575439
4: -263.1836548, 1003.9241333, -343.1902466, 1339.4356689, -1602.6193848, 1347.1143799

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -354.9854431, 1530.8850098, -464.4246826, 1968.1245117, -2323.1093750, 1995.3096924
1: -258.9927063, 887.3123169, -335.7988586, 1178.6876221, -1437.6801758, 1223.1108398
2: -141.2203369, 816.1882324, -183.5575409, 1091.0032959, -1232.2235107, 999.7457275
3: -186.4590759, 1224.3547363, -239.2728119, 1614.0750732, -1800.5341797, 1463.6275635
4: -262.4097290, 999.0028687, -339.9553528, 1326.6611328, -1589.0708008, 1338.9582520

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -391.3260803, 1680.3891602, -392.5169678, 1685.5590820, -2076.8852539, 2072.9062500
1: -285.4822693, 981.1067505, -286.3705139, 984.1212769, -1269.6035156, 1267.4772949
2: -155.6468201, 901.9078979, -156.1260986, 904.6725464, -1060.3192139, 1058.0338135
3: -205.1308746, 1350.5274658, -205.7637482, 1354.7049561, -1559.8358154, 1556.2912598
4: -288.7197571, 1103.4877930, -289.6065979, 1106.8894043, -1395.6091309, 1393.0943604

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4983075, upper bound: 1988.4983428
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4983075, upper bound: 1988.4983428
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -400.1287842, 1715.9349365, -388.5094299, 1667.9791260, -2068.1079102, 2104.4443359
1: -291.1745911, 1006.7157593, -283.3233032, 974.3171387, -1265.4916992, 1290.0389404
2: -158.9143677, 926.2129517, -154.5031433, 895.7128296, -1054.6271973, 1080.7159424
3: -209.6903534, 1384.7171631, -203.6512756, 1341.0545654, -1550.7447510, 1588.3684082
4: -295.0205994, 1132.1480713, -286.6466370, 1095.7784424, -1390.7988281, 1418.7946777

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954071
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954423
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -391.3260803, 1680.3891602, -473.0801697, 2005.9555664, -2397.2812500, 2153.4692383
1: -285.4822693, 981.1067505, -342.3330994, 1201.5083008, -1486.9906006, 1323.4398193
2: -155.6468201, 901.9078979, -187.0977325, 1111.8410645, -1267.4877930, 1089.0052490
3: -205.1308746, 1350.5274658, -243.8048096, 1645.5061035, -1850.6369629, 1594.3322754
4: -288.7197571, 1103.4877930, -346.4430542, 1352.2399902, -1640.9597168, 1449.9305420

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -400.1287842, 1715.9349365, -468.8900757, 1987.5688477, -2387.6977539, 2184.8249512
1: -291.1745911, 1006.7157593, -339.1936340, 1191.2513428, -1482.4259033, 1345.9089355
2: -158.9143677, 926.2129517, -185.4124756, 1102.4560547, -1261.3703613, 1111.6253662
3: -209.6903534, 1384.7171631, -241.5973206, 1631.2910156, -1840.9813232, 1626.3144531
4: -295.0205994, 1132.1480713, -343.3616333, 1340.6289062, -1635.6495361, 1475.5097656

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -408.1951904, 1730.2661133, -394.1461792, 1685.6818848, -2093.8769531, 2124.4121094
1: -296.9629211, 1031.4068604, -287.6713257, 982.6719971, -1279.6347656, 1319.0781250
2: -162.2163849, 952.8221436, -157.3017731, 903.3204956, -1065.5368652, 1110.1239014
3: -211.4069519, 1413.9222412, -207.6497192, 1354.9459229, -1566.3529053, 1621.5718994
4: -300.0684509, 1161.1396484, -292.2830505, 1105.7171631, -1405.7856445, 1453.4227295

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1988.4904884, upper bound: 1988.4914797
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4917592, upper bound: 1988.4943144
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4917592, upper bound: 1988.4950928
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -419.0800476, 1783.3748779, -396.5867004, 1696.8767090, -2115.9567871, 2179.9616699
1: -305.4845886, 1055.1022949, -289.2713623, 987.4564819, -1292.9410400, 1344.3736572
2: -166.7750702, 973.2833252, -158.2303162, 907.5897217, -1074.3647461, 1131.5134277
3: -217.5997009, 1448.2976074, -208.9225311, 1362.2449951, -1579.8447266, 1657.2200928
4: -308.5199280, 1188.0372314, -294.1614380, 1111.3333740, -1419.8532715, 1482.1986084

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1988.4905813, upper bound: 1988.4905705
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1988.4886206, upper bound: 1988.4894046
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -415.8579102, 1764.2965088, -345.6874084, 1484.6220703, -1900.4799805, 2109.9838867
1: -302.6210632, 1051.7777100, -251.7953186, 863.5180664, -1166.1391602, 1303.5729980
2: -165.2890472, 971.4323730, -137.5294037, 794.1294556, -959.4185181, 1108.9617920
3: -215.3279266, 1442.2077637, -181.0005493, 1190.9803467, -1406.3082275, 1623.2082520
4: -305.7166138, 1184.1370850, -255.3735199, 972.1371460, -1277.8537598, 1439.5103760

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4977617, upper bound: 1988.4966439
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4977617, upper bound: 1988.4966439
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -426.5001831, 1817.5213623, -350.8555908, 1508.1077881, -1934.6079102, 2168.3767090
1: -311.1097107, 1075.1522217, -255.3709717, 877.5393677, -1188.6490479, 1330.5231934
2: -169.7918854, 991.6198730, -139.5103912, 807.1340332, -976.9259033, 1131.1301270
3: -221.4493713, 1476.1881104, -183.6592712, 1210.1362305, -1431.5854492, 1659.8474121
4: -314.0539551, 1210.8052979, -259.2372742, 987.7756958, -1301.8293457, 1470.0426025

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5009428, upper bound: 1988.5009428
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5009428, upper bound: 1988.5009428
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -462.8242798, 1960.5451660, -404.2005005, 1728.9105225, -2191.7346191, 2364.7453613
1: -334.7291260, 1175.2923584, -294.6521301, 1007.8382568, -1342.5673828, 1469.9444580
2: -182.9704895, 1087.9645996, -161.1680450, 926.6148071, -1109.5850830, 1249.1323242
3: -238.4151001, 1609.1397705, -212.8758698, 1389.8476562, -1628.2626953, 1822.0156250
4: -338.8371887, 1322.7147217, -299.6665344, 1134.1793213, -1473.0164795, 1622.3812256

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4920398, upper bound: 1988.4956101
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4926524, upper bound: 1988.4956101
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -473.3475342, 2004.2276611, -400.1947327, 1711.3116455, -2184.6591797, 2404.4218750
1: -341.7003479, 1206.4642334, -291.6013184, 997.9644775, -1339.6647949, 1498.0653076
2: -186.9273529, 1117.6378174, -159.5440063, 917.5830078, -1104.5103760, 1277.1800537
3: -243.7795410, 1650.5695801, -210.7657776, 1376.1429443, -1619.9222412, 1861.3353271
4: -346.3622131, 1357.4237061, -296.7127686, 1122.9956055, -1469.3577881, 1653.7282715

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4920398, upper bound: 1988.4956101
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4926524, upper bound: 1988.4956101
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -470.6290894, 1995.1164551, -357.5568848, 1536.1671143, -2006.7961426, 2352.6730957
1: -340.5061646, 1195.2143555, -260.1064453, 894.7820435, -1235.2882080, 1455.3208008
2: -186.1143341, 1106.0721436, -142.1008148, 823.1566772, -1009.2709961, 1248.1728516
3: -242.4943848, 1636.8231201, -187.1425781, 1233.6350098, -1476.1292725, 1823.9656982
4: -344.6201172, 1345.1574707, -264.0863342, 1007.1880493, -1351.8081055, 1609.2437744

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -481.1739502, 2039.6989746, -353.5920105, 1518.7343750, -1999.9083252, 2393.2900391
1: -347.5114136, 1226.5999756, -257.0634766, 885.0989990, -1232.6103516, 1483.6630859
2: -190.0856781, 1135.9549561, -140.4833374, 814.3125610, -1004.3981323, 1276.3565674
3: -247.8889618, 1678.5717773, -185.0486603, 1220.1240234, -1468.0129395, 1863.6202393
4: -352.1942749, 1380.1081543, -261.1489258, 996.2033691, -1348.3975830, 1641.0402832

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -422.4085388, 1793.6374512, -426.7030945, 1818.7218018, -2241.1301270, 2220.3405762
1: -307.0640564, 1067.6486816, -310.3608093, 1070.6892090, -1377.7531738, 1378.0095215
2: -167.7931213, 985.9752197, -169.6423950, 985.7564087, -1153.5491943, 1155.6176758
3: -219.0599060, 1463.9981689, -223.4792938, 1471.3923340, -1690.4520264, 1687.4774170
4: -310.7373047, 1201.9144287, -315.3156738, 1203.5928955, -1514.3302002, 1517.2301025

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -431.0650024, 1829.7780762, -422.7162781, 1801.2088623, -2232.2739258, 2252.4941406
1: -312.6660461, 1093.6282959, -307.3578491, 1060.8857422, -1373.5516357, 1400.9858398
2: -170.9594269, 1010.8770142, -168.0385132, 976.7705688, -1147.7299805, 1178.9154053
3: -223.4806519, 1498.3857422, -221.3791046, 1457.7868652, -1681.2675781, 1719.7648926
4: -316.8619385, 1230.9478760, -312.3869934, 1192.4697266, -1509.3316650, 1543.3348389

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -430.3626709, 1829.3861084, -384.8901672, 1649.2077637, -2079.5703125, 2214.2753906
1: -313.0073242, 1088.7613525, -280.0491638, 963.2838135, -1276.2911377, 1368.8104248
2: -171.0060883, 1005.2631836, -152.9328461, 885.3435059, -1056.3496094, 1158.1958008
3: -223.1625519, 1493.3292236, -201.2039032, 1326.3482666, -1549.5108643, 1694.5330811
4: -316.6130981, 1225.8286133, -283.7504578, 1083.4108887, -1400.0239258, 1509.5789795

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969537, upper bound: 1988.4964663
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969537, upper bound: 1988.4965292
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -438.7783508, 1864.9650879, -380.9886169, 1632.1296387, -2070.9079590, 2245.9536133
1: -318.4289551, 1114.1837158, -277.0957336, 953.7605591, -1272.1894531, 1391.2794189
2: -174.0759125, 1029.6795654, -151.3536987, 876.6362305, -1050.7121582, 1181.0329590
3: -227.4496918, 1526.8653564, -199.1534576, 1313.1629639, -1540.6125488, 1726.0187988
4: -322.5899658, 1254.1389160, -280.8736572, 1072.6191406, -1395.2091064, 1535.0124512

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4956802, upper bound: 1988.4942325
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4956802, upper bound: 1988.4942325
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -462.8242798, 1960.5451660, -426.7030945, 1818.7218018, -2281.5461426, 2387.2482910
1: -334.7291260, 1175.2923584, -310.3608093, 1070.6892090, -1405.4180908, 1485.6531982
2: -182.9704895, 1087.9645996, -169.6423950, 985.7564087, -1168.7263184, 1257.6069336
3: -238.4151001, 1609.1397705, -223.4792938, 1471.3923340, -1709.8073730, 1832.6190186
4: -338.8371887, 1322.7147217, -315.3156738, 1203.5928955, -1542.4300537, 1638.0303955

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -473.3475342, 2004.2276611, -422.7162781, 1801.2088623, -2274.5563965, 2426.9436035
1: -341.7003479, 1206.4642334, -307.3578491, 1060.8857422, -1402.5858154, 1513.8218994
2: -186.9273529, 1117.6378174, -168.0385132, 976.7705688, -1163.6978760, 1285.6762695
3: -243.7795410, 1650.5695801, -221.3791046, 1457.7868652, -1701.5661621, 1871.9487305
4: -346.3622131, 1357.4237061, -312.3869934, 1192.4697266, -1538.8319092, 1669.8106689

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -470.6290894, 1995.1164551, -384.8901672, 1649.2077637, -2119.8366699, 2380.0063477
1: -340.5061646, 1195.2143555, -280.0491638, 963.2838135, -1303.7900391, 1475.2634277
2: -186.1143341, 1106.0721436, -152.9328461, 885.3435059, -1071.4576416, 1259.0046387
3: -242.4943848, 1636.8231201, -201.2039032, 1326.3482666, -1568.8425293, 1838.0269775
4: -344.6201172, 1345.1574707, -283.7504578, 1083.4108887, -1428.0310059, 1628.9079590

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969104, upper bound: 1988.4964853
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969104, upper bound: 1988.4964853
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -481.1739502, 2039.6989746, -380.9886169, 1632.1296387, -2113.3037109, 2420.6872559
1: -347.5114136, 1226.5999756, -277.0957336, 953.7605591, -1301.2719727, 1503.6955566
2: -190.0856781, 1135.9549561, -151.3536987, 876.6362305, -1066.7219238, 1287.3085938
3: -247.8889618, 1678.5717773, -199.1534576, 1313.1629639, -1561.0518799, 1877.7252197
4: -352.1942749, 1380.1081543, -280.8736572, 1072.6191406, -1424.8134766, 1660.9816895

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4964673, upper bound: 1988.4964019
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4964673, upper bound: 1988.4964019
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.60 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4978365, upper bound: 1988.4990012
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5010866, upper bound: 1988.5027350
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4978365, upper bound: 1988.5003802
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5010866, upper bound: 1988.5037449
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030092
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030092
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030986
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5004492, upper bound: 1988.5030986
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5008270, upper bound: 1988.5002610
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5051688, upper bound: 1988.5050261
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5033513, upper bound: 1988.5007513
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5045801, upper bound: 1988.5021544
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4983075, upper bound: 1988.4983428
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4983075, upper bound: 1988.4983428
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954071
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954423
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4968230, upper bound: 1988.4981337
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4917592, upper bound: 1988.4943144
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4917592, upper bound: 1988.4950928
NS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4905813, upper bound: 1988.4905705
NS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4886206, upper bound: 1988.4894046
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4977617, upper bound: 1988.4966439
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4977617, upper bound: 1988.4966439
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5009428, upper bound: 1988.5009428
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.5009428, upper bound: 1988.5009428
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4920398, upper bound: 1988.4956101
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4926524, upper bound: 1988.4956101
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4920398, upper bound: 1988.4956101
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4926524, upper bound: 1988.4956101
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4942325, upper bound: 1988.4956802
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4924788, upper bound: 1988.4938111
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4969537, upper bound: 1988.4964663
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4969537, upper bound: 1988.4965292
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4956802, upper bound: 1988.4942325
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4956802, upper bound: 1988.4942325
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4932261, upper bound: 1988.4959226
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4969104, upper bound: 1988.4964853
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4969104, upper bound: 1988.4964853
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4964673, upper bound: 1988.4964019
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.60
Output dim: 0, lower bound: -1988.4964673, upper bound: 1988.4964019

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -336.2080078, 1448.2994385, -334.2420654, 1440.6687012, -1776.8762207, 1782.5415039
1: -246.6555328, 841.4644775, -245.4104919, 836.4733276, -1083.1286621, 1086.8750000
2: -134.2317657, 773.2816772, -133.5367737, 768.5191650, -902.7509155, 906.8184814
3: -176.6567078, 1160.4377441, -175.6851044, 1153.3720703, -1330.0285645, 1336.1228027
4: -248.6660767, 947.3320312, -247.2867889, 941.7741699, -1190.4401855, 1194.6187744

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4978685, upper bound: 1988.4990012
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4968572, upper bound: 1988.4975230
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -339.8196716, 1464.9111328, -347.2242737, 1499.6037598, -1839.4233398, 1812.1353760
1: -249.0110626, 851.3541870, -254.9758453, 868.9857788, -1117.9968262, 1106.3300781
2: -135.5733032, 782.9483643, -138.6902313, 798.8509521, -934.4242554, 921.6384277
3: -178.5104370, 1174.4779053, -182.5312347, 1199.3989258, -1377.9094238, 1357.0089111
4: -251.4056091, 958.3096313, -257.1356812, 978.3206177, -1229.7261963, 1215.4451904

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4992993, upper bound: 1988.5008999
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5007542, upper bound: 1988.5012127
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -373.6232910, 1603.6319580, -334.2420654, 1440.6687012, -1814.2917480, 1937.8740234
1: -272.8569946, 936.5432129, -245.4104919, 836.4733276, -1109.3303223, 1181.9537354
2: -148.6526031, 861.2494507, -133.5367737, 768.5191650, -917.1717529, 994.7861938
3: -195.6856689, 1289.0097656, -175.6851044, 1153.3720703, -1349.0574951, 1464.6948242
4: -275.3379517, 1053.6320801, -247.2867889, 941.7741699, -1217.1119385, 1300.9187012

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4983737, upper bound: 1988.5003802
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4983737, upper bound: 1988.5003802
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -370.0754395, 1590.5480957, -347.2242737, 1499.6037598, -1869.6791992, 1937.7723389
1: -270.3694153, 925.9437866, -254.9758453, 868.9857788, -1139.3551025, 1180.9196777
2: -147.3458557, 850.8115845, -138.6902313, 798.8509521, -946.1967163, 989.5016479
3: -193.9704742, 1275.0213623, -182.5312347, 1199.3989258, -1393.3693848, 1457.5523682
4: -273.0272522, 1041.4501953, -257.1356812, 978.3206177, -1251.3479004, 1298.5859375

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5012426, upper bound: 1988.5036389
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5010279, upper bound: 1988.5027504
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -426.4330750, 1835.2335205, -338.4541626, 1460.7221680, -1887.1549072, 2173.6875000
1: -312.8148193, 1082.8702393, -247.2367096, 850.3599854, -1163.1748047, 1330.1069336
2: -169.9058380, 995.4548340, -134.5234375, 782.3606567, -952.2663574, 1129.9782715
3: -225.5690765, 1491.4764404, -177.6595306, 1172.3897705, -1397.9588623, 1669.1359863
4: -316.0650330, 1216.7205811, -249.9606781, 956.7592163, -1272.8242188, 1466.6812744

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4975440, upper bound: 1988.4979137
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4998739, upper bound: 1988.5013760
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -426.4330750, 1835.2335205, -405.0321960, 1718.9606934, -2145.3937988, 2240.2651367
1: -312.8148193, 1082.8702393, -294.2910767, 1026.0321045, -1338.8469238, 1377.1613770
2: -169.9058380, 995.4548340, -160.6315002, 947.3333740, -1117.2391357, 1156.0863037
3: -225.5690765, 1491.4764404, -209.7732239, 1407.1624756, -1632.7315674, 1701.2496338
4: -316.0650330, 1216.7205811, -297.5987244, 1154.6407471, -1470.7058105, 1514.3193359

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4975440, upper bound: 1988.4979137
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4998739, upper bound: 1988.5013760
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -441.9566040, 1894.5395508, -338.4541626, 1460.7221680, -1902.6787109, 2232.9934082
1: -322.2830200, 1127.2900391, -247.2367096, 850.3599854, -1172.6430664, 1374.5266113
2: -175.3214264, 1038.9511719, -134.5234375, 782.3606567, -957.6820068, 1173.4744873
3: -232.3989868, 1548.8479004, -177.6595306, 1172.3897705, -1404.7888184, 1726.5074463
4: -326.4559631, 1266.2268066, -249.9606781, 956.7592163, -1283.2152100, 1516.1875000

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4975187, upper bound: 1988.4998628
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4980001, upper bound: 1988.4997682
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -441.9566040, 1894.5395508, -405.0321960, 1718.9606934, -2160.9172363, 2299.5710449
1: -322.2830200, 1127.2900391, -294.2910767, 1026.0321045, -1348.3150635, 1421.5810547
2: -175.3214264, 1038.9511719, -160.6315002, 947.3333740, -1122.6546631, 1199.5825195
3: -232.3989868, 1548.8479004, -209.7732239, 1407.1624756, -1639.5615234, 1758.6210938
4: -326.4559631, 1266.2268066, -297.5987244, 1154.6407471, -1481.0966797, 1563.8255615

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4976557, upper bound: 1988.4998628
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4980001, upper bound: 1988.4997682
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -325.8724365, 1402.4206543, -301.2319031, 1286.9963379, -1612.8687744, 1703.6523438
1: -239.1589355, 814.0640869, -219.0728149, 760.2135010, -999.3724365, 1033.1369629
2: -130.1011505, 747.9392090, -119.8096466, 699.8145752, -829.9156494, 867.7487793
3: -171.3838501, 1121.8493652, -157.7566833, 1045.2170410, -1216.6008301, 1279.6059570
4: -241.1619415, 916.7626953, -222.6598053, 854.6519775, -1095.8139648, 1139.4224854

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -352.8089600, 1523.0991211, -379.2702637, 1627.9790039, -1980.7879639, 1902.3692627
1: -257.7261963, 883.6748047, -276.8346252, 946.1523438, -1203.8785400, 1160.5093994
2: -140.4607697, 812.9912720, -150.8728790, 869.5496826, -1010.0104370, 963.8641357
3: -185.4230652, 1218.1845703, -199.0279388, 1303.4321289, -1488.8552246, 1417.2124023
4: -261.0544128, 994.4271240, -279.9602356, 1064.4322510, -1325.4864502, 1274.3873291

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5051688, upper bound: 1988.5050261
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5050652, upper bound: 1988.5041147
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -326.4569092, 1402.9235840, -294.3883972, 1256.7868652, -1583.2435303, 1697.3120117
1: -239.2560425, 814.0557861, -213.9029694, 742.4011230, -981.6571655, 1027.9587402
2: -130.2108459, 747.8974609, -117.0395584, 683.3484497, -813.5593262, 864.9370117
3: -171.6954498, 1121.9940186, -154.0379486, 1020.6480103, -1192.3433838, 1276.0318604
4: -241.3914032, 916.8622437, -217.5162048, 834.5057373, -1075.8970947, 1134.3784180

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -352.1776428, 1518.2467041, -373.5631409, 1601.6455078, -1953.8231201, 1891.8096924
1: -256.9249878, 879.4350586, -272.5353088, 930.0523682, -1186.9771729, 1151.9703369
2: -140.0575256, 809.0692139, -148.5357056, 854.7460327, -994.8035889, 957.6049194
3: -185.0237274, 1213.2049561, -195.9908447, 1281.1732178, -1466.1968994, 1409.1958008
4: -260.3769226, 989.9727783, -275.6342163, 1046.5703125, -1306.9472656, 1265.6069336

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -391.3260803, 1680.3891602, -391.3260803, 1680.3891602, -2071.7153320, 2071.7153320
1: -285.4822693, 981.1067505, -285.4822693, 981.1067505, -1266.5889893, 1266.5889893
2: -155.6468201, 901.9078979, -155.6468201, 901.9078979, -1057.5545654, 1057.5545654
3: -205.1308746, 1350.5274658, -205.1308746, 1350.5274658, -1555.6583252, 1555.6583252
4: -288.7197571, 1103.4877930, -288.7197571, 1103.4877930, -1392.2073975, 1392.2073975

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4976898, upper bound: 1988.4997094
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969939, upper bound: 1988.4979351
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -391.3260803, 1680.3891602, -400.1287842, 1715.9349365, -2107.2609863, 2080.5180664
1: -285.4822693, 981.1067505, -291.1745911, 1006.7157593, -1292.1978760, 1272.2813721
2: -155.6468201, 901.9078979, -158.9143677, 926.2129517, -1081.8597412, 1060.8221436
3: -205.1308746, 1350.5274658, -209.6903534, 1384.7171631, -1589.8480225, 1560.2177734
4: -288.7197571, 1103.4877930, -295.0205994, 1132.1480713, -1420.8677979, 1398.5081787

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4976898, upper bound: 1988.4997094
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969939, upper bound: 1988.4979351
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -375.8002014, 1608.5715332, -300.6209717, 1283.8837891, -1659.6839600, 1909.1922607
1: -273.8834839, 946.3248901, -218.5723724, 759.5706177, -1033.4539795, 1164.8972168
2: -149.3218842, 871.3676758, -119.5756989, 699.2446899, -848.5665894, 990.9433594
3: -196.8097534, 1300.9754639, -157.3782043, 1044.1134033, -1240.9230957, 1458.3536377
4: -276.8919983, 1064.5169678, -222.2668152, 853.7442017, -1130.6362305, 1286.7838135

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954071
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954071
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -393.8724365, 1687.8905029, -379.1794434, 1626.8120117, -2020.6843262, 2067.0698242
1: -286.7552490, 987.4489136, -276.7415161, 945.9796753, -1232.7348633, 1264.1899414
2: -156.4595490, 908.1203003, -150.8460846, 869.4310913, -1025.8905029, 1058.9661865
3: -206.5429230, 1358.2659912, -198.9627991, 1303.0715332, -1509.6145020, 1557.2287598
4: -290.4766541, 1110.7780762, -279.9269104, 1064.3232422, -1354.7998047, 1390.7048340

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954423
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4954071, upper bound: 1988.4954423
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -391.3260803, 1680.3891602, -471.7948914, 2000.2430420, -2391.5688477, 2152.1838379
1: -285.4822693, 981.1067505, -341.3650818, 1198.2525635, -1483.7348633, 1322.4718018
2: -155.6468201, 901.9078979, -186.5798035, 1108.8553467, -1264.5021973, 1088.4873047
3: -205.1308746, 1350.5274658, -243.1178589, 1640.9943848, -1846.1252441, 1593.6452637
4: -288.7197571, 1103.4877930, -345.4853210, 1348.5664062, -1637.2861328, 1448.9730225

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4975790, upper bound: 1988.4996124
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969454, upper bound: 1988.4982960
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -391.3260803, 1680.3891602, -482.3102417, 2044.5889893, -2435.9150391, 2162.6994629
1: -285.4822693, 981.1067505, -348.3421326, 1229.5478516, -1515.0301514, 1329.4488525
2: -155.6468201, 901.9078979, -190.5377502, 1138.6625977, -1294.3094482, 1092.4455566
3: -205.1308746, 1350.5274658, -248.4920959, 1682.6179199, -1887.7487793, 1599.0195312
4: -288.7197571, 1103.4877930, -353.0332336, 1383.4182129, -1672.1379395, 1456.5209961

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4975790, upper bound: 1988.4996124
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4969454, upper bound: 1988.4982960
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -400.1287842, 1715.9349365, -471.7948914, 2000.2430420, -2400.3715820, 2187.7297363
1: -291.1745911, 1006.7157593, -341.3650818, 1198.2525635, -1489.4271240, 1348.0804443
2: -158.9143677, 926.2129517, -186.5798035, 1108.8553467, -1267.7696533, 1112.7927246
3: -209.6903534, 1384.7171631, -243.1178589, 1640.9943848, -1850.6846924, 1627.8349609
4: -295.0205994, 1132.1480713, -345.4853210, 1348.5664062, -1643.5870361, 1477.6334229

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4953404, upper bound: 1988.4964531
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4953587, upper bound: 1988.4954596
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -400.1287842, 1715.9349365, -482.3102417, 2044.5889893, -2444.7177734, 2198.2451172
1: -291.1745911, 1006.7157593, -348.3421326, 1229.5478516, -1520.7224121, 1355.0577393
2: -158.9143677, 926.2129517, -190.5377502, 1138.6625977, -1297.5769043, 1116.7507324
3: -209.6903534, 1384.7171631, -248.4920959, 1682.6179199, -1892.3082275, 1633.2092285
4: -295.0205994, 1132.1480713, -353.0332336, 1383.4182129, -1678.4388428, 1485.1812744

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4953404, upper bound: 1988.4964531
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4953587, upper bound: 1988.4954596
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -408.1951904, 1730.2661133, -386.8474426, 1654.3623047, -2062.5573730, 2117.1132812
1: -296.9629211, 1031.4068604, -282.6605530, 964.5530396, -1261.5159912, 1314.0673828
2: -162.2163849, 952.8221436, -154.5096130, 886.2257080, -1048.4418945, 1107.3317871
3: -211.4069519, 1413.9222412, -203.8850555, 1329.5994873, -1541.0064697, 1617.8072510
4: -300.0684509, 1161.1396484, -287.0013428, 1084.9763184, -1385.0447998, 1448.1408691

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1988.4908129, upper bound: 1988.4908129
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4908129, upper bound: 1988.4946386
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -408.1951904, 1730.2661133, -397.5685120, 1703.0267334, -2111.2219238, 2127.8344727
1: -296.9629211, 1031.4068604, -290.5296021, 989.4019775, -1286.3647461, 1321.9365234
2: -162.2163849, 952.8221436, -158.7554474, 909.2700806, -1071.4862061, 1111.5776367
3: -211.4069519, 1413.9222412, -209.4470215, 1365.2443848, -1576.6513672, 1623.3692627
4: -300.0684509, 1161.1396484, -294.9172974, 1113.6965332, -1413.7650146, 1456.0568848

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1988.4908129, upper bound: 1988.4911399
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4908129, upper bound: 1988.4950928
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -415.8579102, 1764.2965088, -347.4818420, 1499.6597900, -1915.5177002, 2111.7783203
1: -302.6210632, 1051.7777100, -254.6630859, 869.8602295, -1172.4813232, 1306.4407959
2: -165.2890472, 971.4323730, -138.6759796, 799.6108398, -964.8999023, 1110.1081543
3: -215.3279266, 1442.2077637, -182.6663055, 1199.8677979, -1415.1955566, 1624.8739014
4: -305.7166138, 1184.1370850, -257.2517395, 979.2994385, -1285.0161133, 1441.3887939

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -415.8579102, 1764.2965088, -418.8427124, 1778.0511475, -2193.9091797, 2183.1386719
1: -302.6210632, 1051.7777100, -304.7229919, 1059.8736572, -1362.4947510, 1356.5007324
2: -165.2890472, 971.4323730, -166.4434357, 978.9017944, -1144.1904297, 1137.8757324
3: -215.3279266, 1442.2077637, -216.8949738, 1453.3201904, -1668.6480713, 1659.1027832
4: -305.7166138, 1184.1370850, -307.9479370, 1193.2429199, -1498.9594727, 1492.0848389

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -426.5001831, 1817.5213623, -352.5950928, 1522.7591553, -1949.2592773, 2170.1157227
1: -311.1097107, 1075.1522217, -258.1372375, 883.6760864, -1194.7857666, 1333.2894287
2: -169.7918854, 991.6198730, -140.6020966, 812.6431885, -982.4349976, 1132.2219238
3: -221.4493713, 1476.1881104, -185.2773895, 1218.9412842, -1440.3905029, 1661.4654541
4: -314.0539551, 1210.8052979, -261.0605164, 994.6870728, -1308.7404785, 1471.8658447

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4971796, upper bound: 1988.4979720
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4966345, upper bound: 1988.4966345
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -426.5001831, 1817.5213623, -419.3099060, 1783.7752686, -2210.2753906, 2236.8312988
1: -311.1097107, 1075.1522217, -305.3355103, 1059.2249756, -1370.3347168, 1380.4876709
2: -169.7918854, 991.6198730, -166.7719269, 977.5511475, -1147.3430176, 1158.3918457
3: -221.4493713, 1476.1881104, -217.4816742, 1453.3621826, -1674.8114014, 1693.6697998
4: -314.0539551, 1210.8052979, -308.6198730, 1192.6828613, -1506.7366943, 1519.4251709

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4971796, upper bound: 1988.4979720
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4966345, upper bound: 1988.4966345
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -462.8242798, 1960.5451660, -403.0615845, 1723.9637451, -2186.7880859, 2363.6062012
1: -334.7291260, 1175.2923584, -293.7975769, 1004.9770508, -1339.7061768, 1469.0897217
2: -182.9704895, 1087.9645996, -160.7115479, 923.9638062, -1106.9342041, 1248.6760254
3: -238.4151001, 1609.1397705, -212.2805328, 1385.9051514, -1624.3201904, 1821.4202881
4: -338.8371887, 1322.7147217, -298.8309021, 1130.9335938, -1469.7707520, 1621.5455322

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1988.4925307, upper bound: 1988.4930631
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4925307, upper bound: 1988.4964004
time: 0.95 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.78 + 417.52 = 421.30 seconds
