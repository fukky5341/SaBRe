## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
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
execution time: IAR + RelationalAnalysis = 1.84 + 1.89 = 3.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5129669

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5129653, upper bound: 1988.5128598
time: 0.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5129669
time: 0.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -1988.5129653, upper bound: 1988.5128598
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5129669

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -427.5719910, 1845.1744385, -412.0866699, 1770.5843506, -2198.1562500, 2257.2609863
1: -310.5957642, 1078.4002686, -300.0654297, 1035.0330811, -1345.6289062, 1378.4656982
2: -169.6991425, 992.0928955, -163.7318268, 952.1762695, -1121.8753662, 1155.8247070
3: -224.7886505, 1487.0450439, -215.9046173, 1425.0424805, -1649.8311768, 1702.9497070
4: -316.0958557, 1213.2071533, -304.0985718, 1164.4720459, -1480.5678711, 1517.3056641

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5129653, upper bound: 1988.5127832
time: 0.70 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5129653, upper bound: 1988.5127311
time: 0.99 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -403.8203430, 1734.9710693, -412.2035828, 1771.2501221, -2175.0705566, 2147.1745605
1: -293.7789612, 1016.4409180, -299.8886414, 1036.4454346, -1330.2243652, 1316.3293457
2: -160.4777527, 935.7810059, -163.7719421, 953.9990234, -1114.4766846, 1099.5527344
3: -211.4084473, 1399.7779541, -215.8461914, 1427.4410400, -1638.8494873, 1615.6241455
4: -298.2576294, 1143.4577637, -304.3381653, 1166.0527344, -1464.3101807, 1447.7958984

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5128174
time: 0.75 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5129669
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.37 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -1988.5129653, upper bound: 1988.5127832
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -1988.5129653, upper bound: 1988.5127311
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5128174
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -1988.5129669, upper bound: 1988.5129669

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -372.4161987, 1599.6984863, -398.7648010, 1711.6734619, -2084.0895996, 1998.4630127
1: -270.0293274, 942.3277588, -290.4829407, 1000.3730469, -1270.4023438, 1232.8106689
2: -147.8569031, 867.8069458, -158.5164490, 919.9156494, -1067.7725830, 1026.3233643
3: -196.2015839, 1298.5543213, -208.9828339, 1377.4359131, -1573.6374512, 1507.5371094
4: -275.6720886, 1060.5567627, -294.2872620, 1125.6239014, -1401.2960205, 1354.8439941

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5094542, upper bound: 1988.5092801
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5127311, upper bound: 1988.5127832
time: 0.63 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5127311, upper bound: 1988.5127832
time: 0.70 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -421.4123535, 1818.2546387, -407.4792175, 1750.3548584, -2171.7670898, 2225.7336426
1: -306.1087341, 1062.9010010, -296.8910522, 1023.5783081, -1329.6870117, 1359.7917480
2: -167.2639313, 977.9395752, -161.9669647, 941.7489624, -1109.0129395, 1139.9064941
3: -221.5706024, 1465.6555176, -213.5244293, 1409.1846924, -1630.7552490, 1679.1799316
4: -311.3899841, 1195.7893066, -300.6502991, 1151.5146484, -1462.9046631, 1496.4395752

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5128094, upper bound: 1988.5127311
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5128094, upper bound: 1988.5127311
time: 0.76 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -333.7939453, 1426.7482910, -397.4642029, 1706.6577148, -2040.4516602, 1824.2121582
1: -242.0747223, 841.2408447, -289.3919373, 998.1356201, -1240.2103271, 1130.6325684
2: -132.5777588, 775.8034058, -158.0448761, 918.1994019, -1050.7769775, 933.8481445
3: -174.8605804, 1157.7049561, -208.2790070, 1374.5092773, -1549.3698730, 1365.9840088
4: -246.5855408, 946.4917603, -293.5660706, 1123.0175781, -1369.6031494, 1240.0578613

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097036, upper bound: 1988.5094966
time: 0.86 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097556, upper bound: 1988.5097556
time: 0.67 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -399.6012878, 1716.1192627, -408.9624023, 1756.8414307, -2156.4426270, 2125.0812988
1: -290.8214722, 1005.8336792, -297.6264648, 1028.3096924, -1319.1311035, 1303.4599609
2: -158.8240509, 926.1121826, -162.5118713, 946.6056519, -1105.4296875, 1088.6240234
3: -209.1835785, 1385.1188965, -214.1495209, 1416.1892090, -1625.3728027, 1599.2681885
4: -294.9515686, 1131.5091553, -301.8394775, 1156.8764648, -1451.8280029, 1433.3486328

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5128174, upper bound: 1988.5129669
time: 0.66 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5128174, upper bound: 1988.5129669
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.46 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5127311, upper bound: 1988.5127832
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5127311, upper bound: 1988.5127832
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5128094, upper bound: 1988.5127311
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5128094, upper bound: 1988.5127311
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5097036, upper bound: 1988.5094966
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5097556, upper bound: 1988.5097556
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5128174, upper bound: 1988.5129669
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -1988.5128174, upper bound: 1988.5129669

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -372.4161987, 1599.6984863, -411.4973145, 1774.7050781, -2147.1210938, 2011.1956787
1: -270.0293274, 942.3277588, -299.1470032, 1035.3498535, -1305.3791504, 1241.4747314
2: -147.8569031, 867.8069458, -163.4472961, 951.9797363, -1099.8366699, 1031.2542725
3: -196.2015839, 1298.5543213, -216.4116211, 1428.5755615, -1624.7770996, 1514.9656982
4: -275.6720886, 1060.5567627, -304.3067627, 1164.9381104, -1440.6102295, 1364.8635254

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
time: 0.76 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
time: 0.61 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -372.4161987, 1599.6984863, -388.4694519, 1667.2901611, -2039.7062988, 1988.1679688
1: -270.0293274, 942.3277588, -282.8243103, 976.4136353, -1246.4429932, 1225.1519775
2: -147.8569031, 867.8069458, -154.5059357, 898.3294678, -1046.1862793, 1022.3128662
3: -196.2015839, 1298.5543213, -203.5020905, 1344.3978271, -1540.5993652, 1502.0563965
4: -275.6720886, 1060.5567627, -287.0222778, 1098.5128174, -1374.1849365, 1347.5791016

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
time: 0.89 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -421.4123535, 1818.2546387, -336.8312988, 1439.7578125, -1861.1701660, 2155.0856934
1: -306.1087341, 1062.9010010, -244.2926331, 847.6206055, -1153.7291260, 1307.1934814
2: -167.2639313, 977.9395752, -133.6994781, 781.3884888, -948.6524048, 1111.6390381
3: -221.5706024, 1465.6555176, -176.3841705, 1166.2827148, -1387.8532715, 1642.0396729
4: -311.3899841, 1195.7893066, -248.5417938, 953.7298584, -1265.1197510, 1444.3310547

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
time: 0.80 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
time: 0.86 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -421.4123535, 1818.2546387, -405.4753723, 1741.4392090, -2162.8515625, 2223.7299805
1: -306.1087341, 1062.9010010, -295.4772034, 1018.4823608, -1324.5910645, 1358.3781738
2: -167.2639313, 977.9395752, -161.1810760, 937.1079712, -1104.3719482, 1139.1206055
3: -221.5706024, 1465.6555176, -212.4681702, 1402.1818848, -1623.7524414, 1678.1236572
4: -311.3899841, 1195.7893066, -299.1132202, 1145.7701416, -1457.1601562, 1494.9024658

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
time: 0.62 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
time: 0.73 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -247.4265442, 1041.3784180, -368.8153687, 1577.0782471, -1824.5047607, 1410.1936035
1: -178.0705566, 626.5579834, -269.0192261, 925.0196533, -1103.0899658, 895.5772095
2: -97.9595032, 578.9024048, -146.7438507, 851.0785522, -949.0379639, 725.6462402
3: -129.1749420, 859.8123779, -193.1645508, 1273.1618652, -1402.3367920, 1052.9769287
4: -182.5711823, 704.2663574, -272.0927124, 1040.8035889, -1223.3746338, 976.3590698

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5095793, upper bound: 1988.5090823
time: 0.84 seconds

## Relational analysis of NS_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5096413, upper bound: 1988.5090693
time: 0.88 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -329.7008667, 1409.3867188, -394.7047729, 1694.1771240, -2023.8779297, 1804.0913086
1: -239.1085205, 830.6886597, -287.3559265, 990.4630737, -1229.5715332, 1118.0445557
2: -130.9466248, 766.0565796, -156.9183655, 911.4351196, -1042.3817139, 922.9749756
3: -172.8370667, 1143.1757812, -206.8766785, 1364.1320801, -1536.9691162, 1350.0523682
4: -243.7056427, 934.6406860, -291.5679016, 1114.4249268, -1358.1302490, 1226.2086182

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5093240, upper bound: 1988.5096757
time: 0.89 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5093240, upper bound: 1988.5097556
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -399.6012878, 1716.1192627, -340.9376831, 1457.9508057, -1857.5518799, 2057.0566406
1: -290.8214722, 1005.8336792, -247.2396240, 858.5258179, -1149.3472900, 1253.0732422
2: -158.8240509, 926.1121826, -135.3764954, 791.6227417, -950.4467773, 1061.4886475
3: -209.1835785, 1385.1188965, -178.6121674, 1181.6656494, -1390.8491211, 1563.7310791
4: -294.9515686, 1131.5091553, -251.7689514, 966.0161133, -1260.9676514, 1383.2780762

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5094966, upper bound: 1988.5097036
time: 0.68 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097556, upper bound: 1988.5097556
time: 0.68 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -399.6012878, 1716.1192627, -407.6912231, 1751.0693359, -2150.6701660, 2123.8100586
1: -290.8214722, 1005.8336792, -296.7160950, 1025.0285645, -1315.8500977, 1302.5496826
2: -158.8240509, 926.1121826, -162.0018158, 943.6217651, -1102.4458008, 1088.1140137
3: -209.1835785, 1385.1188965, -213.4668579, 1411.6727295, -1620.8562012, 1598.5856934
4: -294.9515686, 1131.5091553, -300.8144836, 1153.1896973, -1448.1411133, 1432.3236084

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5094966, upper bound: 1988.5097036
time: 0.76 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097556, upper bound: 1988.5097556
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.45 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127832
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5126941, upper bound: 1988.5127311
NS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5095793, upper bound: 1988.5090823
NS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5096413, upper bound: 1988.5090693
NS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5093240, upper bound: 1988.5096757
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5093240, upper bound: 1988.5097556
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5094966, upper bound: 1988.5097036
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5097556, upper bound: 1988.5097556
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5094966, upper bound: 1988.5097036
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -1988.5097556, upper bound: 1988.5097556

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -372.4161987, 1599.6984863, -372.4161987, 1599.6984863, -1972.1147461, 1972.1147461
1: -270.0293274, 942.3277588, -270.0293274, 942.3277588, -1212.3570557, 1212.3570557
2: -147.8569031, 867.8069458, -147.8569031, 867.8069458, -1015.6638184, 1015.6638184
3: -196.2015839, 1298.5543213, -196.2015839, 1298.5543213, -1494.7558594, 1494.7558594
4: -275.6720886, 1060.5567627, -275.6720886, 1060.5567627, -1336.2288818, 1336.2288818

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113725, upper bound: 1988.5116632
time: 0.67 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5110898, upper bound: 1988.5112692
time: 1.08 seconds

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -372.4161987, 1599.6984863, -421.4123535, 1818.2546387, -2190.6706543, 2021.1107178
1: -270.0293274, 942.3277588, -306.1087341, 1062.9010010, -1332.9302979, 1248.4365234
2: -147.8569031, 867.8069458, -167.2639313, 977.9395752, -1125.7965088, 1035.0709229
3: -196.2015839, 1298.5543213, -221.5706024, 1465.6555176, -1661.8570557, 1520.1247559
4: -275.6720886, 1060.5567627, -311.3899841, 1195.7893066, -1471.4614258, 1371.9467773

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5121695, upper bound: 1988.5118055
time: 0.62 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5110898, upper bound: 1988.5112692
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -372.4161987, 1599.6984863, -333.7939453, 1426.7482910, -1799.1645508, 1933.4923096
1: -270.0293274, 942.3277588, -242.0747223, 841.2408447, -1111.2701416, 1184.4024658
2: -147.8569031, 867.8069458, -132.5777588, 775.8034058, -923.6602783, 1000.3847046
3: -196.2015839, 1298.5543213, -174.8605804, 1157.7049561, -1353.9064941, 1473.4149170
4: -275.6720886, 1060.5567627, -246.5855408, 946.4917603, -1222.1638184, 1307.1423340

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124278, upper bound: 1988.5123890
time: 0.78 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124281, upper bound: 1988.5123625
time: 0.70 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -372.4161987, 1599.6984863, -399.6012878, 1716.1192627, -2088.5354004, 1999.2994385
1: -270.0293274, 942.3277588, -290.8214722, 1005.8336792, -1275.8630371, 1233.1491699
2: -147.8569031, 867.8069458, -158.8240509, 926.1121826, -1073.9691162, 1026.6309814
3: -196.2015839, 1298.5543213, -209.1835785, 1385.1188965, -1581.3204346, 1507.7377930
4: -275.6720886, 1060.5567627, -294.9515686, 1131.5091553, -1407.1811523, 1355.5083008

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5122689, upper bound: 1988.5118177
time: 0.64 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124278, upper bound: 1988.5123890
time: 0.72 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124281, upper bound: 1988.5123625
time: 0.61 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -421.4123535, 1818.2546387, -372.4161987, 1599.6984863, -2021.1105957, 2190.6706543
1: -306.1087341, 1062.9010010, -270.0293274, 942.3277588, -1248.4365234, 1332.9302979
2: -167.2639313, 977.9395752, -147.8569031, 867.8069458, -1035.0708008, 1125.7965088
3: -221.5706024, 1465.6555176, -196.2015839, 1298.5543213, -1520.1248779, 1661.8570557
4: -311.3899841, 1195.7893066, -275.6720886, 1060.5567627, -1371.9467773, 1471.4614258

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5123796, upper bound: 1988.5124733
time: 1.08 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5123583, upper bound: 1988.5125242
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -421.4123535, 1818.2546387, -333.7939453, 1426.7482910, -1848.1604004, 2152.0483398
1: -306.1087341, 1062.9010010, -242.0747223, 841.2408447, -1147.3493652, 1304.9757080
2: -167.2639313, 977.9395752, -132.5777588, 775.8034058, -943.0673218, 1110.5172119
3: -221.5706024, 1465.6555176, -174.8605804, 1157.7049561, -1379.2755127, 1640.5161133
4: -311.3899841, 1195.7893066, -246.5855408, 946.4917603, -1257.8817139, 1442.3748779

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5120909, upper bound: 1988.5118518
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5119126, upper bound: 1988.5119127
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -421.4123535, 1818.2546387, -421.4123535, 1818.2546387, -2239.6665039, 2239.6665039
1: -306.1087341, 1062.9010010, -306.1087341, 1062.9010010, -1369.0097656, 1369.0097656
2: -167.2639313, 977.9395752, -167.2639313, 977.9395752, -1145.2034912, 1145.2034912
3: -221.5706024, 1465.6555176, -221.5706024, 1465.6555176, -1687.2260742, 1687.2260742
4: -311.3899841, 1195.7893066, -311.3899841, 1195.7893066, -1507.1793213, 1507.1793213

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5119472, upper bound: 1988.5117493
time: 0.78 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117470, upper bound: 1988.5118729
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -421.4123535, 1818.2546387, -399.6012878, 1716.1192627, -2137.5314941, 2217.8554688
1: -306.1087341, 1062.9010010, -290.8214722, 1005.8336792, -1311.9423828, 1353.7224121
2: -167.2639313, 977.9395752, -158.8240509, 926.1121826, -1093.3760986, 1136.7636719
3: -221.5706024, 1465.6555176, -209.1835785, 1385.1188965, -1606.6894531, 1674.8391113
4: -311.3899841, 1195.7893066, -294.9515686, 1131.5091553, -1442.8991699, 1490.7408447

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5119472, upper bound: 1988.5118518
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117470, upper bound: 1988.5119301
time: 0.66 seconds

## BFS NS instance: NS_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -219.1640625, 917.4870605, -356.0086060, 1521.5582275, -1740.7221680, 1273.4953613
1: -157.7118225, 551.6383057, -259.9613953, 892.4428101, -1050.1545410, 811.5997314
2: -86.8654709, 509.1175232, -141.7308350, 821.2568359, -908.1223145, 650.8483887
3: -114.1791077, 757.3843994, -186.3953400, 1228.2022705, -1342.3813477, 943.7797241
4: -161.7084198, 619.8987427, -262.5409851, 1004.2290039, -1165.9371338, 882.4395142

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5076686, upper bound: 1988.5068527
time: 0.65 seconds

## Relational analysis of NS_A2_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5092018, upper bound: 1988.5090823
time: 0.79 seconds

## Relational analysis of NS_A2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5092018, upper bound: 1988.5090823
time: 0.81 seconds

## BFS NS instance: NS_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -238.9683228, 1006.0363159, -359.1230774, 1534.8481445, -1773.8164062, 1365.1593018
1: -171.8482666, 603.5031128, -261.9486389, 898.6743774, -1070.5227051, 865.4517212
2: -94.6287918, 557.2852173, -142.9180145, 826.4520264, -921.0808105, 700.2032471
3: -124.7288971, 828.4636841, -187.9496002, 1237.2149658, -1361.9436035, 1016.4132690
4: -176.4165192, 678.2586670, -264.9144287, 1011.0630493, -1187.4796143, 943.1730347

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_A1_A2_A1

### Relational analysis result of NS_A2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5079766, upper bound: 1988.5070513
time: 0.61 seconds

## Relational analysis of NS_A2_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_A1_A2_B1

### Relational analysis result of NS_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5091964, upper bound: 1988.5090397
time: 0.75 seconds

## Relational analysis of NS_A2_A1_A1_A2_B2

### Relational analysis result of NS_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5091964, upper bound: 1988.5090693
time: 0.91 seconds

## BFS NS instance: NS_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -329.7008667, 1409.3867188, -309.8848572, 1316.9005127, -1646.6013184, 1719.2713623
1: -239.1085205, 830.6886597, -224.7691345, 781.8184204, -1020.9269409, 1055.4577637
2: -130.9466248, 766.0565796, -123.0581207, 719.9089355, -850.8554077, 889.1145630
3: -172.8370667, 1143.1757812, -162.0807495, 1075.0485840, -1247.8854980, 1305.2565918
4: -243.7056427, 934.6406860, -228.8416595, 878.9530640, -1122.6585693, 1163.4822998

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_A2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5087717, upper bound: 1988.5089845
time: 0.58 seconds

## Relational analysis of NS_A2_A1_A2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5091004, upper bound: 1988.5096206
time: 0.66 seconds

## BFS NS instance: NS_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -329.7008667, 1409.3867188, -393.4365234, 1688.7572021, -2018.4580078, 1802.8232422
1: -239.1085205, 830.6886597, -286.4118958, 987.3236084, -1226.4321289, 1117.1005859
2: -130.9466248, 766.0565796, -156.4099121, 908.5196533, -1039.4663086, 922.4664917
3: -172.8370667, 1143.1757812, -206.2511444, 1359.8134766, -1532.6505127, 1349.4268799
4: -243.7056427, 934.6406860, -290.6714172, 1110.8807373, -1354.5860596, 1225.3121338

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_A2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5087717, upper bound: 1988.5090404
time: 0.78 seconds

## Relational analysis of NS_A2_A1_A2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5091004, upper bound: 1988.5097262
time: 0.85 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -370.3918152, 1584.4935303, -253.6225281, 1068.4840088, -1438.8758545, 1838.1158447
1: -270.1127319, 931.5620117, -182.5741272, 641.6346436, -911.7473755, 1114.1361084
2: -147.3456116, 857.6211548, -100.4124756, 592.6696167, -740.0152588, 958.0336304
3: -193.8452301, 1281.4725342, -132.4341888, 880.7500610, -1074.5950928, 1413.9067383
4: -273.1831970, 1047.8588867, -187.1121521, 721.2974243, -994.4805908, 1234.9710693

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090840, upper bound: 1988.5092804
time: 0.69 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090840, upper bound: 1988.5096730
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -396.9595642, 1704.5773926, -336.7902222, 1440.2618408, -1837.2210693, 2041.3675537
1: -288.8568420, 998.9509888, -244.2259827, 847.7672729, -1136.6241455, 1243.1768799
2: -157.7445984, 919.7991943, -133.7190552, 781.6755371, -939.4201050, 1053.5181885
3: -207.8522491, 1375.4459229, -176.5486908, 1166.8376465, -1374.6895752, 1551.9945068
4: -293.0443115, 1123.7186279, -248.8411407, 953.8999023, -1246.9440918, 1372.5595703

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5096757, upper bound: 1988.5093240
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5096757, upper bound: 1988.5097556
time: 0.87 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -370.3918152, 1584.4935303, -319.0867004, 1357.3745117, -1727.7663574, 1903.5799561
1: -270.1127319, 931.5620117, -231.3012390, 805.6589355, -1075.7717285, 1162.8632812
2: -147.3456116, 857.6211548, -126.6064606, 742.3884277, -889.7340088, 984.2276001
3: -193.8452301, 1281.4725342, -166.7475433, 1107.9818115, -1301.8270264, 1448.2200928
4: -273.1831970, 1047.8588867, -235.3379974, 905.6230469, -1178.8061523, 1283.1968994

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5091376, upper bound: 1988.5092804
time: 0.75 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5091376, upper bound: 1988.5096730
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -396.9595642, 1704.5773926, -403.6055603, 1733.4029541, -2130.3620605, 2108.1828613
1: -288.8568420, 998.9509888, -293.6866455, 1014.5359497, -1303.3925781, 1292.6376953
2: -157.7445984, 919.7991943, -160.3482819, 933.9457397, -1091.6900635, 1080.1473389
3: -207.8522491, 1375.4459229, -211.4171753, 1396.9808350, -1604.8327637, 1586.8630371
4: -293.0443115, 1123.7186279, -297.8879395, 1141.2938232, -1434.3378906, 1421.6063232

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5096730, upper bound: 1988.5093240
time: 0.80 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5096730, upper bound: 1988.5097556
time: 0.97 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.71 seconds
NS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5113725, upper bound: 1988.5116632
NS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5110898, upper bound: 1988.5112692
NS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5121695, upper bound: 1988.5118055
NS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5110898, upper bound: 1988.5112692
NS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5124278, upper bound: 1988.5123890
NS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5124281, upper bound: 1988.5123625
NS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5124278, upper bound: 1988.5123890
NS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5124281, upper bound: 1988.5123625
NS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5123796, upper bound: 1988.5124733
NS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5123583, upper bound: 1988.5125242
NS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5120909, upper bound: 1988.5118518
NS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5119126, upper bound: 1988.5119127
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5119472, upper bound: 1988.5117493
NS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5117470, upper bound: 1988.5118729
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5119472, upper bound: 1988.5118518
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5117470, upper bound: 1988.5119301
NS_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5092018, upper bound: 1988.5090823
NS_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5092018, upper bound: 1988.5090823
NS_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5091964, upper bound: 1988.5090397
NS_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5091964, upper bound: 1988.5090693
NS_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5087717, upper bound: 1988.5089845
NS_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5091004, upper bound: 1988.5096206
NS_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5087717, upper bound: 1988.5090404
NS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5091004, upper bound: 1988.5097262
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5090840, upper bound: 1988.5092804
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5090840, upper bound: 1988.5096730
NS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5096757, upper bound: 1988.5093240
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5096757, upper bound: 1988.5097556
NS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5091376, upper bound: 1988.5092804
NS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5091376, upper bound: 1988.5096730
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5096730, upper bound: 1988.5093240
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 0, lower bound: -1988.5096730, upper bound: 1988.5097556

## BFS NS instance: NS_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -367.2597351, 1577.8812256, -371.1057129, 1594.2114258, -1961.4710693, 1948.9869385
1: -266.1236267, 930.2934570, -269.0437927, 939.2839966, -1205.4075928, 1199.3371582
2: -145.7555542, 856.8742065, -147.3260498, 865.0406494, -1010.7961426, 1004.2002563
3: -193.5350342, 1281.7968750, -195.5293427, 1294.3179932, -1487.8530273, 1477.3261719
4: -271.8663635, 1046.9752197, -274.7109375, 1057.1232910, -1328.9896240, 1321.6861572

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5116456, upper bound: 1988.5116456
time: 0.94 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5116456, upper bound: 1988.5116456
time: 0.79 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -358.1880798, 1540.6791992, -361.3902893, 1552.9165039, -1911.1043701, 1902.0694580
1: -259.3468018, 904.9525146, -261.9198914, 913.0919800, -1172.4387207, 1166.8719482
2: -142.0380249, 833.3172607, -143.4176331, 840.7447510, -982.7826538, 976.7348633
3: -188.4001007, 1247.4707031, -190.2362518, 1258.6851807, -1447.0852051, 1437.7069092
4: -264.9048157, 1018.6882324, -267.3275757, 1027.7955322, -1292.7003174, 1286.0158691

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097396, upper bound: 1988.5098649
time: 0.78 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5094911, upper bound: 1988.5094911
time: 0.81 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -371.1057129, 1594.2114258, -414.8801270, 1790.7520752, -2161.8579102, 2009.0914307
1: -269.0437927, 939.2839966, -301.1653748, 1047.7479248, -1316.7917480, 1240.4493408
2: -147.3260498, 865.0406494, -164.6259308, 964.1951904, -1111.5211182, 1029.6665039
3: -195.5293427, 1294.3179932, -218.2227173, 1444.4472656, -1639.9765625, 1512.5405273
4: -274.7109375, 1057.1232910, -306.5779114, 1178.6370850, -1453.3480225, 1363.7011719

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
time: 0.80 seconds

## Relational analysis of NS_A1_A1_B1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
time: 0.81 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -361.3902893, 1552.9165039, -410.0518799, 1769.6093750, -2130.9992676, 1962.9683838
1: -261.9198914, 913.0919800, -297.2399597, 1032.2169189, -1294.1367188, 1210.3319092
2: -143.4176331, 840.7447510, -162.4854584, 949.7734375, -1093.1910400, 1003.2301636
3: -190.2362518, 1258.6851807, -215.1368561, 1423.2961426, -1613.5323486, 1473.8220215
4: -267.3275757, 1027.7955322, -302.5274963, 1161.5067139, -1428.8342285, 1330.3229980

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
time: 0.70 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -360.6166382, 1548.6755371, -327.6122131, 1400.4986572, -1761.1152344, 1876.2877197
1: -261.1275940, 911.7692871, -237.4667816, 824.7174683, -1085.8450928, 1149.2357178
2: -143.0683899, 839.6314087, -130.0790100, 760.4795532, -903.5479736, 969.7103271
3: -190.0061188, 1256.4957275, -171.6403656, 1135.1394043, -1325.1455078, 1428.1361084
4: -266.9199524, 1026.2280273, -242.0093536, 927.8819580, -1194.8018799, 1268.2373047

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123023
time: 0.91 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123189
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -367.4802246, 1575.8980713, -327.6094666, 1399.3939209, -1766.8741455, 1903.5073242
1: -266.0974121, 927.9514771, -237.3476410, 825.5910034, -1091.6884766, 1165.2990723
2: -145.7360687, 854.7364502, -130.0359955, 761.5786743, -907.3146973, 984.7724609
3: -193.4766388, 1278.6960449, -171.5725555, 1135.8326416, -1329.3092041, 1450.2685547
4: -271.7196655, 1044.4227295, -241.8950043, 928.9860229, -1200.7056885, 1286.3177490

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123547
time: 1.04 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123625
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -360.6166382, 1548.6755371, -395.7755737, 1699.9890137, -2060.6052246, 1944.4509277
1: -261.1275940, 911.7692871, -287.9488831, 996.2236938, -1257.3513184, 1199.7181396
2: -143.0683899, 839.6314087, -157.2714386, 917.3029175, -1060.3712158, 996.9028320
3: -190.0061188, 1256.4957275, -207.1820984, 1371.6060791, -1561.6121826, 1463.6776123
4: -266.9199524, 1026.2280273, -292.0809631, 1120.6864014, -1387.6062012, 1318.3089600

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5122993
time: 0.80 seconds

## Relational analysis of NS_A1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5123189
time: 1.12 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -367.4802246, 1575.8980713, -389.9722900, 1673.5422363, -2041.0224609, 1965.8701172
1: -266.0974121, 927.9514771, -283.4963684, 980.6463013, -1246.7436523, 1211.4478760
2: -145.7360687, 854.7364502, -154.8688354, 903.4940796, -1049.2301025, 1009.6052246
3: -193.4766388, 1278.6960449, -204.0893250, 1350.6130371, -1544.0897217, 1482.7852783
4: -271.7196655, 1044.4227295, -287.6905518, 1103.0855713, -1374.8051758, 1332.1132812

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5123447
time: 0.81 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5123625
time: 0.80 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -416.4453735, 1796.6567383, -360.6166382, 1548.6755371, -1965.1208496, 2157.2734375
1: -302.3494263, 1050.1081543, -261.1275940, 911.7692871, -1214.1186523, 1311.2355957
2: -165.2422485, 966.2098389, -143.0683899, 839.6314087, -1004.8736572, 1109.2781982
3: -218.9506073, 1448.0997314, -190.0061188, 1256.4957275, -1475.4462891, 1638.1058350
4: -307.6609802, 1181.5115967, -266.9199524, 1026.2280273, -1333.8887939, 1448.4315186

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5115054, upper bound: 1988.5112434
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112864, upper bound: 1988.5113066
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -413.5924072, 1784.1998291, -367.4802246, 1575.8980713, -1989.4904785, 2151.6801758
1: -300.2604675, 1042.6057129, -266.0974121, 927.9514771, -1228.2119141, 1308.7031250
2: -164.0913849, 959.3992920, -145.7360687, 854.7364502, -1018.8278198, 1105.1353760
3: -217.4538727, 1437.5267334, -193.4766388, 1278.6960449, -1496.1499023, 1631.0032959
4: -305.4898071, 1172.9144287, -271.7196655, 1044.4227295, -1349.9125977, 1444.6340332

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5110046, upper bound: 1988.5118881
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5123060, upper bound: 1988.5125025
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -414.0748291, 1786.5034180, -332.8182678, 1422.5678711, -1836.6425781, 2119.3217773
1: -300.6527710, 1043.1757812, -241.3675690, 838.5983887, -1139.2509766, 1284.5432129
2: -164.2835236, 959.8468018, -132.1879272, 773.3909302, -937.6744385, 1092.0346680
3: -217.6616364, 1438.6409912, -174.3522491, 1154.0825195, -1371.7441406, 1612.9932861
4: -305.8672485, 1173.8259277, -245.8634338, 943.5474854, -1249.4145508, 1419.6893311

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5118179, upper bound: 1988.5118177
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5118179, upper bound: 1988.5118177
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -417.4580688, 1801.6279297, -333.7939453, 1426.7482910, -1844.2062988, 2135.4216309
1: -303.1703186, 1053.2519531, -242.0747223, 841.2408447, -1144.4110107, 1295.3266602
2: -165.6922150, 969.1755981, -132.5777588, 775.8034058, -941.4956055, 1101.7534180
3: -219.5612488, 1452.1956787, -174.8605804, 1157.7049561, -1377.2662354, 1627.0562744
4: -308.4869385, 1184.9212646, -246.5855408, 946.4917603, -1254.9787598, 1431.5067139

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117939, upper bound: 1988.5119127
time: 0.90 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5118179, upper bound: 1988.5119127
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -414.0748291, 1786.5034180, -420.4465332, 1814.1029053, -2228.1777344, 2206.9499512
1: -300.6527710, 1043.1757812, -305.3977356, 1060.2264404, -1360.8789062, 1348.5734863
2: -164.2835236, 959.8468018, -166.8747101, 975.4767456, -1139.7600098, 1126.7214355
3: -217.6616364, 1438.6409912, -221.0569000, 1462.0092773, -1679.6708984, 1659.6978760
4: -305.8672485, 1173.8259277, -310.6605225, 1192.8128662, -1498.6801758, 1484.4864502

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5117195
time: 0.78 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5117195
time: 0.90 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -417.4580688, 1801.6279297, -421.4123535, 1818.2546387, -2235.7124023, 2223.0400391
1: -303.1703186, 1053.2519531, -306.1087341, 1062.9010010, -1366.0712891, 1359.3605957
2: -165.6922150, 969.1755981, -167.2639313, 977.9395752, -1143.6318359, 1136.4395752
3: -219.5612488, 1452.1956787, -221.5706024, 1465.6555176, -1685.2167969, 1673.7662354
4: -308.4869385, 1184.9212646, -311.3899841, 1195.7893066, -1504.2762451, 1496.3111572

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5118729
time: 1.33 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5118729
time: 0.73 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -414.0748291, 1786.5034180, -398.7873840, 1712.5726318, -2126.6474609, 2185.2902832
1: -300.6527710, 1043.1757812, -290.2246094, 1003.6961060, -1304.3487549, 1333.4002686
2: -164.2835236, 959.8468018, -158.4984436, 924.1687012, -1088.4522705, 1118.3452148
3: -217.6616364, 1438.6409912, -208.7571106, 1382.1323242, -1599.7939453, 1647.3980713
4: -305.8672485, 1173.8259277, -294.3447876, 1129.1307373, -1434.9980469, 1468.1706543

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5115278, upper bound: 1988.5112760
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5121379, upper bound: 1988.5118371
time: 0.89 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -417.4580688, 1801.6279297, -399.6012878, 1716.1192627, -2133.5773926, 2201.2290039
1: -303.1703186, 1053.2519531, -290.8214722, 1005.8336792, -1309.0040283, 1344.0733643
2: -165.6922150, 969.1755981, -158.8240509, 926.1121826, -1091.8044434, 1127.9996338
3: -219.5612488, 1452.1956787, -209.1835785, 1385.1188965, -1604.6801758, 1661.3791504
4: -308.4869385, 1184.9212646, -294.9515686, 1131.5091553, -1439.9960938, 1479.8725586

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111851, upper bound: 1988.5114292
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5119392, upper bound: 1988.5119051
time: 0.85 seconds

## BFS NS instance: NS_A2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -219.1640625, 917.4870605, -295.8348999, 1255.9833984, -1475.1474609, 1213.3220215
1: -157.7118225, 551.6383057, -214.7605133, 745.7926025, -903.5043945, 766.3988037
2: -86.8654709, 509.1175232, -117.5858459, 686.7536621, -773.6191406, 626.7033081
3: -114.1791077, 757.3843994, -154.6614990, 1025.5067139, -1139.6856689, 912.0457764
4: -161.7084198, 619.8987427, -218.4282684, 838.5256958, -1000.2341309, 838.3269653

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
time: 0.68 seconds

## Relational analysis of NS_A2_A1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
time: 0.67 seconds

## BFS NS instance: NS_A2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -219.1640625, 917.4870605, -378.3598938, 1622.5616455, -1841.7257080, 1295.8469238
1: -157.7118225, 551.6383057, -275.7722778, 948.0942993, -1105.8061523, 827.4104614
2: -86.8654709, 509.1175232, -150.5299072, 872.4895020, -959.3549805, 659.6474609
3: -114.1791077, 757.3843994, -198.2985229, 1305.7414551, -1419.9205322, 955.6828613
4: -161.7084198, 619.8987427, -279.5054321, 1066.7464600, -1228.4547119, 899.4041748

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A1_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
time: 0.74 seconds

## Relational analysis of NS_A2_A1_A1_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
time: 0.84 seconds

## BFS NS instance: NS_A2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -238.9683228, 1006.0363159, -301.2904053, 1280.1693115, -1519.1376953, 1307.3266602
1: -171.8482666, 603.5031128, -218.5370636, 758.6947021, -930.5429688, 822.0401611
2: -94.6287918, 557.2852173, -119.6868362, 698.3255615, -792.9542847, 676.9720459
3: -124.7288971, 828.4636841, -157.4613800, 1043.5815430, -1168.3101807, 985.9250488
4: -176.4165192, 678.2586670, -222.5400391, 852.8181152, -1029.2346191, 900.7985840

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_A1_A2_B1_A1

### Relational analysis result of NS_A2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5078572, upper bound: 1988.5082003
time: 0.83 seconds

## Relational analysis of NS_A2_A1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090397
time: 0.70 seconds

## Relational analysis of NS_A2_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090397
time: 0.62 seconds

## BFS NS instance: NS_A2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -238.9683228, 1006.0363159, -384.6956177, 1650.8795166, -1889.8477783, 1390.7316895
1: -171.8482666, 603.5031128, -280.0421143, 963.8174438, -1135.6657715, 883.5452271
2: -94.6287918, 557.2852173, -152.9742126, 886.5571289, -981.1859131, 710.2593994
3: -124.7288971, 828.4636841, -201.5563812, 1327.7675781, -1452.4962158, 1030.0200195
4: -176.4165192, 678.2586670, -284.2148743, 1084.3557129, -1260.7722168, 962.4734497

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090693
time: 0.71 seconds

## Relational analysis of NS_A2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090693
time: 0.66 seconds

## BFS NS instance: NS_A2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -294.6404724, 1256.1848145, -295.8348999, 1255.9833984, -1550.6237793, 1552.0197754
1: -214.0937500, 740.3949585, -214.7605133, 745.7926025, -959.8863525, 955.1554565
2: -117.1905823, 682.4744873, -117.5858459, 686.7536621, -803.9442139, 800.0601807
3: -154.3628387, 1018.7535400, -154.6614990, 1025.5067139, -1179.8695068, 1173.4149170
4: -217.8730774, 832.7695923, -218.4282684, 838.5256958, -1056.3984375, 1051.1978760

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090004, upper bound: 1988.5090825
time: 0.70 seconds

## Relational analysis of NS_A2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090004, upper bound: 1988.5090825
time: 0.80 seconds

## BFS NS instance: NS_A2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -320.3485107, 1370.7280273, -301.2904053, 1280.1693115, -1600.5178223, 1672.0184326
1: -232.2768860, 805.9603271, -218.5370636, 758.6947021, -990.9714966, 1024.4974365
2: -127.3069229, 742.7835693, -119.6868362, 698.3255615, -825.6324463, 862.4703979
3: -167.8936005, 1109.7901611, -157.4613800, 1043.5815430, -1211.4749756, 1267.2515869
4: -236.9422150, 906.7779541, -222.5400391, 852.8181152, -1089.7602539, 1129.3177490

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090693, upper bound: 1988.5096413
time: 0.86 seconds

## Relational analysis of NS_A2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090693, upper bound: 1988.5096413
time: 1.02 seconds

## BFS NS instance: NS_A2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -294.6404724, 1256.1848145, -378.3598938, 1622.5616455, -1917.2020264, 1634.5446777
1: -214.0937500, 740.3949585, -275.7722778, 948.0942993, -1162.1879883, 1016.1671143
2: -117.1905823, 682.4744873, -150.5299072, 872.4895020, -989.6800537, 833.0043335
3: -154.3628387, 1018.7535400, -198.2985229, 1305.7414551, -1460.1042480, 1217.0518799
4: -217.8730774, 832.7695923, -279.5054321, 1066.7464600, -1284.6195068, 1112.2750244

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5039780, upper bound: 1988.5043156
time: 0.78 seconds

## Relational analysis of NS_A2_A1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089271, upper bound: 1988.5090404
time: 0.75 seconds

## Relational analysis of NS_A2_A1_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5089271, upper bound: 1988.5090404
time: 0.64 seconds

## BFS NS instance: NS_A2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -320.3485107, 1370.7280273, -384.6956177, 1650.8795166, -1971.2280273, 1755.4235840
1: -232.2768860, 805.9603271, -280.0421143, 963.8174438, -1196.0943604, 1086.0024414
2: -127.3069229, 742.7835693, -152.9742126, 886.5571289, -1013.8640747, 895.7578125
3: -167.8936005, 1109.7901611, -201.5563812, 1327.7675781, -1495.6611328, 1311.3465576
4: -236.9422150, 906.7779541, -284.2148743, 1084.3557129, -1321.2976074, 1190.9926758

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_A2_B2_A2_A1

### Relational analysis result of NS_A2_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5074975, upper bound: 1988.5075421
time: 0.63 seconds

## Relational analysis of NS_A2_A1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097331, upper bound: 1988.5097262
time: 0.63 seconds

## Relational analysis of NS_A2_A1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097331, upper bound: 1988.5097262
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -312.2982483, 1327.3442383, -253.6225281, 1068.4840088, -1380.7822266, 1580.9665527
1: -226.2983551, 789.3562012, -182.5741272, 641.6346436, -867.9329834, 971.9302979
2: -123.9032974, 727.5835571, -100.4124756, 592.6696167, -716.5729370, 827.9960327
3: -163.1528168, 1085.4044189, -132.4341888, 880.7500610, -1043.9027100, 1217.8386230
4: -230.3784637, 887.1671143, -187.1121521, 721.2974243, -951.6757812, 1074.2792969

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090823, upper bound: 1988.5092018
time: 0.66 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090397, upper bound: 1988.5091964
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -395.6603088, 1699.1223145, -253.6225281, 1068.4840088, -1464.1441650, 1952.7445068
1: -287.8921204, 995.7379761, -182.5741272, 641.6346436, -929.5267334, 1178.3121338
2: -157.2266541, 916.8106689, -100.4124756, 592.6696167, -749.8962402, 1017.2231445
3: -207.2196655, 1370.9768066, -132.4341888, 880.7500610, -1087.9697266, 1503.4110107
4: -292.1338196, 1120.0863037, -187.1121521, 721.2974243, -1013.4310913, 1307.1984863

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090823, upper bound: 1988.5095793
time: 0.69 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5090397, upper bound: 1988.5096413
time: 0.69 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -312.2982483, 1327.3442383, -336.7902222, 1440.2618408, -1752.5600586, 1664.1343994
1: -226.2983551, 789.3562012, -244.2259827, 847.7672729, -1074.0656738, 1033.5821533
2: -123.9032974, 727.5835571, -133.7190552, 781.6755371, -905.5787964, 861.3026123
3: -163.1528168, 1085.4044189, -176.5486908, 1166.8376465, -1329.9903564, 1261.9528809
4: -230.3784637, 887.1671143, -248.8411407, 953.8999023, -1184.2781982, 1136.0081787

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5086582, upper bound: 1988.5087717
time: 0.68 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5088000, upper bound: 1988.5091004
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -395.6603088, 1699.1223145, -336.7902222, 1440.2618408, -1835.9218750, 2035.9123535
1: -287.8921204, 995.7379761, -244.2259827, 847.7672729, -1135.6591797, 1239.9637451
2: -157.2266541, 916.8106689, -133.7190552, 781.6755371, -938.9021606, 1050.5297852
3: -207.2196655, 1370.9768066, -176.5486908, 1166.8376465, -1374.0573730, 1547.5255127
4: -292.1338196, 1120.0863037, -248.8411407, 953.8999023, -1246.0335693, 1368.9272461

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5086582, upper bound: 1988.5095493
time: 0.69 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5088000, upper bound: 1988.5097257
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -312.2982483, 1327.3442383, -319.0867004, 1357.3745117, -1669.6727295, 1646.4306641
1: -226.2983551, 789.3562012, -231.3012390, 805.6589355, -1031.9572754, 1020.6574097
2: -123.9032974, 727.5835571, -126.6064606, 742.3884277, -866.2917480, 854.1900024
3: -163.1528168, 1085.4044189, -166.7475433, 1107.9818115, -1271.1346436, 1252.1518555
4: -230.3784637, 887.1671143, -235.3379974, 905.6230469, -1136.0014648, 1122.5051270

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4983656, upper bound: 1988.4977290
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4980750, upper bound: 1988.4977290
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -395.6603088, 1699.1223145, -319.0867004, 1357.3745117, -1753.0346680, 2018.2086182
1: -287.8921204, 995.7379761, -231.3012390, 805.6589355, -1093.5509033, 1227.0391846
2: -157.2266541, 916.8106689, -126.6064606, 742.3884277, -899.6151123, 1043.4171143
3: -207.2196655, 1370.9768066, -166.7475433, 1107.9818115, -1315.2014160, 1537.7243652
4: -292.1338196, 1120.0863037, -235.3379974, 905.6230469, -1197.7567139, 1355.4243164

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4983656, upper bound: 1988.4977290
time: 1.11 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4980750, upper bound: 1988.4977290
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -312.2982483, 1327.3442383, -403.6055603, 1733.4029541, -2045.7011719, 1730.9495850
1: -226.2983551, 789.3562012, -293.6866455, 1014.5359497, -1240.8343506, 1083.0428467
2: -123.9032974, 727.5835571, -160.3482819, 933.9457397, -1057.8487549, 887.9317017
3: -163.1528168, 1085.4044189, -211.4171753, 1396.9808350, -1560.1335449, 1296.8215332
4: -230.3784637, 887.1671143, -297.8879395, 1141.2938232, -1371.6721191, 1185.0550537

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4961916, upper bound: 1988.5017933
time: 0.76 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4967267, upper bound: 1988.4977391
time: 0.93 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -395.6603088, 1699.1223145, -403.6055603, 1733.4029541, -2129.0627441, 2102.7277832
1: -287.8921204, 995.7379761, -293.6866455, 1014.5359497, -1302.4276123, 1289.4245605
2: -157.2266541, 916.8106689, -160.3482819, 933.9457397, -1091.1723633, 1077.1589355
3: -207.2196655, 1370.9768066, -211.4171753, 1396.9808350, -1604.2004395, 1582.3940430
4: -292.1338196, 1120.0863037, -297.8879395, 1141.2938232, -1433.4274902, 1417.9739990

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4982396, upper bound: 1988.4978955
time: 0.79 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.4977290, upper bound: 1988.4978129
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.55 seconds
NS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5116456, upper bound: 1988.5116456
NS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5116456, upper bound: 1988.5116456
NS_A1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5097396, upper bound: 1988.5098649
NS_A1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5094911, upper bound: 1988.5094911
NS_A1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
NS_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
NS_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
NS_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5111901, upper bound: 1988.5112692
NS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123023
NS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123189
NS_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123547
NS_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124222, upper bound: 1988.5123625
NS_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5122993
NS_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5123189
NS_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5123447
NS_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5124358, upper bound: 1988.5123625
NS_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5115054, upper bound: 1988.5112434
NS_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5112864, upper bound: 1988.5113066
NS_A1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5110046, upper bound: 1988.5118881
NS_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5123060, upper bound: 1988.5125025
NS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5118179, upper bound: 1988.5118177
NS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5118179, upper bound: 1988.5118177
NS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5117939, upper bound: 1988.5119127
NS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5118179, upper bound: 1988.5119127
NS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5117195
NS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5117195
NS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5118729
NS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5117036, upper bound: 1988.5118729
NS_A1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5115278, upper bound: 1988.5112760
NS_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5121379, upper bound: 1988.5118371
NS_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5111851, upper bound: 1988.5114292
NS_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5119392, upper bound: 1988.5119051
NS_A2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
NS_A2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
NS_A2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
NS_A2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089596, upper bound: 1988.5090823
NS_A2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090397
NS_A2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090397
NS_A2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090693
NS_A2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089505, upper bound: 1988.5090693
NS_A2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090004, upper bound: 1988.5090825
NS_A2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090004, upper bound: 1988.5090825
NS_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090693, upper bound: 1988.5096413
NS_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090693, upper bound: 1988.5096413
NS_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089271, upper bound: 1988.5090404
NS_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5089271, upper bound: 1988.5090404
NS_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5097331, upper bound: 1988.5097262
NS_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5097331, upper bound: 1988.5097262
NS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090823, upper bound: 1988.5092018
NS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090397, upper bound: 1988.5091964
NS_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090823, upper bound: 1988.5095793
NS_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5090397, upper bound: 1988.5096413
NS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5086582, upper bound: 1988.5087717
NS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5088000, upper bound: 1988.5091004
NS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5086582, upper bound: 1988.5095493
NS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.5088000, upper bound: 1988.5097257
NS_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4983656, upper bound: 1988.4977290
NS_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4980750, upper bound: 1988.4977290
NS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4983656, upper bound: 1988.4977290
NS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4980750, upper bound: 1988.4977290
NS_A2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4961916, upper bound: 1988.5017933
NS_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4967267, upper bound: 1988.4977391
NS_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4982396, upper bound: 1988.4978955
NS_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.55
Output dim: 0, lower bound: -1988.4977290, upper bound: 1988.4978129

## BFS NS instance: NS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -367.2597351, 1577.8812256, -367.2597351, 1577.8812256, -1945.1408691, 1945.1407471
1: -266.1236267, 930.2934570, -266.1236267, 930.2934570, -1196.4171143, 1196.4171143
2: -145.7555542, 856.8742065, -145.7555542, 856.8742065, -1002.6297607, 1002.6297607
3: -193.5350342, 1281.7968750, -193.5350342, 1281.7968750, -1475.3319092, 1475.3319092
4: -271.8663635, 1046.9752197, -271.8663635, 1046.9752197, -1318.8415527, 1318.8415527

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5098235, upper bound: 1988.5102359
time: 0.85 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097807, upper bound: 1988.5104012
time: 0.71 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -367.2597351, 1577.8812256, -358.1880798, 1540.6791992, -1907.9388428, 1936.0689697
1: -266.1236267, 930.2934570, -259.3468018, 904.9525146, -1171.0760498, 1189.6400146
2: -145.7555542, 856.8742065, -142.0380249, 833.3172607, -979.0728149, 998.9121704
3: -193.5350342, 1281.7968750, -188.4001007, 1247.4707031, -1441.0057373, 1470.1968994
4: -271.8663635, 1046.9752197, -264.9048157, 1018.6882324, -1290.5545654, 1311.8800049

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5102190, upper bound: 1988.5106083
time: 0.73 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097807, upper bound: 1988.5104012
time: 0.70 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -341.2267151, 1465.9737549, -352.2810364, 1512.6256104, -1853.8522949, 1818.2547607
1: -247.3908081, 860.1039429, -255.5015564, 888.8181763, -1136.2087402, 1115.6054688
2: -135.4415436, 791.8029175, -139.8533783, 818.3583984, -953.7997437, 931.6562500
3: -179.3059845, 1185.6306152, -185.3265076, 1225.3376465, -1404.6435547, 1370.9570312
4: -252.1025696, 968.5939331, -260.3885498, 1000.5936279, -1252.6961670, 1228.9822998

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5084765, upper bound: 1988.5077880
time: 0.66 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097396, upper bound: 1988.5098649
time: 0.71 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5097396, upper bound: 1988.5098649
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -351.8425598, 1516.2307129, -353.9483032, 1522.4362793, -1874.2788086, 1870.1789551
1: -255.2263336, 886.1033325, -256.7427979, 893.1594849, -1148.3854980, 1142.8459473
2: -139.7197418, 815.3568115, -140.6123352, 821.9009399, -961.6206665, 955.9691162
3: -185.2443695, 1222.5432129, -186.5514832, 1232.0166016, -1417.2609863, 1409.0947266
4: -260.4025574, 997.6814575, -262.1647034, 1005.4132690, -1265.8156738, 1259.8460693

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5094911, upper bound: 1988.5094911
time: 0.71 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5094911, upper bound: 1988.5094911
time: 0.90 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -367.2597351, 1577.8812256, -414.8801270, 1790.7520752, -2158.0117188, 1992.7611084
1: -266.1236267, 930.2934570, -301.1653748, 1047.7479248, -1313.8715820, 1231.4586182
2: -145.7555542, 856.8742065, -164.6259308, 964.1951904, -1109.9505615, 1021.5001221
3: -193.5350342, 1281.7968750, -218.2227173, 1444.4472656, -1637.9822998, 1500.0195312
4: -271.8663635, 1046.9752197, -306.5779114, 1178.6370850, -1450.5034180, 1353.5531006

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_A1

### Relational analysis result of NS_A1_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5118522, upper bound: 1988.5110346
time: 0.82 seconds

## Relational analysis of NS_A1_A1_B1_B2_B1_A1_A2

### Relational analysis result of NS_A1_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5121403, upper bound: 1988.5117786
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -358.1880798, 1540.6791992, -414.8801270, 1790.7520752, -2148.9399414, 1955.5592041
1: -259.3468018, 904.9525146, -301.1653748, 1047.7479248, -1307.0947266, 1206.1175537
2: -142.0380249, 833.3172607, -164.6259308, 964.1951904, -1106.2330322, 997.9430542
3: -188.4001007, 1247.4707031, -218.2227173, 1444.4472656, -1632.8472900, 1465.6933594
4: -264.9048157, 1018.6882324, -306.5779114, 1178.6370850, -1443.5418701, 1325.2661133

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_A1

### Relational analysis result of NS_A1_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5118522, upper bound: 1988.5110346
time: 0.82 seconds

## Relational analysis of NS_A1_A1_B1_B2_B1_A2_A2

### Relational analysis result of NS_A1_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5121403, upper bound: 1988.5117786
time: 0.61 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -366.8061523, 1575.8150635, -410.0518799, 1769.6093750, -2136.4147949, 1985.8669434
1: -265.7897949, 929.0634766, -297.2399597, 1032.2169189, -1298.0067139, 1226.3034668
2: -145.5696106, 855.7653809, -162.4854584, 949.7734375, -1095.3428955, 1018.2508545
3: -193.2890625, 1280.0599365, -215.1368561, 1423.2961426, -1616.5852051, 1495.1967773
4: -271.5040283, 1045.5976562, -302.5274963, 1161.5067139, -1433.0106201, 1348.1250000

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5107438, upper bound: 1988.5107290
time: 0.80 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111271, upper bound: 1988.5111648
time: 0.92 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -358.1880798, 1540.6791992, -410.0518799, 1769.6093750, -2127.7971191, 1950.7310791
1: -259.3468018, 904.9525146, -297.2399597, 1032.2169189, -1291.5635986, 1202.1925049
2: -142.0380249, 833.3172607, -162.4854584, 949.7734375, -1091.8114014, 995.8027344
3: -188.4001007, 1247.4707031, -215.1368561, 1423.2961426, -1611.6961670, 1462.6075439
4: -264.9048157, 1018.6882324, -302.5274963, 1161.5067139, -1426.4114990, 1321.2156982

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_A1

### Relational analysis result of NS_A1_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5105759, upper bound: 1988.5107290
time: 0.80 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2_A2_A2

### Relational analysis result of NS_A1_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111271, upper bound: 1988.5111648
time: 0.90 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -360.6166382, 1548.6755371, -323.3616333, 1382.5936279, -1743.2100830, 1872.0371094
1: -261.1275940, 911.7692871, -234.2932739, 813.6262817, -1074.7537842, 1146.0625000
2: -143.0683899, 839.6314087, -128.3689270, 750.0479736, -893.1162109, 968.0002441
3: -190.0061188, 1256.4957275, -169.4234924, 1119.9161377, -1309.9221191, 1425.9191895
4: -266.9199524, 1026.2280273, -238.8618774, 915.3405151, -1182.2603760, 1265.0897217

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5115558, upper bound: 1988.5113637
time: 0.70 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113284, upper bound: 1988.5114011
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -360.6166382, 1548.6755371, -326.9103394, 1395.1011963, -1755.7176514, 1875.5856934
1: -261.1275940, 911.7692871, -236.4125824, 822.9236450, -1084.0510254, 1148.1817627
2: -143.0683899, 839.6314087, -129.5828857, 759.2948608, -902.3632202, 969.2142334
3: -190.0061188, 1256.4957275, -171.1571655, 1132.1265869, -1322.1326904, 1427.6528320
4: -266.9199524, 1026.2280273, -241.1563721, 926.0340576, -1192.9539795, 1267.3843994

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5115558, upper bound: 1988.5113637
time: 0.61 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113284, upper bound: 1988.5114011
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -367.4802246, 1575.8980713, -323.3616333, 1382.5936279, -1750.0738525, 1899.2595215
1: -266.0974121, 927.9514771, -234.2932739, 813.6262817, -1079.7236328, 1162.2447510
2: -145.7360687, 854.7364502, -128.3689270, 750.0479736, -895.7839355, 983.1052856
3: -193.4766388, 1278.6960449, -169.4234924, 1119.9161377, -1313.3927002, 1448.1195068
4: -271.7196655, 1044.4227295, -238.8618774, 915.3405151, -1187.0600586, 1283.2846680

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111602, upper bound: 1988.5114173
time: 0.77 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113429, upper bound: 1988.5113185
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -367.4802246, 1575.8980713, -326.9103394, 1395.1011963, -1762.5812988, 1902.8081055
1: -266.0974121, 927.9514771, -236.4125824, 822.9236450, -1089.0209961, 1164.3638916
2: -145.7360687, 854.7364502, -129.5828857, 759.2948608, -905.0308838, 984.3192749
3: -193.4766388, 1278.6960449, -171.1571655, 1132.1265869, -1325.6032715, 1449.8532715
4: -271.7196655, 1044.4227295, -241.1563721, 926.0340576, -1197.7536621, 1285.5791016

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5115704, upper bound: 1988.5112202
time: 0.86 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113429, upper bound: 1988.5113185
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -360.6166382, 1548.6755371, -392.8483582, 1687.7443848, -2048.3608398, 1941.5239258
1: -261.1275940, 911.7692871, -285.7554321, 988.8925781, -1250.0201416, 1197.5245361
2: -143.0683899, 839.6314087, -156.0852509, 910.5680542, -1053.6363525, 995.7166138
3: -190.0061188, 1256.4957275, -205.6541748, 1361.3476562, -1551.3537598, 1462.1496582
4: -266.9199524, 1026.2280273, -289.8985901, 1112.4561768, -1379.3760986, 1316.1265869

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111635, upper bound: 1988.5114940
time: 0.70 seconds

## Relational analysis of NS_A1_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113509, upper bound: 1988.5114011
time: 0.77 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -360.6166382, 1548.6755371, -382.7471313, 1641.4199219, -2002.0363770, 1931.4226074
1: -261.1275940, 911.7692871, -277.9400024, 961.5264282, -1222.6539307, 1189.7092285
2: -143.0683899, 839.6314087, -151.9176788, 885.9844971, -1029.0528564, 991.5490723
3: -190.0061188, 1256.4957275, -200.3911438, 1324.0527344, -1514.0588379, 1456.8867188
4: -266.9199524, 1026.2280273, -282.2958374, 1081.5437012, -1348.4635010, 1308.5239258

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111635, upper bound: 1988.5115054
time: 0.88 seconds

## Relational analysis of NS_A1_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113509, upper bound: 1988.5114011
time: 1.02 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -367.4802246, 1575.8980713, -392.8483582, 1687.7443848, -2055.2246094, 1968.7463379
1: -266.0974121, 927.9514771, -285.7554321, 988.8925781, -1254.9899902, 1213.7066650
2: -145.7360687, 854.7364502, -156.0852509, 910.5680542, -1056.3040771, 1010.8216553
3: -193.4766388, 1278.6960449, -205.6541748, 1361.3476562, -1554.8243408, 1484.3500977
4: -271.7196655, 1044.4227295, -289.8985901, 1112.4561768, -1384.1757812, 1334.3212891

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111764, upper bound: 1988.5114108
time: 0.69 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113545, upper bound: 1988.5113202
time: 0.83 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -367.4802246, 1575.8980713, -382.7471313, 1641.4199219, -2008.9001465, 1958.6451416
1: -266.0974121, 927.9514771, -277.9400024, 961.5264282, -1227.6237793, 1205.8914795
2: -145.7360687, 854.7364502, -151.9176788, 885.9844971, -1031.7205811, 1006.6541138
3: -193.4766388, 1278.6960449, -200.3911438, 1324.0527344, -1517.5294189, 1479.0871582
4: -271.7196655, 1044.4227295, -282.2958374, 1081.5437012, -1353.2631836, 1326.7185059

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111764, upper bound: 1988.5114389
time: 0.82 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113545, upper bound: 1988.5113202
time: 0.64 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -409.1972961, 1765.3673096, -359.5274658, 1544.0267334, -1953.2238770, 2124.8947754
1: -296.9673462, 1030.6984863, -260.3295898, 908.8184204, -1205.7857666, 1291.0280762
2: -162.3015747, 948.4084473, -142.6327820, 836.9208984, -999.2224731, 1091.0412598
3: -215.0833588, 1421.4739990, -189.4330902, 1252.4589844, -1467.5423584, 1610.9071045
4: -302.2049255, 1159.8836670, -266.1082764, 1022.9354858, -1325.1403809, 1425.9916992

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112043, upper bound: 1988.5112158
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112043, upper bound: 1988.5112158
time: 1.03 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -412.5324402, 1780.1214600, -360.6166382, 1548.6755371, -1961.2078857, 2140.7377930
1: -299.4470520, 1040.5441895, -261.1275940, 911.7692871, -1211.2163086, 1301.6716309
2: -163.6848907, 957.5258789, -143.0683899, 839.6314087, -1003.3162231, 1100.5942383
3: -216.9601746, 1434.7325439, -190.0061188, 1256.4957275, -1473.4558105, 1624.7386475
4: -304.7831116, 1170.7050781, -266.9199524, 1026.2280273, -1331.0111084, 1437.6250000

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112043, upper bound: 1988.5113066
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112043, upper bound: 1988.5113066
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -406.7685242, 1754.5429688, -355.4763489, 1525.0483398, -1931.8168945, 2110.0190430
1: -295.4182129, 1024.5499268, -258.0070496, 897.4730835, -1192.8909912, 1282.5568848
2: -161.4292450, 942.5147095, -141.2181091, 826.1176758, -987.5469360, 1083.7327881
3: -213.8704681, 1412.8389893, -187.0686188, 1237.2629395, -1451.1334229, 1599.9075928
4: -300.4354858, 1152.5085449, -263.0865173, 1010.2358398, -1310.6713867, 1415.5950928

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5100695, upper bound: 1988.5109090
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5105043, upper bound: 1988.5115126
time: 0.91 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5105043, upper bound: 1988.5118881
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -413.5924072, 1784.1998291, -364.8883362, 1564.8944092, -1978.4868164, 2149.0878906
1: -300.2604675, 1042.6057129, -264.2282104, 920.8178101, -1221.0781250, 1306.8339844
2: -164.0913849, 959.3992920, -144.7318420, 847.9454956, -1012.0368652, 1104.1311035
3: -217.4538727, 1437.5267334, -192.1758270, 1269.1622314, -1486.6160889, 1629.7022705
4: -305.4898071, 1172.9144287, -269.8959045, 1036.3803711, -1341.8698730, 1442.8103027

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5113781, upper bound: 1988.5112521
time: 0.88 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5111563, upper bound: 1988.5113321
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -414.0748291, 1786.5034180, -326.8874817, 1397.1420898, -1811.2167969, 2113.3906250
1: -300.6527710, 1043.1757812, -236.9638062, 822.8157959, -1123.4682617, 1280.1395264
2: -164.2835236, 959.8468018, -129.7891846, 758.9492798, -923.2327881, 1089.6357422
3: -217.6616364, 1438.6409912, -171.2551880, 1132.4133301, -1350.0749512, 1609.8962402
4: -305.8672485, 1173.8259277, -241.5423584, 925.8751221, -1231.7424316, 1415.3682861

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5105152, upper bound: 1988.5101928
time: 0.84 seconds

## Relational analysis of NS_A1_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5104930, upper bound: 1988.5102925
time: 0.85 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -414.0748291, 1786.5034180, -330.0051270, 1410.5595703, -1824.6342773, 2116.5083008
1: -300.6527710, 1043.1757812, -239.1762238, 831.3624878, -1132.0150146, 1282.3518066
2: -164.2835236, 959.8468018, -131.0351868, 766.7129517, -930.9964600, 1090.8819580
3: -217.6616364, 1438.6409912, -172.8560638, 1144.1269531, -1361.7885742, 1611.4970703
4: -305.8672485, 1173.8259277, -243.7138214, 935.4141846, -1241.2813721, 1417.5395508

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B2_A1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5105152, upper bound: 1988.5101928
time: 0.91 seconds

## Relational analysis of NS_A1_A2_B1_B2_A1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5102849, upper bound: 1988.5102925
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -417.4580688, 1801.6279297, -326.8874817, 1397.1420898, -1814.6000977, 2128.5151367
1: -303.1703186, 1053.2519531, -236.9638062, 822.8157959, -1125.9860840, 1290.2156982
2: -165.6922150, 969.1755981, -129.7891846, 758.9492798, -924.6414185, 1098.9648438
3: -219.5612488, 1452.1956787, -171.2551880, 1132.4133301, -1351.9746094, 1623.4509277
4: -308.4869385, 1184.9212646, -241.5423584, 925.8751221, -1234.3620605, 1426.4636230

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5103545, upper bound: 1988.5102878
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5103103, upper bound: 1988.5103941
time: 0.92 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -417.4580688, 1801.6279297, -330.0051270, 1410.5595703, -1828.0175781, 2131.6328125
1: -303.1703186, 1053.2519531, -239.1762238, 831.3624878, -1134.5328369, 1292.4279785
2: -165.6922150, 969.1755981, -131.0351868, 766.7129517, -932.4050903, 1100.2108154
3: -219.5612488, 1452.1956787, -172.8560638, 1144.1269531, -1363.6881104, 1625.0517578
4: -308.4869385, 1184.9212646, -243.7138214, 935.4141846, -1243.9011230, 1428.6346436

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5103545, upper bound: 1988.5102878
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5103103, upper bound: 1988.5103941
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -414.0748291, 1786.5034180, -414.0748291, 1786.5034180, -2200.5781250, 2200.5781250
1: -300.6527710, 1043.1757812, -300.6527710, 1043.1757812, -1343.8282471, 1343.8282471
2: -164.2835236, 959.8468018, -164.2835236, 959.8468018, -1124.1301270, 1124.1300049
3: -217.6616364, 1438.6409912, -217.6616364, 1438.6409912, -1656.3026123, 1656.3026123
4: -305.8672485, 1173.8259277, -305.8672485, 1173.8259277, -1479.6931152, 1479.6931152

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5104970, upper bound: 1988.5112760
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5118489, upper bound: 1988.5117108
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -414.0748291, 1786.5034180, -417.4580688, 1801.6279297, -2215.7026367, 2203.9614258
1: -300.6527710, 1043.1757812, -303.1703186, 1053.2519531, -1353.9045410, 1346.3460693
2: -164.2835236, 959.8468018, -165.6922150, 969.1755981, -1133.4591064, 1125.5389404
3: -217.6616364, 1438.6409912, -219.5612488, 1452.1956787, -1669.8572998, 1658.2021484
4: -305.8672485, 1173.8259277, -308.4869385, 1184.9212646, -1490.7883301, 1482.3128662

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5104970, upper bound: 1988.5112760
time: 0.83 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5118489, upper bound: 1988.5117108
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -417.4580688, 1801.6279297, -414.0748291, 1786.5034180, -2203.9614258, 2215.7026367
1: -303.1703186, 1053.2519531, -300.6527710, 1043.1757812, -1346.3460693, 1353.9045410
2: -165.6922150, 969.1755981, -164.2835236, 959.8468018, -1125.5389404, 1133.4591064
3: -219.5612488, 1452.1956787, -217.6616364, 1438.6409912, -1658.2022705, 1669.8572998
4: -308.4869385, 1184.9212646, -305.8672485, 1173.8259277, -1482.3128662, 1490.7883301

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5108886, upper bound: 1988.5108192
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5116559, upper bound: 1988.5118461
time: 1.05 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -417.4580688, 1801.6279297, -417.4580688, 1801.6279297, -2219.0859375, 2219.0859375
1: -303.1703186, 1053.2519531, -303.1703186, 1053.2519531, -1356.4222412, 1356.4222412
2: -165.6922150, 969.1755981, -165.6922150, 969.1755981, -1134.8677979, 1134.8677979
3: -219.5612488, 1452.1956787, -219.5612488, 1452.1956787, -1671.7569580, 1671.7569580
4: -308.4869385, 1184.9212646, -308.4869385, 1184.9212646, -1493.4080811, 1493.4080811

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5107222, upper bound: 1988.5114325
time: 0.79 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5116560, upper bound: 1988.5118461
time: 0.67 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -401.3492737, 1731.1121826, -391.7552185, 1682.1750488, -2083.5236816, 2122.8669434
1: -291.9624023, 1010.8637085, -285.2627258, 985.7570801, -1277.7194824, 1296.1264648
2: -159.4381866, 929.6323242, -155.7636108, 907.3757324, -1066.8139648, 1085.3959961
3: -210.8210907, 1394.5179443, -205.0729218, 1357.2421875, -1568.0632324, 1599.5908203
4: -296.5444031, 1137.5570068, -289.1729431, 1108.8065186, -1405.3509521, 1426.7299805

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112316, upper bound: 1988.5112760
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112316, upper bound: 1988.5112760
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -411.3985596, 1775.1433105, -398.7873840, 1712.5726318, -2123.9711914, 2173.9301758
1: -298.7272034, 1035.7750244, -290.2246094, 1003.6961060, -1302.4233398, 1325.9995117
2: -163.2498016, 952.8171997, -158.4984436, 924.1687012, -1087.4184570, 1111.3156738
3: -216.3228912, 1428.7243652, -208.7571106, 1382.1323242, -1598.4550781, 1637.4814453
4: -303.9850769, 1165.4807129, -294.3447876, 1129.1307373, -1433.1158447, 1459.8254395

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5112316, upper bound: 1988.5118371
time: 0.85 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5119596, upper bound: 1988.5118371
time: 0.92 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -405.1488037, 1748.2052002, -392.5681458, 1685.7341309, -2090.8828125, 2140.7734375
1: -294.7977295, 1021.7031250, -285.8599243, 987.8507690, -1282.6484375, 1307.5629883
2: -161.0256805, 939.5158081, -156.0898438, 909.3227539, -1070.3483887, 1095.6057129
3: -212.9231110, 1409.4346924, -205.4999847, 1360.2164307, -1573.1395264, 1614.9346924
4: -299.5287170, 1149.5000000, -289.7815552, 1111.1409912, -1410.6695557, 1439.2814941

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5108716, upper bound: 1988.5114292
time: 0.73 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5108716, upper bound: 1988.5114292
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -414.6160583, 1789.6325684, -399.6012878, 1716.1192627, -2130.7353516, 2189.2333984
1: -301.1383667, 1045.4251709, -290.8214722, 1005.8336792, -1306.9720459, 1336.2465820
2: -164.6013336, 961.7515869, -158.8240509, 926.1121826, -1090.7135010, 1120.5755615
3: -218.1556702, 1441.7077637, -209.1835785, 1385.1188965, -1603.2745361, 1650.8912354
4: -306.5065308, 1176.0928955, -294.9515686, 1131.5091553, -1438.0156250, 1471.0444336

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117585, upper bound: 1988.5119051
time: 0.82 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1988.5117585, upper bound: 1988.5119051
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -219.1640625, 917.4870605, -234.5091400, 985.3015137, -1204.4655762, 1151.9960938
1: -157.7118225, 551.6383057, -168.8243561, 591.3866577, -749.0985107, 720.4625244
2: -86.8654709, 509.1175232, -92.9478836, 545.9583740, -632.8238525, 602.0654297
3: -114.1791077, 757.3843994, -122.2828064, 812.0121460, -926.1912231, 879.6671753
4: -161.7084198, 619.8987427, -173.0506439, 664.6170654, -826.3255005, 792.9494019

Time for backsubstitution: 1.96 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.73 + 416.84 = 420.57 seconds
