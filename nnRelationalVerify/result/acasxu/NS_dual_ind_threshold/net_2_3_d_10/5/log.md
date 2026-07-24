## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 1783.0300611210841


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-444.5947571, 1548.2216797, -444.5947571, 1548.2216797, -1992.8161621, 1992.8161621)
1: (-450.9839478, 972.3717651, -450.9839478, 972.3717651, -1423.3557129, 1423.3557129)
2: (-411.3054810, 953.4530640, -411.3054810, 953.4530640, -1364.7585449, 1364.7585449)
3: (-487.6748962, 1164.5722656, -487.6748962, 1164.5722656, -1652.2471924, 1652.2471924)
4: (-566.0885620, 1053.3640137, -566.0885620, 1053.3640137, -1619.4525146, 1619.4525146)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 2.68 = 4.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1783.0478916, upper bound: 1783.0478916

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0415565, upper bound: 1783.0449026
time: 1.05 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0413661, upper bound: 1783.0413661
time: 1.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.42 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 0, lower bound: -1783.0415565, upper bound: 1783.0449026
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.42
Output dim: 0, lower bound: -1783.0413661, upper bound: 1783.0413661

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -419.9491577, 1460.7547607, -436.4644470, 1519.3850098, -1939.3342285, 1897.2192383
1: -425.8788452, 918.2283936, -442.6996460, 954.5236206, -1380.4024658, 1360.9279785
2: -388.2780457, 899.9025269, -403.7214661, 935.7937622, -1324.0715332, 1303.6239014
3: -460.5874939, 1099.5120850, -478.7429810, 1143.1335449, -1603.7208252, 1578.2547607
4: -534.2117920, 994.2152100, -555.5866089, 1033.8365479, -1568.0483398, 1549.8015137

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0334545, upper bound: 1783.0394292
time: 0.83 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
time: 1.02 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -505.3092651, 1747.1318359, -429.8446655, 1495.3983154, -2000.7075195, 2176.9765625
1: -511.7761536, 1098.8765869, -435.6668091, 938.8695679, -1450.6457520, 1534.5434570
2: -464.0763855, 1074.2147217, -396.7877808, 919.6296387, -1383.7060547, 1471.0024414
3: -555.3375244, 1316.5640869, -471.5013428, 1124.3775635, -1679.7150879, 1788.0654297
4: -638.4319458, 1189.4644775, -546.3327026, 1016.4550781, -1654.8869629, 1735.7971191

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0332518, upper bound: 1783.0365677
time: 0.87 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0380017, upper bound: 1783.0380017
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.54 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -1783.0334545, upper bound: 1783.0394292
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -1783.0332518, upper bound: 1783.0365677
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -1783.0380017, upper bound: 1783.0380017

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -388.5021057, 1349.2286377, -389.3547058, 1352.2838135, -1740.7857666, 1738.5833740
1: -393.7440491, 848.6144409, -394.5747681, 850.1501465, -1243.8940430, 1243.1889648
2: -359.1326294, 830.5811157, -359.9565430, 831.9376221, -1191.0701904, 1190.5373535
3: -426.1652527, 1016.3430176, -427.1968079, 1018.3776855, -1444.5427246, 1443.5396729
4: -494.0426941, 918.2666626, -495.3037415, 920.1406860, -1414.1832275, 1413.5701904

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321845, upper bound: 1783.0371250
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
time: 1.07 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -413.3592224, 1438.1618652, -436.3599548, 1520.9099121, -1934.2690430, 1874.5212402
1: -419.2026672, 903.9061890, -442.4592285, 954.1855469, -1373.3881836, 1346.3654785
2: -382.1832886, 885.8966675, -403.2653809, 935.9389038, -1318.1219482, 1289.1621094
3: -453.3074036, 1082.2817383, -478.4354248, 1143.1956787, -1596.5030518, 1560.7171631
4: -525.8378906, 978.5892944, -555.3441772, 1033.8522949, -1559.6899414, 1533.9333496

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0438755
time: 0.80 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -478.3979797, 1652.5197754, -386.8804321, 1344.7387695, -1823.1365967, 2039.4000244
1: -484.2864380, 1039.5904541, -391.8643188, 844.1987305, -1328.4851074, 1431.4545898
2: -439.1901855, 1015.3512573, -357.0350952, 826.0891724, -1265.2792969, 1372.3863525
3: -525.8350830, 1245.6364746, -424.3561096, 1011.1557617, -1536.9908447, 1669.9925537
4: -604.1735840, 1124.5616455, -491.6246033, 913.5283813, -1517.7019043, 1616.1862793

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0098163, upper bound: 1783.0070266
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0329233, upper bound: 1783.0354503
time: 1.02 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -494.1518860, 1706.8345947, -423.5024109, 1473.8757324, -1968.0274658, 2130.3366699
1: -500.1768799, 1074.1672363, -428.8291321, 924.6866455, -1424.8635254, 1502.9963379
2: -453.8429565, 1049.1915283, -390.6888123, 905.5676270, -1359.4105225, 1439.8803711
3: -543.1939087, 1287.2016602, -464.3729858, 1108.1802979, -1651.3739014, 1751.5747070
4: -624.1737671, 1162.1586914, -538.2242432, 1001.2924805, -1625.4663086, 1700.3828125

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0201107, upper bound: 1783.0132186
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0366145, upper bound: 1783.0366145
time: 1.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.51 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0321845, upper bound: 1783.0371250
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0438755
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0098163, upper bound: 1783.0070266
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0329233, upper bound: 1783.0354503
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0201107, upper bound: 1783.0132186
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -1783.0366145, upper bound: 1783.0366145

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -370.5444641, 1286.8553467, -384.0268250, 1333.7058105, -1704.2502441, 1670.8819580
1: -375.4595032, 809.5010376, -389.1553345, 838.5153809, -1213.9748535, 1198.6563721
2: -342.7420959, 791.8787231, -355.0819702, 820.4135132, -1163.1556396, 1146.9606934
3: -406.5063171, 969.7111206, -421.3773193, 1004.5128174, -1411.0189209, 1391.0883789
4: -471.5563354, 875.6832886, -488.6176758, 907.4833374, -1379.0396729, 1364.3007812

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321845, upper bound: 1783.0371250
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321845, upper bound: 1783.0371250
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -453.3980408, 1569.8293457, -375.7026672, 1305.5209961, -1758.9190674, 1945.5318604
1: -459.2045593, 987.4512939, -380.6124878, 820.5278931, -1279.7324219, 1368.0637207
2: -417.6663513, 965.2349243, -347.4339600, 802.5048218, -1220.1711426, 1312.6688232
3: -496.8771057, 1184.5999756, -412.2413940, 983.1297607, -1480.0068359, 1596.8413086
4: -576.2149658, 1070.8068848, -478.2459412, 887.7013550, -1463.9162598, 1549.0528564

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -395.6381226, 1376.6391602, -431.2745056, 1503.2669678, -1898.9050293, 1807.9135742
1: -401.1484375, 865.2772827, -437.2764893, 943.1077271, -1344.2559814, 1302.5533447
2: -365.9910889, 847.6456909, -398.6178589, 924.9737549, -1290.9645996, 1246.2635498
3: -433.8909912, 1036.2770996, -472.8663635, 1129.9926758, -1563.8836670, 1509.1434326
4: -503.6525879, 936.5847778, -548.9766235, 1021.7967529, -1525.4493408, 1485.5611572

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -467.2244263, 1620.2131348, -419.0989685, 1460.2989502, -1927.5233154, 2039.3121338
1: -473.2426758, 1018.5515747, -424.8621521, 916.3959961, -1389.6385498, 1443.4135742
2: -430.6642456, 996.7711792, -387.4810791, 898.2953491, -1328.9595947, 1384.2521973
3: -511.7570190, 1221.6851807, -459.6763611, 1098.2545166, -1610.0114746, 1681.3615723
4: -593.9347534, 1104.6567383, -533.6204834, 992.7221069, -1586.6568604, 1638.2769775

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0438755
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0438755
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -473.3594666, 1630.8686523, -379.1863403, 1318.7785645, -1792.1380615, 2010.0549316
1: -481.1797485, 1028.5979004, -384.0657959, 827.6951294, -1308.8747559, 1412.6633301
2: -435.3088684, 1005.3312988, -349.9857788, 810.0647583, -1245.3736572, 1355.3170166
3: -521.7151489, 1232.7957764, -415.8925476, 991.3872681, -1513.1024170, 1648.6883545
4: -597.8550415, 1115.2441406, -482.0414734, 895.6055298, -1493.4604492, 1597.2855225

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0329233, upper bound: 1783.0354398
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0318905, upper bound: 1783.0354374
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -490.6418457, 1691.0894775, -408.5649719, 1421.6695557, -1912.3112793, 2099.6545410
1: -498.7025146, 1066.8609619, -413.6631470, 892.0806274, -1390.7832031, 1480.5240479
2: -451.5683899, 1043.1568604, -377.0000305, 873.3084106, -1324.8768311, 1420.1568604
3: -540.6549683, 1278.5758057, -447.9307861, 1069.2105713, -1609.8654785, 1726.5065918
4: -619.9623413, 1157.0010986, -519.3138428, 965.6456299, -1585.6077881, 1676.3148193

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0361737
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0366145
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.61 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0321845, upper bound: 1783.0371250
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0321845, upper bound: 1783.0371250
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0440007
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0438755
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0383475, upper bound: 1783.0438755
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0329233, upper bound: 1783.0354398
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0318905, upper bound: 1783.0354374
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0361737
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0366145

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -370.5444641, 1286.8553467, -371.0857239, 1288.5886230, -1659.1330566, 1657.9409180
1: -375.4595032, 809.5010376, -375.9960632, 810.2749023, -1185.7343750, 1185.4970703
2: -342.7420959, 791.8787231, -343.2705383, 792.4560547, -1135.1981201, 1135.1492920
3: -406.5063171, 969.7111206, -407.2419128, 970.8518677, -1377.3581543, 1376.9528809
4: -471.5563354, 875.6832886, -472.3852539, 876.7652588, -1348.3215332, 1348.0683594

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0349919
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0371250
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -370.5444641, 1286.8553467, -454.6180725, 1574.7283936, -1945.2728271, 1741.4731445
1: -375.4595032, 809.5010376, -460.4725037, 990.0717773, -1365.5312500, 1269.9735107
2: -342.7420959, 791.8787231, -418.7968750, 967.8302002, -1310.5722656, 1210.6755371
3: -406.5063171, 969.7111206, -498.3403931, 1187.7213135, -1594.2276611, 1468.0515137
4: -471.5563354, 875.6832886, -577.7756958, 1073.6546631, -1545.2108154, 1453.4587402

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0349919
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0371250
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -453.3980408, 1569.8293457, -371.0857239, 1288.5886230, -1741.9866943, 1940.9149170
1: -459.2045593, 987.4512939, -375.9960632, 810.2749023, -1269.4793701, 1363.4472656
2: -417.6663513, 965.2349243, -343.2705383, 792.4560547, -1210.1224365, 1308.5054932
3: -496.8771057, 1184.5999756, -407.2419128, 970.8518677, -1467.7290039, 1591.8417969
4: -576.2149658, 1070.8068848, -472.3852539, 876.7652588, -1452.9802246, 1543.1921387

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0172216, upper bound: 1783.0158814
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -453.3980408, 1569.8293457, -454.6180725, 1574.7283936, -2028.1264648, 2024.4471436
1: -459.2045593, 987.4512939, -460.4725037, 990.0717773, -1449.2763672, 1447.9237061
2: -417.6663513, 965.2349243, -418.7968750, 967.8302002, -1385.4965820, 1384.0317383
3: -496.8771057, 1184.5999756, -498.3403931, 1187.7213135, -1684.5983887, 1682.9404297
4: -576.2149658, 1070.8068848, -577.7756958, 1073.6546631, -1649.8695068, 1648.5825195

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0172216, upper bound: 1783.0158814
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -395.6381226, 1376.6391602, -415.1087036, 1446.3033447, -1841.9414062, 1791.7476807
1: -401.1484375, 865.2772827, -420.7788391, 907.6912231, -1308.8395996, 1286.0559082
2: -365.9910889, 847.6456909, -383.5251160, 890.0151978, -1256.0063477, 1231.1707764
3: -433.8909912, 1036.2770996, -455.0106201, 1087.4580078, -1521.3489990, 1491.2877197
4: -503.6525879, 936.5847778, -528.1647339, 983.0987549, -1486.7513428, 1464.7495117

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0419260
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0365928, upper bound: 1783.0381616
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -395.6381226, 1376.6391602, -493.5822144, 1705.3973389, -2101.0354004, 1870.2214355
1: -401.1484375, 865.2772827, -499.5321045, 1072.4593506, -1473.6077881, 1364.8090820
2: -365.9910889, 847.6456909, -453.1493225, 1047.7823486, -1413.7734375, 1300.7950439
3: -433.8909912, 1036.2770996, -542.8446655, 1286.0699463, -1719.9609375, 1579.1218262
4: -503.6525879, 936.5847778, -623.2517090, 1161.1148682, -1664.7674561, 1559.8364258

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0419260
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0365928, upper bound: 1783.0381616
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -467.2244263, 1620.2131348, -402.5465393, 1401.9788818, -1869.2031250, 2022.7595215
1: -473.2426758, 1018.5515747, -407.9720154, 880.1345825, -1353.3771973, 1426.5235596
2: -430.6642456, 996.7711792, -372.0025024, 862.4982910, -1293.1625977, 1368.7736816
3: -511.7570190, 1221.6851807, -441.4025574, 1054.6939697, -1566.4509277, 1663.0877686
4: -593.9347534, 1104.6567383, -512.3066406, 953.0955200, -1547.0302734, 1616.9631348

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0417972
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0367446, upper bound: 1783.0382375
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -467.2244263, 1620.2131348, -480.3717957, 1661.2320557, -2128.4565430, 2100.5849609
1: -473.2426758, 1018.5515747, -486.4217834, 1044.3088379, -1517.5515137, 1504.9733887
2: -430.6642456, 996.7711792, -441.0840759, 1020.7759399, -1451.4399414, 1437.8552246
3: -511.7570190, 1221.6851807, -528.2887573, 1252.0629883, -1763.8200684, 1749.9737549
4: -593.9347534, 1104.6567383, -606.9225464, 1130.9235840, -1724.8582764, 1711.5789795

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0417972
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0367446, upper bound: 1783.0382375
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -468.3905640, 1613.4549561, -359.1709290, 1248.6413574, -1717.0319824, 1972.6258545
1: -476.1025391, 1017.7616577, -363.6957703, 783.8158569, -1259.9182129, 1381.4573975
2: -430.7594604, 994.5413208, -331.6672058, 766.5206909, -1197.2800293, 1326.2082520
3: -516.2573242, 1219.8764648, -394.0513000, 939.2201538, -1455.4775391, 1613.9277344
4: -591.6321411, 1103.3957520, -456.8395691, 847.9529419, -1439.5847168, 1560.2349854

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0324767, upper bound: 1783.0315368
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0324767, upper bound: 1783.0354398
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -458.6808167, 1581.0718994, -419.7067566, 1457.0153809, -1915.6961670, 2000.7784424
1: -466.2931213, 997.0509644, -424.7546997, 915.2185059, -1381.5113525, 1421.8056641
2: -421.9665527, 974.2153320, -386.3999939, 895.4342041, -1317.4007568, 1360.6153564
3: -505.6035767, 1195.0739746, -459.7492676, 1096.7008057, -1602.3043213, 1654.8231201
4: -579.7188110, 1080.8011475, -533.0698853, 991.7040405, -1571.4224854, 1613.8710938

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0257535, upper bound: 1783.0333531
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321502
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -485.5966187, 1673.4671631, -388.7231750, 1352.2871094, -1837.8837891, 2062.1899414
1: -493.5395508, 1055.8863525, -393.4429932, 848.6218262, -1342.1613770, 1449.3293457
2: -446.9482727, 1032.2567139, -358.8584290, 830.2459717, -1277.1942139, 1391.1151123
3: -535.1157837, 1265.4724121, -426.2611389, 1017.5057983, -1552.6215820, 1691.7335205
4: -613.6395874, 1145.0020752, -494.3044739, 918.4599609, -1532.0996094, 1639.3065186

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0361738
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -475.8843689, 1640.6311035, -456.1265259, 1582.6715088, -2058.5556641, 2096.7575684
1: -483.7481079, 1034.9080811, -461.2081299, 993.9960938, -1477.7441406, 1496.1159668
2: -438.1076965, 1011.5429688, -419.3567810, 972.5565796, -1410.6640625, 1430.8996582
3: -524.4717407, 1240.4854736, -499.2810364, 1191.9847412, -1716.4565430, 1739.7664795
4: -601.6232910, 1122.1370850, -578.6307373, 1077.2333984, -1678.8566895, 1700.7678223

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0340939, upper bound: 1783.0363907
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.30 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0349919
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0371250
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0349919
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0320039, upper bound: 1783.0371250
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0172216, upper bound: 1783.0158814
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0172216, upper bound: 1783.0158814
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0321987, upper bound: 1783.0371250
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0419260
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0365928, upper bound: 1783.0381616
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0419260
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0365928, upper bound: 1783.0381616
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0417972
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0367446, upper bound: 1783.0382375
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0417972
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0367446, upper bound: 1783.0382375
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0324767, upper bound: 1783.0315368
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0324767, upper bound: 1783.0354398
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0257535, upper bound: 1783.0333531
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321502
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0361738
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0340939, upper bound: 1783.0363907
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.30
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -354.5960999, 1230.3244629, -371.0857239, 1288.5886230, -1643.1846924, 1601.4100342
1: -359.2226257, 774.1882935, -375.9960632, 810.2749023, -1169.4975586, 1150.1843262
2: -327.9598694, 756.7957764, -343.2705383, 792.4560547, -1120.4158936, 1100.0661621
3: -389.1460876, 927.5302734, -407.2419128, 970.8518677, -1359.9979248, 1334.7719727
4: -451.2164307, 837.3272095, -472.3852539, 876.7652588, -1327.9816895, 1309.7124023

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0276612, upper bound: 1783.0334467
time: 4.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0329527, upper bound: 1783.0363849
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -402.9652710, 1404.1981201, -371.0857239, 1288.5886230, -1691.5538330, 1775.2836914
1: -408.4078674, 881.2474976, -375.9960632, 810.2749023, -1218.6826172, 1257.2432861
2: -372.4385071, 863.8676758, -343.2705383, 792.4560547, -1164.8945312, 1207.1381836
3: -441.7192383, 1055.9412842, -407.2419128, 970.8518677, -1412.5710449, 1463.1829834
4: -512.9583130, 954.3428345, -472.3852539, 876.7652588, -1389.7236328, 1426.7280273

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0276612, upper bound: 1783.0362764
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0329527, upper bound: 1783.0391966
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -354.5960999, 1230.3244629, -454.6180725, 1574.7283936, -1929.3244629, 1684.9422607
1: -359.2226257, 774.1882935, -460.4725037, 990.0717773, -1349.2944336, 1234.6607666
2: -327.9598694, 756.7957764, -418.7968750, 967.8302002, -1295.7900391, 1175.5924072
3: -389.1460876, 927.5302734, -498.3403931, 1187.7213135, -1576.8674316, 1425.8706055
4: -451.2164307, 837.3272095, -577.7756958, 1073.6546631, -1524.8708496, 1415.1029053

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259752, upper bound: 1783.0315870
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0318144, upper bound: 1783.0347140
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -402.9652710, 1404.1981201, -454.6180725, 1574.7283936, -1977.6936035, 1858.8159180
1: -408.4078674, 881.2474976, -460.4725037, 990.0717773, -1398.4796143, 1341.7197266
2: -372.4385071, 863.8676758, -418.7968750, 967.8302002, -1340.2685547, 1282.6645508
3: -441.7192383, 1055.9412842, -498.3403931, 1187.7213135, -1629.4405518, 1554.2817383
4: -512.9583130, 954.3428345, -577.7756958, 1073.6546631, -1586.6127930, 1532.1185303

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0259752, upper bound: 1783.0338114
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0318144, upper bound: 1783.0347140
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -423.8577881, 1482.0897217, -363.4806519, 1263.7991943, -1687.6568604, 1845.5703125
1: -429.7337646, 926.8208008, -368.3274231, 793.9136353, -1223.6470947, 1295.1477051
2: -391.5395203, 909.7054443, -336.0782776, 777.0695190, -1168.6090088, 1245.7835693
3: -464.5064697, 1111.1035156, -398.8419495, 951.1995850, -1415.7060547, 1509.9454346
4: -541.3591919, 1006.1580811, -462.7727661, 859.4220581, -1400.7812500, 1468.9307861

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0274059, upper bound: 1783.0350626
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0330792, upper bound: 1783.0384387
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -423.8577881, 1482.0897217, -416.2765503, 1446.4176025, -1870.2752686, 1898.3662109
1: -429.7337646, 926.8208008, -421.7601318, 908.0234375, -1337.7570801, 1348.5805664
2: -391.5395203, 909.7054443, -384.0879517, 888.8494873, -1280.3890381, 1293.7933350
3: -464.5064697, 1111.1035156, -456.3569336, 1088.8952637, -1553.4017334, 1567.4604492
4: -541.3591919, 1006.1580811, -529.7846680, 984.5352173, -1525.8944092, 1535.9426270

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0261867, upper bound: 1783.0338114
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0320147, upper bound: 1783.0369141
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -370.8946533, 1291.4030762, -405.7168884, 1414.0841064, -1784.9787598, 1697.1199951
1: -376.1710815, 811.9562988, -411.3143616, 887.4600830, -1263.6311035, 1223.2706299
2: -343.5343018, 795.5648193, -375.0251160, 870.3001709, -1213.8344727, 1170.5899658
3: -406.8708191, 972.1621094, -444.7489014, 1063.1774902, -1470.0482178, 1416.9107666
4: -472.6137695, 878.7756958, -516.4182129, 961.2274170, -1433.8411865, 1395.1938477

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0419659
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0430737
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -445.4020081, 1550.7475586, -397.5556641, 1384.3072510, -1829.7092285, 1948.3031006
1: -452.6294861, 975.1928101, -402.9368591, 869.2273560, -1321.8568115, 1378.1295166
2: -413.8821716, 956.4993896, -367.4233704, 851.7774048, -1265.6595459, 1323.9227295
3: -488.8147278, 1168.4487305, -435.8141479, 1041.5505371, -1530.3652344, 1604.2629395
4: -568.4031982, 1057.5091553, -505.7610474, 940.9335938, -1509.3367920, 1563.1835938

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0381694
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0382813
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -370.8946533, 1291.4030762, -487.5344238, 1684.2056885, -2055.1000977, 1778.9373779
1: -376.1710815, 811.9562988, -493.3351440, 1059.3560791, -1435.5269775, 1305.2913818
2: -343.5343018, 795.5648193, -447.6162720, 1034.8150635, -1378.3493652, 1243.1810303
3: -406.8708191, 972.1621094, -536.1660767, 1270.3223877, -1677.1932373, 1508.3278809
4: -472.6137695, 878.7756958, -615.5903931, 1146.7110596, -1619.3248291, 1494.3660889

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352024, upper bound: 1783.0402080
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0419260
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -445.4020081, 1550.7475586, -460.9708862, 1595.5236816, -2040.9256592, 2011.7185059
1: -452.6294861, 975.1928101, -466.9300842, 1002.3512573, -1454.8955078, 1442.1226807
2: -413.8821716, 956.4993896, -423.4807739, 980.1077271, -1393.9898682, 1379.9799805
3: -488.8147278, 1168.4487305, -507.1293335, 1201.8819580, -1690.6966553, 1675.5780029
4: -568.4031982, 1057.5091553, -582.6647949, 1085.7628174, -1654.1660156, 1640.1739502

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0378734
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0381616
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -450.9024353, 1563.9761963, -392.6697998, 1368.1169434, -1819.0194092, 1956.6458740
1: -456.6354675, 983.3072510, -398.0116882, 858.8610840, -1315.4965820, 1381.3189697
2: -415.8283997, 962.1726074, -363.0589600, 841.7449341, -1257.5732422, 1325.2314453
3: -493.7169800, 1179.4477539, -430.6176147, 1029.1691895, -1522.8862305, 1610.0650635
4: -573.5116577, 1066.2404785, -499.9628601, 930.0690308, -1503.5806885, 1566.2031250

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0363809, upper bound: 1783.0409292
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0385447, upper bound: 1783.0429362
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -509.4385376, 1771.7482910, -387.9705811, 1349.9064941, -1859.3449707, 2159.7187500
1: -516.5257568, 1114.5417480, -393.1463318, 847.9681396, -1364.4936523, 1507.6881104
2: -472.4851379, 1091.1086426, -358.5947571, 830.4225464, -1302.9072266, 1449.7033691
3: -558.2073975, 1336.8637695, -425.4894104, 1016.3629150, -1574.5699463, 1762.3531494
4: -651.0598755, 1208.4147949, -493.5428772, 917.8697510, -1568.9294434, 1701.9576416

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362503, upper bound: 1783.0380390
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383647, upper bound: 1783.0383647
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -450.9024353, 1563.9761963, -473.1918335, 1636.0386963, -2086.9409180, 2037.1678467
1: -456.6354675, 983.3072510, -479.0883179, 1028.7371826, -1485.3726807, 1462.3955078
2: -415.8283997, 962.1726074, -434.4957581, 1005.3934937, -1421.2216797, 1396.6683350
3: -493.7169800, 1179.4477539, -520.3858032, 1233.3576660, -1727.0747070, 1699.8334961
4: -573.5116577, 1066.2404785, -597.8034058, 1113.8652344, -1687.3768311, 1664.0437012

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352041, upper bound: 1783.0399652
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0417972
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -509.4385376, 1771.7482910, -450.3114014, 1560.1026611, -2069.5412598, 2222.0590820
1: -516.5257568, 1114.5417480, -456.3477173, 979.7077026, -1496.2333984, 1570.8894043
2: -472.4851379, 1091.1086426, -413.8320923, 958.4082031, -1430.8930664, 1504.9406738
3: -558.2073975, 1336.8637695, -495.2709961, 1174.4466553, -1732.6540527, 1832.1347656
4: -651.0598755, 1208.4147949, -569.6361694, 1061.5031738, -1712.5629883, 1778.0509033

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0351661, upper bound: 1783.0379639
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0367446, upper bound: 1783.0382375
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -454.2059021, 1563.7454834, -359.1709290, 1248.6413574, -1702.8471680, 1922.9163818
1: -461.6883240, 986.6500854, -363.6957703, 783.8158569, -1245.5039062, 1350.3458252
2: -417.6288757, 963.7373047, -331.6672058, 766.5206909, -1184.1495361, 1295.4045410
3: -500.7580872, 1182.5876465, -394.0513000, 939.2201538, -1439.9782715, 1576.6389160
4: -573.6844482, 1069.4864502, -456.8395691, 847.9529419, -1421.6372070, 1526.3258057

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0324767, upper bound: 1783.0315368
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0254376, upper bound: 1783.0123082
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0264982, upper bound: 1783.0279234
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0313326
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -491.8795166, 1696.2880859, -359.1709290, 1248.6413574, -1740.5207520, 2055.4589844
1: -500.0223389, 1069.3328857, -363.6957703, 783.8158569, -1283.8378906, 1433.0286865
2: -452.4264832, 1045.9659424, -331.6672058, 766.5206909, -1218.9471436, 1377.6331787
3: -542.1281128, 1282.1462402, -394.0513000, 939.2201538, -1481.3482666, 1676.1975098
4: -621.3480835, 1160.4478760, -456.8395691, 847.9529419, -1469.3007812, 1617.2871094

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0324767, upper bound: 1783.0353069
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0254376, upper bound: 1783.0354398
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0264982, upper bound: 1783.0320379
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0352466
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -446.1933289, 1538.2498779, -419.7067566, 1457.0153809, -1903.2086182, 1957.9564209
1: -453.7132263, 970.0908203, -424.7546997, 915.2185059, -1368.9315186, 1394.8454590
2: -410.5796204, 947.9033203, -386.3999939, 895.4342041, -1306.0137939, 1334.3033447
3: -491.9682007, 1162.7434082, -459.7492676, 1096.7008057, -1588.6689453, 1622.4926758
4: -564.0499268, 1051.5867920, -533.0698853, 991.7040405, -1555.7539062, 1584.6567383

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321502
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321501
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -554.5968018, 1908.2766113, -391.3443604, 1360.6081543, -1915.2047119, 2299.6206055
1: -564.5874634, 1210.0679932, -396.5010986, 854.4075928, -1417.9536133, 1606.5690918
2: -515.1151733, 1180.7308350, -361.1186523, 836.2473145, -1351.3623047, 1541.8494873
3: -613.9132690, 1448.5699463, -429.2714844, 1023.5183105, -1635.5075684, 1877.8413086
4: -704.3701782, 1309.4874268, -497.6284485, 925.7377319, -1630.1079102, 1806.5245361

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321501
time: 1.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0124856, upper bound: 1783.0250253
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -473.8604431, 1632.3419189, -388.7231750, 1352.2871094, -1826.1474609, 2021.0649414
1: -481.5693970, 1030.2668457, -393.4429932, 848.6218262, -1330.1911621, 1423.7098389
2: -436.1738281, 1006.8287964, -358.8584290, 830.2459717, -1266.4197998, 1365.6872559
3: -522.2797852, 1234.8970947, -426.2611389, 1017.5057983, -1539.7856445, 1661.1582031
4: -598.8626099, 1117.0506592, -494.3044739, 918.4599609, -1517.3225098, 1611.3551025

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -558.5107422, 1920.0985107, -388.7231750, 1352.2871094, -1910.7978516, 2308.8217773
1: -566.2781982, 1212.1225586, -393.4429932, 848.6218262, -1414.8997803, 1605.5655518
2: -511.5931091, 1183.7622070, -358.8584290, 830.2459717, -1341.8391113, 1542.6206055
3: -613.4376831, 1453.4951172, -426.2611389, 1017.5057983, -1630.9434814, 1879.7562256
4: -704.5358276, 1315.1362305, -494.3044739, 918.4599609, -1622.9957275, 1809.4406738

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0361738
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0361738
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -463.6236267, 1598.6301270, -456.1265259, 1582.6715088, -2046.2951660, 2054.7563477
1: -471.3917236, 1008.4439697, -461.2081299, 993.9960938, -1465.3878174, 1469.6518555
2: -426.9319153, 985.7647095, -419.3567810, 972.5565796, -1399.4881592, 1405.1209717
3: -511.0774536, 1208.7369385, -499.2810364, 1191.9847412, -1703.0621338, 1708.0179443
4: -586.2481689, 1093.5031738, -578.6307373, 1077.2333984, -1663.4815674, 1672.1337891

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -567.1723022, 1954.3489990, -423.4552917, 1471.7003174, -2038.8725586, 2377.8039551
1: -577.1541138, 1237.6097412, -428.5612488, 923.9992065, -1500.9837646, 1666.1708984
2: -525.9295654, 1208.3693848, -390.0352783, 904.4733276, -1430.4028320, 1598.4045410
3: -627.1679077, 1481.3962402, -464.1114807, 1107.5447998, -1733.7421875, 1945.5076904
4: -719.8764038, 1339.3283691, -537.8455811, 1001.3142090, -1721.1906738, 1877.1737061

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0301580, upper bound: 1783.0332298
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698
time: 0.94 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.34 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0276612, upper bound: 1783.0334467
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0329527, upper bound: 1783.0363849
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0276612, upper bound: 1783.0362764
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0329527, upper bound: 1783.0391966
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0259752, upper bound: 1783.0315870
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0318144, upper bound: 1783.0347140
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0259752, upper bound: 1783.0338114
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0318144, upper bound: 1783.0347140
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0274059, upper bound: 1783.0350626
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0330792, upper bound: 1783.0384387
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0261867, upper bound: 1783.0338114
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0320147, upper bound: 1783.0369141
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0419659
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0430737
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0381694
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0382813
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0352024, upper bound: 1783.0402080
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0419260
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0378734
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0381616
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0363809, upper bound: 1783.0409292
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0385447, upper bound: 1783.0429362
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0362503, upper bound: 1783.0380390
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0383647, upper bound: 1783.0383647
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0352041, upper bound: 1783.0399652
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0368912, upper bound: 1783.0417972
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0351661, upper bound: 1783.0379639
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0367446, upper bound: 1783.0382375
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0264982, upper bound: 1783.0279234
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0313326
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0264982, upper bound: 1783.0320379
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0352466
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321502
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321501
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0257109, upper bound: 1783.0321501
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0124856, upper bound: 1783.0250253
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0361738
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0361738
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0301580, upper bound: 1783.0332298
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1783.0338698, upper bound: 1783.0338698

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -350.1374512, 1214.9858398, -361.2637329, 1254.7673340, -1604.9047852, 1576.2493896
1: -354.7462463, 764.3355103, -366.1192322, 788.5585327, -1143.3048096, 1130.4547119
2: -323.9302368, 747.0905151, -334.3944702, 771.0590820, -1094.9892578, 1081.4849854
3: -384.2922668, 915.8708496, -396.5347290, 945.1630249, -1329.4553223, 1312.4051514
4: -445.5504761, 826.5473633, -459.8999329, 852.9987183, -1298.5491943, 1286.4468994

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0169972, upper bound: 1783.0282293
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0166338, upper bound: 1783.0226133
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -353.3733215, 1225.9704590, -373.0060730, 1294.9168701, -1648.2901611, 1598.9765625
1: -357.9756470, 771.4149780, -378.1611023, 814.1995850, -1172.1752930, 1149.5760498
2: -326.8262634, 754.0294189, -344.9660950, 796.3600464, -1123.1862793, 1098.9954834
3: -387.8043213, 924.2521362, -409.3947449, 975.5523682, -1363.3566895, 1333.6468506
4: -449.6150818, 834.3043823, -474.6195679, 881.3504028, -1330.9653320, 1308.9239502

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0232813, upper bound: 1783.0322191
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0228867, upper bound: 1783.0262711
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -398.6147461, 1389.2434082, -361.2637329, 1254.7673340, -1653.3818359, 1750.5070801
1: -404.0415955, 871.6345825, -366.1192322, 788.5585327, -1192.6000977, 1237.7537842
2: -368.4976501, 854.4028931, -334.3944702, 771.0590820, -1139.5566406, 1188.7973633
3: -436.9852600, 1044.5606689, -396.5347290, 945.1630249, -1382.1483154, 1441.0953369
4: -507.4194336, 943.8282471, -459.8999329, 852.9987183, -1360.4182129, 1403.7281494

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0253017, upper bound: 1783.0361141
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0253068, upper bound: 1783.0361129
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -401.3359375, 1398.4466553, -373.0060730, 1294.9168701, -1696.2525635, 1771.4527588
1: -406.7612305, 877.5958862, -378.1611023, 814.1995850, -1220.9606934, 1255.7569580
2: -370.9447021, 860.2454834, -344.9660950, 796.3600464, -1167.3046875, 1205.2115479
3: -439.9451599, 1051.6251221, -409.3947449, 975.5523682, -1415.4975586, 1461.0198975
4: -510.8468933, 950.3657227, -474.6195679, 881.3504028, -1392.1972656, 1424.9852295

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0312012, upper bound: 1783.0391213
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0312076, upper bound: 1783.0391213
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -350.1374512, 1214.9858398, -446.1711426, 1545.4686279, -1895.6059570, 1661.1568604
1: -354.7462463, 764.3355103, -452.0044861, 971.3084717, -1326.0546875, 1216.3399658
2: -323.9302368, 747.0905151, -411.1310730, 949.2970581, -1273.2272949, 1158.2215576
3: -384.2922668, 915.8708496, -489.1489258, 1165.5152588, -1549.8074951, 1405.0195312
4: -445.5504761, 826.5473633, -567.0037842, 1053.1633301, -1498.7138672, 1393.5507812

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0155805, upper bound: 1783.0269036
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0152126, upper bound: 1783.0213329
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -353.3733215, 1225.9704590, -455.5590210, 1577.7929688, -1931.1662598, 1681.5295410
1: -357.9756470, 771.4149780, -461.6063232, 991.8204346, -1349.7961426, 1233.0212402
2: -326.8262634, 754.0294189, -419.6071472, 969.6205444, -1296.4467773, 1173.6365967
3: -387.8043213, 924.2521362, -499.4291992, 1189.8790283, -1577.6832275, 1423.6812744
4: -449.6150818, 834.3043823, -578.7388306, 1075.7896729, -1525.4044189, 1413.0432129

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0217889, upper bound: 1783.0310984
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0213595, upper bound: 1783.0251256
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -398.6147461, 1389.2434082, -446.1711426, 1545.4686279, -1944.0831299, 1835.4144287
1: -404.0415955, 871.6345825, -452.0044861, 971.3084717, -1375.3499756, 1323.6390381
2: -368.4976501, 854.4028931, -411.1310730, 949.2970581, -1317.7946777, 1265.5339355
3: -436.9852600, 1044.5606689, -489.1489258, 1165.5152588, -1602.5004883, 1533.7095947
4: -507.4194336, 943.8282471, -567.0037842, 1053.1633301, -1560.5827637, 1510.8319092

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0244138, upper bound: 1783.0336766
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0244005, upper bound: 1783.0336279
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -401.3359375, 1398.4466553, -455.5590210, 1577.7929688, -1979.1287842, 1854.0056152
1: -406.7612305, 877.5958862, -461.6063232, 991.8204346, -1398.5815430, 1339.2021484
2: -370.9447021, 860.2454834, -419.6071472, 969.6205444, -1340.5649414, 1279.8526611
3: -439.9451599, 1051.6251221, -499.4291992, 1189.8790283, -1629.8240967, 1551.0543213
4: -510.8468933, 950.3657227, -578.7388306, 1075.7896729, -1586.6362305, 1529.1044922

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0307136, upper bound: 1783.0369140
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0306880, upper bound: 1783.0368530
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -419.8048706, 1468.1218262, -353.6170349, 1229.8261719, -1649.6311035, 1821.7388916
1: -425.6733704, 917.8455811, -358.4129944, 772.0991821, -1197.7724609, 1276.2585449
2: -387.8674011, 900.8682251, -327.1599731, 755.5679321, -1143.4353027, 1228.0280762
3: -460.0859375, 1100.4794922, -388.0892334, 925.3937378, -1385.4797363, 1488.5687256
4: -536.1953735, 996.3502197, -450.2280579, 835.5386963, -1371.7337646, 1446.5782471

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0162005, upper bound: 1783.0299705
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0161586, upper bound: 1783.0289480
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -421.1675110, 1472.4805908, -365.3909912, 1270.0394287, -1691.2069092, 1837.8713379
1: -427.0563049, 920.8126221, -370.4896851, 797.7879028, -1224.8442383, 1291.3022461
2: -389.0823059, 903.6859741, -337.7563171, 780.9200439, -1170.0023193, 1241.4418945
3: -461.5954590, 1104.0117188, -400.9865112, 955.8408813, -1417.4362793, 1504.9980469
4: -537.8592529, 999.5098267, -464.9702148, 863.9686890, -1401.8278809, 1464.4798584

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0229093, upper bound: 1783.0344929
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0228316, upper bound: 1783.0332758
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -419.8048706, 1468.1218262, -407.3485107, 1415.5664062, -1835.3713379, 1875.4703369
1: -425.6733704, 917.8455811, -412.8089600, 888.1838989, -1313.8572998, 1330.6545410
2: -387.8674011, 900.8682251, -376.0060120, 869.2788086, -1257.1462402, 1276.8742676
3: -460.0859375, 1100.4794922, -446.6389160, 1065.4526367, -1525.5385742, 1547.1182861
4: -536.1953735, 996.3502197, -518.4078979, 962.8820190, -1499.0769043, 1514.7580566

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0157921, upper bound: 1783.0295002
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0157489, upper bound: 1783.0283982
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -421.1675110, 1472.4805908, -416.9438782, 1448.6048584, -1869.7723389, 1889.4244385
1: -427.0563049, 920.8126221, -422.6484070, 909.2880859, -1336.3443604, 1343.4610596
2: -389.0823059, 903.6859741, -384.6863403, 890.1093750, -1279.1916504, 1288.3721924
3: -461.5954590, 1104.0117188, -457.1520691, 1090.4780273, -1552.0734863, 1561.1636963
4: -537.8592529, 999.5098267, -530.4467163, 986.0895386, -1523.9482422, 1529.9562988

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0219873, upper bound: 1783.0336248
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0219019, upper bound: 1783.0323083
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -370.8946533, 1291.4030762, -393.6061096, 1372.0916748, -1742.9862061, 1685.0091553
1: -376.1710815, 811.9562988, -398.9753418, 861.0844116, -1237.2553711, 1210.9316406
2: -343.5343018, 795.5648193, -363.9654846, 844.2098389, -1187.7440186, 1159.5302734
3: -406.8708191, 972.1621094, -431.4941101, 1031.7421875, -1438.6130371, 1403.6558838
4: -472.6137695, 878.7756958, -501.2525635, 932.5304565, -1405.1442871, 1380.0283203

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383434, upper bound: 1783.0419089
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0419659
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -370.8946533, 1291.4030762, -478.0037537, 1658.1861572, -2029.0808105, 1769.4068604
1: -376.1710815, 811.9562988, -483.8348083, 1041.4163818, -1417.5872803, 1295.7911377
2: -343.5343018, 795.5648193, -439.8638306, 1019.2927246, -1362.8270264, 1235.4287109
3: -406.8708191, 972.1621094, -523.0938721, 1249.5947266, -1656.4655762, 1495.2556152
4: -472.6137695, 878.7756958, -607.1641846, 1129.6713867, -1602.2851562, 1485.9399414

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0383434, upper bound: 1783.0424500
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0430645
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -445.4020081, 1550.7475586, -385.0281067, 1340.9465332, -1786.3485107, 1935.7756348
1: -452.6294861, 975.1928101, -390.1830750, 841.9517822, -1294.5812988, 1365.3758545
2: -413.8821716, 956.4993896, -355.9884949, 824.8312988, -1238.7135010, 1312.4879150
3: -488.8147278, 1168.4487305, -422.1145630, 1009.0518188, -1497.8665771, 1590.5632324
4: -568.4031982, 1057.5091553, -490.0809937, 911.2945557, -1479.6977539, 1547.5198975

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0381051, upper bound: 1783.0374506
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0381694
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -445.4020081, 1550.7475586, -456.7719421, 1584.4119873, -2029.8139648, 2007.5192871
1: -452.6294861, 975.1928101, -462.2637329, 995.4459229, -1447.8576660, 1437.4565430
2: -413.8821716, 956.4993896, -420.5715942, 974.0122070, -1387.8944092, 1377.0708008
3: -488.8147278, 1168.4487305, -500.1034851, 1194.3859863, -1683.2006836, 1668.5521240
4: -568.4031982, 1057.5091553, -580.2950439, 1079.3602295, -1647.7634277, 1637.1077881

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0381051, upper bound: 1783.0375358
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0382813
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -368.9234009, 1284.7192383, -480.3687134, 1659.4676514, -2028.3911133, 1765.0878906
1: -374.1788635, 807.7479248, -486.0329285, 1043.9162598, -1418.0949707, 1293.7808838
2: -341.7561340, 791.4580078, -441.0789490, 1019.6287231, -1361.3847656, 1232.5363770
3: -404.6945801, 967.0952759, -528.2702637, 1251.7320557, -1656.4266357, 1495.3654785
4: -470.1748962, 874.1718140, -606.5802612, 1129.7546387, -1599.9295654, 1480.7519531

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0351529, upper bound: 1783.0401773
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0351529, upper bound: 1783.0402080
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -369.2351685, 1285.7331543, -515.1064453, 1779.1641846, -2148.3994141, 1800.8395996
1: -374.4981384, 808.3812256, -520.7358398, 1118.9084473, -1493.4066162, 1329.1169434
2: -342.0392151, 792.0660400, -472.5964050, 1091.3721924, -1433.4112549, 1264.6623535
3: -405.0596008, 967.8824463, -565.5351562, 1341.9547119, -1747.0142822, 1533.4176025
4: -470.5480652, 874.8809814, -650.6888428, 1209.5352783, -1680.0833740, 1525.5697021

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0336141, upper bound: 1783.0365670
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0367639, upper bound: 1783.0417102
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -445.4020081, 1550.7475586, -448.2312927, 1550.8658447, -1996.2678223, 1998.9788818
1: -452.6294861, 975.1928101, -453.9140320, 974.4184570, -1426.9672852, 1429.1066895
2: -413.8821716, 956.4993896, -411.8403015, 952.3392944, -1366.2214355, 1368.3397217
3: -488.8147278, 1168.4487305, -493.2107239, 1168.6984863, -1657.5131836, 1661.6594238
4: -568.4031982, 1057.5091553, -566.6119385, 1055.3668213, -1623.7700195, 1624.1210938

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0354411, upper bound: 1783.0371490
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0378659
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -445.4020081, 1550.7475586, -537.7617798, 1852.6429443, -2298.0449219, 2088.2441406
1: -452.6294861, 975.1928101, -543.4125977, 1166.1656494, -1618.1643066, 1518.6052246
2: -413.8821716, 956.4993896, -491.4633484, 1138.7626953, -1552.6448975, 1447.9626465
3: -488.8147278, 1168.4487305, -589.1813354, 1398.5371094, -1887.3518066, 1757.6301270
4: -568.4031982, 1057.5091553, -677.7172241, 1263.7779541, -1832.1811523, 1734.8795166

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0354411, upper bound: 1783.0373775
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0381616
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -449.3633423, 1558.6855469, -384.5331116, 1340.4144287, -1789.7777100, 1943.2183838
1: -455.0716553, 979.9959717, -389.7982178, 841.4143066, -1296.4859619, 1369.7939453
2: -414.4246826, 958.9284058, -355.6894531, 824.7379150, -1239.1622314, 1314.6175537
3: -492.0154724, 1175.4588623, -421.6746521, 1008.2052612, -1500.2207031, 1597.1333008
4: -571.5839844, 1062.6110840, -489.8366394, 911.0842896, -1482.6680908, 1552.4475098

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362114, upper bound: 1783.0366661
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0362114, upper bound: 1783.0409292
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -443.9119263, 1540.2427979, -407.9693604, 1424.9561768, -1868.8681641, 1948.2120361
1: -449.5331421, 968.3586426, -413.4281616, 892.8261719, -1342.3590088, 1381.7867432
2: -409.6064758, 947.5177002, -377.5700989, 874.2658691, -1283.8723145, 1325.0876465
3: -486.0192261, 1161.5306396, -447.1697998, 1071.0631104, -1557.0822754, 1608.7004395
4: -564.9099121, 1049.8520508, -521.1418457, 966.5136108, -1531.4234619, 1570.9938965

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0370026, upper bound: 1783.0404872
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0374072, upper bound: 1783.0404707
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -507.9910583, 1766.7779541, -379.9064636, 1322.5183105, -1830.5093994, 2146.6843262
1: -515.0551147, 1111.4353027, -384.9923096, 830.7092285, -1345.7642822, 1496.4276123
2: -471.1770325, 1088.0615234, -351.2713318, 813.6104736, -1284.7874756, 1439.3327637
3: -556.6145630, 1333.1286621, -416.5949707, 995.5883179, -1552.2028809, 1749.7236328
4: -649.2572632, 1205.0202637, -483.5069275, 899.0590210, -1548.3162842, 1688.5270996

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0361298, upper bound: 1783.0361298
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0361298, upper bound: 1783.0380390
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -503.1260986, 1750.6334229, -402.6914673, 1404.8961182, -1908.0220947, 2153.3244629
1: -510.1456909, 1101.1190186, -408.0035706, 880.6715698, -1390.8172607, 1509.1225586
2: -466.7938538, 1077.9876709, -372.5476685, 861.8056641, -1328.5994873, 1450.5354004
3: -551.2037964, 1320.6810303, -441.4402466, 1056.6807861, -1607.8845215, 1762.1213379
4: -643.3031006, 1193.6215820, -513.9722290, 953.0939331, -1596.3969727, 1707.5937500

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0347295, upper bound: 1783.0327081
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0381976, upper bound: 1783.0381976
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -449.3633423, 1558.6855469, -466.0871582, 1611.5150146, -2060.8784180, 2024.7727051
1: -455.0716553, 979.9959717, -471.8644104, 1013.4359131, -1468.5073242, 1451.8601074
2: -414.4246826, 958.9284058, -428.0356750, 990.3615112, -1404.7856445, 1386.9639893
3: -492.0154724, 1175.4588623, -512.5830688, 1214.9394531, -1706.9549561, 1688.0419922
4: -571.5839844, 1062.6110840, -588.8886108, 1097.0859375, -1668.6697998, 1651.4995117

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0348130, upper bound: 1783.0365365
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0348130, upper bound: 1783.0399652
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -443.9119263, 1540.2427979, -498.2490234, 1722.9190674, -2166.8305664, 2038.4918213
1: -449.5331421, 968.3586426, -504.1037598, 1082.7281494, -1532.2612305, 1472.4624023
2: -409.6064758, 947.5177002, -456.9326172, 1056.8981934, -1466.5042725, 1404.4501953
3: -486.0192261, 1161.5306396, -547.3322144, 1298.3829346, -1784.4020996, 1708.8627930
4: -564.9099121, 1049.8520508, -629.6351929, 1170.9797363, -1735.8896484, 1679.4873047

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0335373, upper bound: 1783.0362519
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0367639, upper bound: 1783.0416030
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -507.9910583, 1766.7779541, -442.8433228, 1534.3973389, -2042.3884277, 2209.6213379
1: -515.0551147, 1111.4353027, -448.8079529, 963.6199341, -1478.6750488, 1560.2432861
2: -471.1770325, 1088.0615234, -407.0476685, 942.6805420, -1413.8575439, 1495.1090088
3: -556.6145630, 1333.1286621, -487.0956116, 1155.0938721, -1711.7082520, 1820.2242432
4: -649.2572632, 1205.0202637, -560.2701416, 1043.9869385, -1693.2441406, 1765.2904053

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0349902, upper bound: 1783.0359810
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0349902, upper bound: 1783.0359811
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -503.1260986, 1750.6334229, -472.3658752, 1636.6309814, -2139.7568359, 2222.9992676
1: -510.1456909, 1101.1190186, -478.4447937, 1026.9484863, -1537.0941162, 1579.5638428
2: -466.7938538, 1077.9876709, -433.5064087, 1003.8631592, -1470.6568604, 1511.4940186
3: -551.2037964, 1320.6810303, -519.1608276, 1231.4615479, -1782.6652832, 1839.8417969
4: -643.3031006, 1193.6215820, -597.4751587, 1112.0242920, -1755.3273926, 1791.0966797

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0311779, upper bound: 1783.0349444
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0325645, upper bound: 1783.0325837
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0366176, upper bound: 1783.0380625
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -452.1547241, 1556.4838867, -361.2634583, 1255.5350342, -1707.6895752, 1917.7473145
1: -459.6120911, 982.0437622, -366.0252686, 788.1878662, -1247.7998047, 1348.0689697
2: -415.7418518, 959.2070923, -333.5520020, 770.8729858, -1186.6147461, 1292.7590332
3: -498.5236206, 1177.1105957, -396.4300537, 944.4650269, -1442.9885254, 1573.5406494
4: -571.0090332, 1064.5053711, -459.3365173, 853.0295410, -1424.0385742, 1523.8417969

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0313326
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0313326
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -487.7510071, 1681.9705811, -349.4010315, 1214.9866943, -1702.7376709, 2031.3714600
1: -495.8648376, 1060.1716309, -353.9042664, 762.2105103, -1258.0753174, 1414.0758057
2: -448.6734009, 1036.9234619, -322.8438721, 745.2062378, -1193.8796387, 1359.7670898
3: -537.6199341, 1271.2790527, -383.4389038, 913.6512451, -1451.2712402, 1654.7180176
4: -616.0717773, 1150.4216309, -444.4169312, 824.2973633, -1440.3690186, 1594.8385010

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0268671, upper bound: 1783.0315954
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0268671, upper bound: 1783.0320379
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -489.5880127, 1688.1722412, -361.2634583, 1255.5350342, -1745.1230469, 2049.4357910
1: -497.7178345, 1064.2161865, -366.0252686, 788.1878662, -1285.9053955, 1430.2414551
2: -450.3405457, 1040.8756104, -333.5520020, 770.8729858, -1221.2133789, 1374.4276123
3: -539.6372070, 1276.0749512, -396.4300537, 944.4650269, -1484.1022949, 1672.5050049
4: -618.4030151, 1154.8582764, -459.3365173, 853.0295410, -1471.4326172, 1614.1947021

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0326024, upper bound: 1783.0347316
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0326024, upper bound: 1783.0352466
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -446.1933289, 1538.2498779, -408.5904236, 1418.4780273, -1864.6712646, 1946.8402100
1: -453.7132263, 970.0908203, -413.4937439, 891.3310547, -1345.0441895, 1383.5844727
2: -410.5796204, 947.9033203, -376.4230652, 871.8692017, -1282.4487305, 1324.3264160
3: -491.9682007, 1162.7434082, -447.5306091, 1067.9993896, -1559.9674072, 1610.2740479
4: -564.0499268, 1051.5867920, -519.2539673, 965.5413818, -1529.5913086, 1570.8408203

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0310372
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0333532
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -446.1933289, 1538.2498779, -489.7971191, 1691.5631104, -2137.7563477, 2028.0467529
1: -453.7132263, 970.0908203, -497.7611694, 1069.5429688, -1523.2562256, 1467.8519287
2: -410.5796204, 947.9033203, -453.7361145, 1043.9504395, -1454.5297852, 1401.6394043
3: -491.9682007, 1162.7434082, -541.0649414, 1280.9931641, -1772.9613037, 1703.6751709
4: -564.0499268, 1051.5867920, -623.6520996, 1157.9628906, -1722.0128174, 1675.2388916

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0310372
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0310372
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -553.0120239, 1902.8767090, -384.1908875, 1336.2885742, -1889.3005371, 2287.0676270
1: -562.9837036, 1206.6761475, -389.2846375, 839.1286011, -1401.0751953, 1595.9608154
2: -513.6862793, 1177.4190674, -354.6428833, 821.3723145, -1335.0584717, 1532.0620117
3: -612.1612549, 1444.4875488, -421.4234619, 1005.1147461, -1615.3542480, 1865.9110107
4: -702.4015503, 1305.7854004, -488.7140808, 909.1140747, -1611.5155029, 1793.9161377

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0169435, upper bound: 1783.0272518
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0168687, upper bound: 1783.0262813
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -473.8604431, 1632.3419189, -385.0281067, 1340.9465332, -1814.8070068, 2017.3699951
1: -481.5693970, 1030.2668457, -390.1830750, 841.9517822, -1323.5212402, 1420.4499512
2: -436.1738281, 1006.8287964, -355.9884949, 824.8312988, -1261.0051270, 1362.8172607
3: -522.2797852, 1234.8970947, -422.1145630, 1009.0518188, -1531.3315430, 1657.0117188
4: -598.8626099, 1117.0506592, -490.0809937, 911.2945557, -1510.1572266, 1607.1315918

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0350146, upper bound: 1783.0334036
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -473.8604431, 1632.3419189, -448.2312927, 1550.8658447, -2024.7261963, 2080.5732422
1: -481.5693970, 1030.2668457, -453.9140320, 974.4184570, -1455.9877930, 1484.1806641
2: -436.1738281, 1006.8287964, -411.8403015, 952.3392944, -1388.5130615, 1418.6690674
3: -522.2797852, 1234.8970947, -493.2107239, 1168.6984863, -1690.9782715, 1728.1077881
4: -598.8626099, 1117.0506592, -566.6119385, 1055.3668213, -1654.2294922, 1683.6625977

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0350146, upper bound: 1783.0334036
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -558.5107422, 1920.0985107, -385.0281067, 1340.9465332, -1899.4572754, 2305.1267090
1: -566.2781982, 1212.1225586, -390.1830750, 841.9517822, -1408.2297363, 1602.3056641
2: -511.5931091, 1183.7622070, -355.9884949, 824.8312988, -1336.4244385, 1539.7507324
3: -613.4376831, 1453.4951172, -422.1145630, 1009.0518188, -1622.4895020, 1875.6096191
4: -704.5358276, 1315.1362305, -490.0809937, 911.2945557, -1615.8303223, 1805.2171631

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0249866, upper bound: 1783.0164029
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0361737
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -558.5107422, 1920.0985107, -448.2312927, 1550.8658447, -2109.3764648, 2368.3298340
1: -566.2781982, 1212.1225586, -453.9140320, 974.4184570, -1540.6965332, 1666.0364990
2: -511.5931091, 1183.7622070, -411.8403015, 952.3392944, -1463.9323730, 1595.6025391
3: -613.4376831, 1453.4951172, -493.2107239, 1168.6984863, -1782.1362305, 1946.7058105
4: -704.5358276, 1315.1362305, -566.6119385, 1055.3668213, -1759.9025879, 1881.7481689

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0249866, upper bound: 1783.0164029
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0361737
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -463.6236267, 1598.6301270, -445.3880310, 1545.1184082, -2008.7420654, 2044.0181885
1: -471.3917236, 1008.4439697, -450.3584290, 970.7478027, -1442.1395264, 1458.8021240
2: -426.9319153, 985.7647095, -409.6915283, 949.5604248, -1376.4921875, 1395.4559326
3: -511.0774536, 1208.7369385, -487.5121460, 1164.0823975, -1675.1596680, 1696.2490234
4: -586.2481689, 1093.5031738, -565.1635742, 1051.7625732, -1638.0107422, 1658.6667480

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0334391, upper bound: 1783.0347494
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0340939, upper bound: 1783.0363907
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -463.6236267, 1598.6301270, -513.1724243, 1778.8511963, -2242.4746094, 2111.8022461
1: -471.3917236, 1008.4439697, -521.2958984, 1121.5737305, -1592.9654541, 1529.7397461
2: -426.9319153, 985.7647095, -474.2212219, 1097.2006836, -1524.1324463, 1459.9857178
3: -511.0774536, 1208.7369385, -565.4969482, 1342.8505859, -1853.9279785, 1774.2338867
4: -586.2481689, 1093.5031738, -652.4999390, 1215.0552979, -1801.3034668, 1746.0031738

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0334391, upper bound: 1783.0347494
time: 1.13 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0340939, upper bound: 1783.0363907
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -565.5212402, 1948.7257080, -416.7383728, 1448.7646484, -2014.2857666, 2365.4633789
1: -575.4838257, 1234.0732422, -421.7964478, 909.5830688, -1484.9022217, 1655.8696289
2: -524.4416504, 1204.9169922, -383.9331970, 890.4259033, -1414.8675537, 1588.8502197
3: -625.3419800, 1477.1452637, -456.7407227, 1090.1875000, -1714.5606689, 1933.8859863
4: -717.8235474, 1335.4685059, -529.4614868, 985.6528320, -1703.4763184, 1864.9299316

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0201760, upper bound: 1783.0288733
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0200710, upper bound: 1783.0282659
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -561.7276001, 1936.1214600, -440.7430115, 1535.5102539, -2097.2377930, 2376.8645020
1: -571.6258545, 1225.9965820, -446.1297607, 962.5903931, -1533.8067627, 1672.1263428
2: -520.9682007, 1197.0109863, -406.0134583, 941.1075439, -1462.0756836, 1603.0244141
3: -621.1488647, 1467.4316406, -482.6005554, 1154.3969727, -1774.5559082, 1950.0322266
4: -713.1274414, 1326.6064453, -561.7135010, 1041.8955078, -1755.0228271, 1888.2241211

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0293463, upper bound: 1783.0328333
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1783.0291062, upper bound: 1783.0291062
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.97 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0169972, upper bound: 1783.0282293
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0166338, upper bound: 1783.0226133
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0232813, upper bound: 1783.0322191
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0228867, upper bound: 1783.0262711
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0253017, upper bound: 1783.0361141
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0253068, upper bound: 1783.0361129
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0312012, upper bound: 1783.0391213
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0312076, upper bound: 1783.0391213
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0155805, upper bound: 1783.0269036
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0152126, upper bound: 1783.0213329
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0217889, upper bound: 1783.0310984
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0213595, upper bound: 1783.0251256
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0244138, upper bound: 1783.0336766
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0244005, upper bound: 1783.0336279
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0307136, upper bound: 1783.0369140
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0306880, upper bound: 1783.0368530
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0162005, upper bound: 1783.0299705
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0161586, upper bound: 1783.0289480
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0229093, upper bound: 1783.0344929
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0228316, upper bound: 1783.0332758
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0157921, upper bound: 1783.0295002
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0157489, upper bound: 1783.0283982
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0219873, upper bound: 1783.0336248
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0219019, upper bound: 1783.0323083
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0383434, upper bound: 1783.0419089
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0419659
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0383434, upper bound: 1783.0424500
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0384104, upper bound: 1783.0430645
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0381051, upper bound: 1783.0374506
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0381694
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0381051, upper bound: 1783.0375358
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0381694, upper bound: 1783.0382813
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0351529, upper bound: 1783.0401773
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0351529, upper bound: 1783.0402080
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0336141, upper bound: 1783.0365670
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0367639, upper bound: 1783.0417102
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0354411, upper bound: 1783.0371490
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0378659
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0354411, upper bound: 1783.0373775
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0355024, upper bound: 1783.0381616
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0362114, upper bound: 1783.0366661
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0362114, upper bound: 1783.0409292
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0370026, upper bound: 1783.0404872
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0374072, upper bound: 1783.0404707
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0361298, upper bound: 1783.0361298
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0361298, upper bound: 1783.0380390
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0347295, upper bound: 1783.0327081
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0381976, upper bound: 1783.0381976
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0348130, upper bound: 1783.0365365
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0348130, upper bound: 1783.0399652
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0335373, upper bound: 1783.0362519
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0367639, upper bound: 1783.0416030
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0349902, upper bound: 1783.0359810
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0349902, upper bound: 1783.0359811
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0325645, upper bound: 1783.0325837
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0366176, upper bound: 1783.0380625
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0313326
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0322337, upper bound: 1783.0313326
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0268671, upper bound: 1783.0315954
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0268671, upper bound: 1783.0320379
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0326024, upper bound: 1783.0347316
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0326024, upper bound: 1783.0352466
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0310372
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0333532
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0310372
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0254353, upper bound: 1783.0310372
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0169435, upper bound: 1783.0272518
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0168687, upper bound: 1783.0262813
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0350146, upper bound: 1783.0334036
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0350146, upper bound: 1783.0334036
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0352105, upper bound: 1783.0352105
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0249866, upper bound: 1783.0164029
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0361737
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0249866, upper bound: 1783.0164029
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0354421, upper bound: 1783.0361737
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0334391, upper bound: 1783.0347494
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0340939, upper bound: 1783.0363907
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0334391, upper bound: 1783.0347494
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0340939, upper bound: 1783.0363907
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0201760, upper bound: 1783.0288733
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0200710, upper bound: 1783.0282659
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0293463, upper bound: 1783.0328333
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.97
Output dim: 0, lower bound: -1783.0291062, upper bound: 1783.0291062

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -348.7622986, 1209.9595947, -372.3951721, 1292.8031006, -1641.5654297, 1582.3547363
1: -353.3539429, 761.4144287, -377.5478821, 812.8771362, -1166.2308350, 1138.9620361
2: -322.6424255, 744.1904907, -344.4125977, 795.0577393, -1117.7001953, 1088.6029053
3: -382.8216858, 912.2678223, -408.7343140, 973.9678345, -1356.7895508, 1321.0019531
4: -443.8030090, 823.4743042, -473.8511047, 879.9152222, -1323.7182617, 1297.3251953

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0199438, upper bound: 1783.0315313
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0229426, upper bound: 1783.0316475
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -388.6953430, 1355.4215088, -356.1957703, 1237.3515625, -1626.0467529, 1711.6173096
1: -394.0676270, 850.2304688, -360.9982910, 777.5819092, -1171.6494141, 1211.2286377
2: -359.4949341, 833.5327148, -329.7993469, 760.3216553, -1119.8165283, 1163.3317871
3: -426.1711731, 1018.9030762, -390.9771729, 932.0015869, -1358.1726074, 1409.8802490
4: -494.9692993, 920.5654297, -453.5457458, 841.0885010, -1336.0578613, 1374.1105957

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0253017, upper bound: 1783.0361129
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0253017, upper bound: 1783.0361129
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -385.0136108, 1340.6046143, -356.2269592, 1237.4165039, -1622.4299316, 1696.8315430
1: -390.2913818, 841.3243408, -361.0548706, 777.6381836, -1167.9295654, 1202.3791504
2: -356.0522766, 824.2190552, -329.8282471, 760.3596191, -1116.4116211, 1154.0472412
3: -422.2565002, 1008.6677246, -391.0603943, 932.0876465, -1354.3441162, 1399.7281494
4: -490.0499878, 911.1420898, -453.5341492, 841.0708008, -1331.1208496, 1364.6762695

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0241827, upper bound: 1783.0308478
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0172123, upper bound: 1783.0308490
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -391.3589172, 1364.4190674, -367.9578552, 1277.5643311, -1668.9232178, 1732.3769531
1: -396.7291870, 856.0631104, -373.0623474, 803.2717896, -1200.0006104, 1229.1252441
2: -361.8914185, 839.2416992, -340.3898926, 785.6589966, -1147.5501709, 1179.6315918
3: -429.0682983, 1025.8088379, -403.8641052, 962.4499512, -1391.5183105, 1429.6729736
4: -498.3277283, 926.9549561, -468.2912292, 869.4821777, -1367.8098145, 1395.2462158

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0279517, upper bound: 1783.0388793
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0308596, upper bound: 1783.0388250
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -387.8710938, 1350.3039551, -367.8558960, 1277.1773682, -1665.0484619, 1718.1599121
1: -393.1569519, 847.5848999, -372.9975281, 803.0435791, -1196.2005615, 1220.5820312
2: -358.6284790, 830.3657227, -340.3022461, 785.4154663, -1144.0439453, 1170.6679688
3: -425.3679199, 1016.0989990, -403.8153076, 962.1985474, -1387.5664062, 1419.9143066
4: -493.6592407, 918.0138550, -468.1133728, 869.1577148, -1362.8167725, 1386.1270752

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0299696, upper bound: 1783.0345773
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1783.0234232, upper bound: 1783.0345573
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -348.7622986, 1209.9595947, -455.0753784, 1576.1025391, -1924.8647461, 1665.0349121
1: -353.3539429, 761.4144287, -461.1217346, 990.7637329, -1344.1175537, 1222.5361328
2: -322.6424255, 744.1904907, -419.1652222, 968.5772095, -1291.2196045, 1163.3553467
3: -382.8216858, 912.2678223, -498.9037781, 1188.6158447, -1571.4375000, 1411.1716309
4: -443.8030090, 823.4743042, -578.1281738, 1074.6472168, -1518.4500732, 1401.6025391

Time for backsubstitution: 1.79 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.35 + 416.38 = 420.72 seconds
