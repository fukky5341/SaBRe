## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 3084.599462796909


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1569.5086670, 2773.5766602, -1569.5086670, 2773.5766602, -4343.0854492, 4343.0854492)
1: (-490.6097412, 1099.5386963, -490.6097412, 1099.5386963, -1590.1484375, 1590.1484375)
2: (-307.9147949, 1094.9610596, -307.9147949, 1094.9610596, -1402.8758545, 1402.8758545)
3: (-652.4367676, 1308.2292480, -652.4367676, 1308.2292480, -1960.6660156, 1960.6660156)
4: (-339.8120422, 1133.2332764, -339.8120422, 1133.2332764, -1473.0451660, 1473.0451660)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.16 + 2.35 = 4.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3084.6303091, upper bound: 3084.6303091

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6302261, upper bound: 3084.6302087
time: 0.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6301852, upper bound: 3084.6301852
time: 0.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.97 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 0, lower bound: -3084.6302261, upper bound: 3084.6302087
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 0, lower bound: -3084.6301852, upper bound: 3084.6301852

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1471.8891602, 2608.2683105, -1513.5014648, 2677.5153809, -4149.4042969, 4121.7690430
1: -461.0383911, 1034.2305908, -473.2620544, 1061.1358643, -1522.1741943, 1507.4925537
2: -289.0705261, 1031.0407715, -296.7979126, 1056.7547607, -1345.8253174, 1327.8386230
3: -613.7186890, 1230.3763428, -629.3562012, 1262.7034912, -1876.4219971, 1859.7322998
4: -319.1829529, 1068.1907959, -327.7245483, 1093.8829346, -1413.0659180, 1395.9152832

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6263891, upper bound: 3084.6268450
time: 0.97 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6270093, upper bound: 3084.6267463
time: 0.94 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1544.0246582, 2729.6240234, -1554.2517090, 2747.2993164, -4291.3242188, 4283.8759766
1: -482.7531738, 1082.2204590, -485.9059143, 1089.1881104, -1571.9411621, 1568.1263428
2: -302.7295227, 1077.6722412, -304.8048096, 1084.6240234, -1387.3533936, 1382.4766846
3: -641.9182739, 1287.5823975, -646.1319580, 1295.8898926, -1937.8081055, 1933.7143555
4: -334.2650146, 1115.3070068, -336.4883423, 1122.5112305, -1456.7762451, 1451.7954102

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6263764, upper bound: 3084.6269701
time: 0.92 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6262830, upper bound: 3084.6270093
time: 0.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.03 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -3084.6263891, upper bound: 3084.6268450
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -3084.6270093, upper bound: 3084.6267463
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -3084.6263764, upper bound: 3084.6269701
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -3084.6262830, upper bound: 3084.6270093

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1388.5378418, 2460.7319336, -1344.4822998, 2377.4975586, -3766.0354004, 3805.2143555
1: -435.1107178, 976.7135010, -419.9771729, 944.1515503, -1379.2622070, 1396.6906738
2: -273.1103516, 973.6558228, -263.4175110, 940.1603394, -1213.2707520, 1237.0733643
3: -579.7278442, 1162.0778809, -558.5626221, 1123.9693604, -1703.6972656, 1720.6403809
4: -301.4309692, 1008.9297485, -291.0951843, 971.7318115, -1273.1628418, 1300.0249023

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6259231, upper bound: 3084.6246843
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257253, upper bound: 3084.6243708
time: 0.88 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1453.9965820, 2577.5727539, -1480.6059570, 2620.3029785, -4074.2993164, 4058.1782227
1: -455.5550842, 1022.2973633, -463.1519470, 1039.1644287, -1494.7193604, 1485.4493408
2: -285.5223389, 1019.1458130, -290.4520264, 1034.8760986, -1320.3984375, 1309.5979004
3: -606.2679443, 1216.1065674, -615.9539185, 1236.5419922, -1842.8099365, 1832.0605469
4: -315.3341370, 1055.7355957, -320.7570496, 1071.3054199, -1386.6395264, 1376.4925537

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248364, upper bound: 3084.6255195
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6245478, upper bound: 3084.6244705
time: 0.92 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1453.6193848, 2571.5590820, -1388.2801514, 2451.9445801, -3905.5639648, 3959.8391113
1: -454.7470398, 1019.9008179, -433.2814941, 973.8925171, -1428.6394043, 1453.1823730
2: -285.5563049, 1015.8515625, -272.1197205, 969.7210693, -1255.2770996, 1287.9713135
3: -605.3583374, 1213.7598877, -576.4294434, 1159.3651123, -1764.7229004, 1790.1893311
4: -315.2242126, 1051.8572998, -300.4143066, 1002.2292480, -1317.4534912, 1352.2716064

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6259231, upper bound: 3084.6257281
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6251248, upper bound: 3084.6242072
time: 0.98 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1522.0770264, 2691.2397461, -1519.4495850, 2686.3635254, -4208.4404297, 4210.6894531
1: -476.0091248, 1067.4632568, -475.2034912, 1065.7510986, -1541.7602539, 1542.6667480
2: -298.5102234, 1062.9735107, -298.1196594, 1061.2824707, -1359.7926025, 1361.0931396
3: -632.9721069, 1270.0006104, -631.9459229, 1267.9691162, -1900.9407959, 1901.9464111
4: -329.6156311, 1100.1385498, -329.1124268, 1098.4227295, -1428.0383301, 1429.2509766

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220767
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6187452, upper bound: 3084.6187452
time: 1.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6259231, upper bound: 3084.6246843
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6257253, upper bound: 3084.6243708
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6248364, upper bound: 3084.6255195
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6245478, upper bound: 3084.6244705
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6259231, upper bound: 3084.6257281
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6251248, upper bound: 3084.6242072
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220767
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.04
Output dim: 0, lower bound: -3084.6187452, upper bound: 3084.6187452

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1280.7658691, 2266.0876465, -1300.9694824, 2297.0708008, -3577.8359375, 3567.0571289
1: -400.4958496, 896.6400757, -405.8454285, 911.3240967, -1311.8199463, 1302.4854736
2: -251.6026306, 894.2377319, -254.6769409, 907.5614014, -1159.1640625, 1148.9146729
3: -533.5979004, 1067.0124512, -539.6300659, 1084.9345703, -1618.5324707, 1606.6425781
4: -277.5353394, 927.6087646, -281.3381348, 938.1801147, -1215.7154541, 1208.9468994

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6258099, upper bound: 3084.6246843
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6258099, upper bound: 3084.6246843
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1409.5510254, 2501.4956055, -1308.4632568, 2314.8891602, -3724.4401855, 3809.9584961
1: -441.5646973, 990.4098511, -408.4656677, 917.4359741, -1359.0004883, 1398.8753662
2: -277.2113647, 988.1425781, -256.1881104, 913.9400024, -1191.1512451, 1244.3305664
3: -588.3942261, 1179.1336670, -543.2615356, 1092.5239258, -1680.9182129, 1722.3951416
4: -306.0040283, 1024.4683838, -283.1249084, 945.1576538, -1251.1616211, 1307.5932617

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257238, upper bound: 3084.6243708
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257238, upper bound: 3084.6243708
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1430.9758301, 2537.6472168, -1470.6553955, 2602.6723633, -4033.6481934, 4008.3027344
1: -448.5110779, 1006.6954956, -460.0630188, 1032.2039795, -1480.7150879, 1466.7585449
2: -281.1750488, 1003.6661377, -288.5603027, 1027.9943848, -1309.1693115, 1292.2264404
3: -597.0690918, 1197.6175537, -611.9309692, 1228.3059082, -1825.3750000, 1809.5482178
4: -310.4815063, 1039.8623047, -318.6429138, 1064.3215332, -1374.8028564, 1358.5051270

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248364, upper bound: 3084.6255195
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248364, upper bound: 3084.6255195
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1526.0377197, 2695.9038086, -1459.6717529, 2582.6215820, -4108.6591797, 4155.5747070
1: -478.5219727, 1070.6894531, -456.6134949, 1024.3698730, -1502.8917236, 1527.3028564
2: -299.5220032, 1066.8051758, -286.3974915, 1020.2363892, -1319.7584229, 1353.2026367
3: -635.8264771, 1272.3562012, -607.2869263, 1218.9586182, -1854.7847900, 1879.6430664
4: -330.6895752, 1104.3614502, -316.2438965, 1056.4370117, -1387.1265869, 1420.6052246

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6245478, upper bound: 3084.6244705
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6245478, upper bound: 3084.6244705
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1347.5354004, 2380.0781250, -1346.8593750, 2375.3005371, -3722.8354492, 3726.9375000
1: -420.6648254, 941.3486938, -419.7322083, 942.5300293, -1363.1948242, 1361.0809326
2: -264.3086548, 937.8557739, -263.8684082, 938.5367432, -1202.8454590, 1201.7239990
3: -559.8751831, 1120.5166016, -558.5147095, 1122.0678711, -1681.9431152, 1679.0310059
4: -291.6369629, 971.7169189, -291.1721497, 970.2588501, -1261.8957520, 1262.8890381

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257162
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257281
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1465.2409668, 2592.3945312, -1350.3034668, 2386.2409668, -3851.4819336, 3942.6979980
1: -457.8080444, 1025.6748047, -421.2406921, 945.9528809, -1403.7609863, 1446.9154053
2: -287.7253418, 1022.5534668, -264.5586853, 942.3357544, -1230.0609131, 1287.1120605
3: -609.7830811, 1221.3778076, -560.3524170, 1126.4842529, -1736.2672119, 1781.7302246
4: -317.5411072, 1059.7623291, -292.0635681, 974.4168091, -1291.9577637, 1351.8258057

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1497.7381592, 2649.1042480, -1518.2441406, 2684.2854004, -4182.0229492, 4167.3486328
1: -468.5217896, 1050.6795654, -474.8340759, 1064.9246826, -1533.4464111, 1525.5136719
2: -293.9061279, 1046.3319092, -297.8914185, 1060.4637451, -1354.3696289, 1344.2230225
3: -623.2193604, 1250.1307373, -631.4635010, 1266.9902344, -1890.2095947, 1881.5942383
4: -324.5085449, 1083.1925049, -328.8599243, 1097.5878906, -1422.0964355, 1412.0522461

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220767
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220767
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1573.4184570, 2777.7290039, -1494.5732422, 2642.4392090, -4215.8574219, 4272.3022461
1: -492.7116089, 1104.8837891, -467.4828796, 1048.6252441, -1541.3369141, 1572.3666992
2: -309.1760864, 1099.4942627, -293.2117310, 1044.1080322, -1353.2838135, 1392.7059326
3: -655.0968018, 1313.6965332, -621.6063232, 1247.4337158, -1902.5305176, 1935.3027344
4: -341.0579834, 1137.5206299, -323.7027588, 1080.5107422, -1421.5687256, 1461.2233887

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6187452, upper bound: 3084.6187452
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6187452, upper bound: 3084.6187452
time: 1.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.33 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6258099, upper bound: 3084.6246843
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6258099, upper bound: 3084.6246843
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6257238, upper bound: 3084.6243708
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6257238, upper bound: 3084.6243708
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6248364, upper bound: 3084.6255195
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6248364, upper bound: 3084.6255195
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6245478, upper bound: 3084.6244705
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6245478, upper bound: 3084.6244705
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257162
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257281
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220767
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220767
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6187452, upper bound: 3084.6187452
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -3084.6187452, upper bound: 3084.6187452

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1280.7658691, 2266.0876465, -1235.7729492, 2181.5710449, -3462.3364258, 3501.8603516
1: -400.4958496, 896.6400757, -384.6885986, 860.1389160, -1260.6347656, 1281.3286133
2: -251.6026306, 894.2377319, -242.1268158, 858.2480469, -1109.8504639, 1136.3645020
3: -533.5979004, 1067.0124512, -512.6699829, 1025.1608887, -1558.7587891, 1579.6823730
4: -277.5353394, 927.6087646, -267.1941833, 889.6706543, -1167.2060547, 1194.8029785

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235671, upper bound: 3084.6238065
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249128, upper bound: 3084.6239278
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1280.7658691, 2266.0876465, -1338.3470459, 2360.5834961, -3641.3486328, 3604.4345703
1: -400.4958496, 896.6400757, -417.0802307, 936.7974854, -1337.2933350, 1313.7203369
2: -251.6026306, 894.2377319, -262.1675415, 932.7509155, -1184.3532715, 1156.4052734
3: -533.5979004, 1067.0124512, -554.9846191, 1115.1882324, -1648.7861328, 1621.9970703
4: -277.5353394, 927.6087646, -289.3200073, 964.2158203, -1241.7509766, 1216.9287109

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235671, upper bound: 3084.6238065
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249128, upper bound: 3084.6239278
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1409.5510254, 2501.4956055, -1241.6719971, 2197.3815918, -3606.9326172, 3743.1674805
1: -441.5646973, 990.4098511, -386.8347778, 865.7549438, -1307.3195801, 1377.2446289
2: -277.2113647, 988.1425781, -243.4109344, 864.0536499, -1141.2650146, 1231.5529785
3: -588.3942261, 1179.1336670, -515.6751709, 1032.2830811, -1620.6771240, 1694.8088379
4: -306.0040283, 1024.4683838, -268.7325439, 896.1911011, -1202.1950684, 1293.2009277

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6007423, upper bound: 3084.6000712
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6007423, upper bound: 3084.5996134
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1409.5510254, 2501.4956055, -1341.0338135, 2370.2185059, -3779.7695312, 3842.5292969
1: -441.5646973, 990.4098511, -418.3566895, 939.6682129, -1381.2329102, 1408.7666016
2: -277.2113647, 988.1425781, -262.6870422, 936.0248413, -1213.2359619, 1250.8292236
3: -588.3942261, 1179.1336670, -556.5010376, 1118.9609375, -1707.3548584, 1735.6347656
4: -306.0040283, 1024.4683838, -290.0379944, 967.8363037, -1273.8403320, 1314.5063477

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6007423, upper bound: 3084.6000712
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235663, upper bound: 3084.6235016
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248647, upper bound: 3084.6236117
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1430.9758301, 2537.6472168, -1435.1505127, 2545.0510254, -3976.0268555, 3972.7973633
1: -448.5110779, 1006.6954956, -449.7777710, 1009.5777588, -1458.0887451, 1456.4732666
2: -281.1750488, 1003.6661377, -281.8625488, 1006.5058594, -1287.6809082, 1285.5286865
3: -597.0690918, 1197.6175537, -598.5499878, 1200.9617920, -1798.0308838, 1796.1674805
4: -310.4815063, 1039.8623047, -311.3069153, 1042.6358643, -1353.1170654, 1351.1690674

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6081159, upper bound: 3084.6140757
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6225067, upper bound: 3084.6231651
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6225015, upper bound: 3084.6226926
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1430.9758301, 2537.6472168, -1500.2601318, 2652.7099609, -4083.6857910, 4037.9072266
1: -448.5110779, 1006.6954956, -469.2753296, 1052.5560303, -1501.0668945, 1475.9707031
2: -281.1750488, 1003.6661377, -294.3404541, 1048.1461182, -1329.3211670, 1298.0065918
3: -597.0690918, 1197.6175537, -624.1179199, 1252.2644043, -1849.3334961, 1821.7353516
4: -310.4815063, 1039.8623047, -324.9897156, 1084.9149170, -1395.3963623, 1364.8520508

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6081159, upper bound: 3084.6140757
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6225067, upper bound: 3084.6231651
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6225015, upper bound: 3084.6226926
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1526.0377197, 2695.9038086, -1420.5667725, 2517.8813477, -4043.9189453, 4116.4707031
1: -478.5219727, 1070.6894531, -445.1415405, 998.7143555, -1477.2363281, 1515.8306885
2: -299.5220032, 1066.8051758, -278.9423523, 995.7543335, -1295.2763672, 1345.7473145
3: -635.8264771, 1272.3562012, -592.2935181, 1188.0086670, -1823.8349609, 1864.6495361
4: -330.6895752, 1104.3614502, -308.0727234, 1031.7286377, -1362.4182129, 1412.4342041

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198657, upper bound: 3084.6217324
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1526.0377197, 2695.9038086, -1486.6966553, 2628.5000000, -4154.5375977, 4182.6005859
1: -478.5219727, 1070.6894531, -465.0055542, 1042.8510742, -1521.3730469, 1535.6950684
2: -299.5220032, 1066.8051758, -291.6439819, 1038.5532227, -1338.0751953, 1358.4489746
3: -635.8264771, 1272.3562012, -618.3553467, 1240.7333984, -1876.5598145, 1890.7115479
4: -330.6895752, 1104.3614502, -322.0122681, 1075.1267090, -1405.8162842, 1426.3737793

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198657, upper bound: 3084.6217324
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1347.5354004, 2380.0781250, -1235.7729492, 2181.5710449, -3529.1059570, 3615.8510742
1: -420.6648254, 941.3486938, -384.6885986, 860.1389160, -1280.8037109, 1326.0373535
2: -264.3086548, 937.8557739, -242.1268158, 858.2480469, -1122.5565186, 1179.9824219
3: -559.8751831, 1120.5166016, -512.6699829, 1025.1608887, -1585.0361328, 1633.1862793
4: -291.6369629, 971.7169189, -267.1941833, 889.6706543, -1181.3076172, 1238.9111328

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257153
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257162
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1347.5354004, 2380.0781250, -1338.3470459, 2360.5834961, -3708.1184082, 3718.4252930
1: -420.6648254, 941.3486938, -417.0802307, 936.7974854, -1357.4620361, 1358.4289551
2: -264.3086548, 937.8557739, -262.1675415, 932.7509155, -1197.0593262, 1200.0233154
3: -559.8751831, 1120.5166016, -554.9846191, 1115.1882324, -1675.0634766, 1675.5010986
4: -291.6369629, 971.7169189, -289.3200073, 964.2158203, -1255.8525391, 1261.0368652

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257243
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257281
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1465.2409668, 2592.3945312, -1241.6719971, 2197.3815918, -3662.6225586, 3834.0661621
1: -457.8080444, 1025.6748047, -386.8347778, 865.7549438, -1323.5629883, 1412.5095215
2: -287.7253418, 1022.5534668, -243.4109344, 864.0536499, -1151.7789307, 1265.9639893
3: -609.7830811, 1221.3778076, -515.6751709, 1032.2830811, -1642.0661621, 1737.0529785
4: -317.5411072, 1059.7623291, -268.7325439, 896.1911011, -1213.7320557, 1328.4948730

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1465.2409668, 2592.3945312, -1341.0338135, 2370.2185059, -3835.4594727, 3933.4282227
1: -457.8080444, 1025.6748047, -418.3566895, 939.6682129, -1397.4763184, 1444.0314941
2: -287.7253418, 1022.5534668, -262.6870422, 936.0248413, -1223.7498779, 1285.2403564
3: -609.7830811, 1221.3778076, -556.5010376, 1118.9609375, -1728.7436523, 1777.8789062
4: -317.5411072, 1059.7623291, -290.0379944, 967.8363037, -1285.3774414, 1349.8002930

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1497.7381592, 2649.1042480, -1442.9174805, 2558.5695801, -4056.3076172, 4092.0214844
1: -468.5217896, 1050.6795654, -452.1521606, 1014.8381348, -1483.3598633, 1502.8316650
2: -293.9061279, 1046.3319092, -283.3224487, 1011.7177734, -1305.6235352, 1329.6541748
3: -623.2193604, 1250.1307373, -601.6433105, 1207.1911621, -1830.4104004, 1851.7740479
4: -324.5085449, 1083.1925049, -312.9393616, 1047.9724121, -1372.4809570, 1396.1318359

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1497.7381592, 2649.1042480, -1508.0346680, 2666.6354980, -4164.3735352, 4157.1386719
1: -468.5217896, 1050.6795654, -471.6849670, 1057.9838867, -1526.5056152, 1522.3645020
2: -293.9061279, 1046.3319092, -295.8126831, 1053.5316162, -1347.4373779, 1342.1442871
3: -623.2193604, 1250.1307373, -627.2502441, 1258.7065430, -1881.9257812, 1877.3809814
4: -324.5085449, 1083.1925049, -326.6379089, 1090.3934326, -1414.9019775, 1409.8302002

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1573.4184570, 2777.7290039, -1415.3221436, 2510.4343262, -4083.8522949, 4193.0512695
1: -492.7116089, 1104.8837891, -443.6344910, 995.9380493, -1488.6496582, 1548.5179443
2: -309.1760864, 1099.4942627, -277.9474792, 992.8273926, -1302.0030518, 1377.4417725
3: -655.0968018, 1313.6965332, -590.3085938, 1184.5935059, -1839.6900635, 1904.0050049
4: -341.0579834, 1137.5206299, -307.0035706, 1028.3703613, -1369.4283447, 1444.5240479

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6163450, upper bound: 3084.6177545
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1573.4184570, 2777.7290039, -1483.7840576, 2623.8076172, -4197.2260742, 4261.5131836
1: -492.7116089, 1104.8837891, -464.1591492, 1041.2960205, -1534.0074463, 1569.0428467
2: -309.1760864, 1099.4942627, -291.0116272, 1036.7889404, -1345.9648438, 1390.5058594
3: -655.0968018, 1313.6965332, -617.1450806, 1238.6921387, -1893.7889404, 1930.8415527
4: -341.0579834, 1137.5206299, -321.3525391, 1072.9230957, -1413.9810791, 1458.8731689

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6163450, upper bound: 3084.6177545
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104
time: 1.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.17 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6235671, upper bound: 3084.6238065
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6249128, upper bound: 3084.6239278
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6235671, upper bound: 3084.6238065
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6249128, upper bound: 3084.6239278
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6007423, upper bound: 3084.6000712
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6007423, upper bound: 3084.5996134
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6235663, upper bound: 3084.6235016
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6248647, upper bound: 3084.6236117
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6225067, upper bound: 3084.6231651
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6225015, upper bound: 3084.6226926
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6225067, upper bound: 3084.6231651
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6225015, upper bound: 3084.6226926
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6198657, upper bound: 3084.6217324
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6198657, upper bound: 3084.6217324
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257153
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257162
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257243
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6257194, upper bound: 3084.6257281
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242072
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6163450, upper bound: 3084.6177545
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6163450, upper bound: 3084.6177545
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1262.3657227, 2234.8601074, -1223.7115479, 2161.5983887, -3423.9636230, 3458.5717773
1: -394.8535461, 884.1936035, -381.0585938, 852.1806030, -1247.0341797, 1265.2521973
2: -248.1076660, 881.8456421, -239.8725281, 850.3458252, -1098.4534912, 1121.7180176
3: -526.1575928, 1052.2976074, -507.8958740, 1015.7522583, -1541.9099121, 1560.1933594
4: -273.6807251, 914.8342285, -264.7111206, 881.5664062, -1155.2470703, 1179.5452881

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235341, upper bound: 3084.6237898
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6231059, upper bound: 3084.6237468
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1272.6185303, 2256.2988281, -1223.3878174, 2160.7016602, -3433.3195801, 3479.6865234
1: -398.4830322, 892.3857422, -380.9085083, 851.7256470, -1250.2083740, 1273.2941895
2: -250.3916321, 890.3233032, -239.7724609, 849.9597778, -1100.3514404, 1130.0957031
3: -531.2075806, 1062.3386230, -507.7389526, 1015.2630615, -1546.4700928, 1570.0776367
4: -276.1660156, 924.0716553, -264.5791626, 881.1807251, -1157.3465576, 1188.6507568

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248990, upper bound: 3084.6239059
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6244562, upper bound: 3084.6238832
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1262.3657227, 2234.8601074, -1325.7030029, 2339.4250488, -3601.7905273, 3560.5629883
1: -394.8535461, 884.1936035, -413.2405396, 928.3425903, -1323.1961670, 1297.4339600
2: -248.1076660, 881.8456421, -259.7993164, 924.3645630, -1172.4720459, 1141.6448975
3: -526.1575928, 1052.2976074, -549.9132080, 1105.2083740, -1631.3658447, 1602.2104492
4: -273.6807251, 914.8342285, -286.7006836, 955.5942993, -1229.2750244, 1201.5349121

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235341, upper bound: 3084.6237898
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232802, upper bound: 3084.6236628
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1272.6185303, 2256.2988281, -1326.4564209, 2340.7355957, -3613.3537598, 3582.7553711
1: -398.4830322, 892.3857422, -413.4590454, 928.7635498, -1327.2465820, 1305.8447266
2: -250.3916321, 890.3233032, -259.9159546, 924.8337402, -1175.2253418, 1150.2392578
3: -531.2075806, 1062.3386230, -550.2807007, 1105.7435303, -1636.9506836, 1612.6193848
4: -276.1660156, 924.0716553, -286.8221436, 956.1341553, -1232.3001709, 1210.8937988

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249827, upper bound: 3084.6239059
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6246464, upper bound: 3084.6238799
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1378.4268799, 2446.3327637, -1219.0769043, 2157.1027832, -3535.5297852, 3665.4096680
1: -432.0175781, 969.3272095, -379.7298279, 849.5515747, -1281.5690918, 1349.0570068
2: -271.1667480, 967.0264282, -238.9762115, 848.0366821, -1119.2033691, 1206.0025635
3: -575.7696533, 1153.9420166, -506.3479614, 1013.0562744, -1588.8259277, 1660.2900391
4: -299.3109436, 1002.6282959, -263.7948608, 879.8794556, -1179.1904297, 1266.4230957

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6018676, upper bound: 3084.6012415
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6057936, upper bound: 3084.6052237
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1342.8906250, 2365.6108398, -1191.2147217, 2104.9067383, -3447.7973633, 3556.8254395
1: -417.7297668, 933.6616821, -370.4041748, 828.2362061, -1245.9659424, 1304.0657959
2: -263.7383118, 931.2359009, -233.3739624, 826.5411377, -1090.2792969, 1164.6097412
3: -557.6700439, 1111.3026123, -494.0062866, 987.4129028, -1545.0827637, 1605.3088379
4: -290.5026550, 966.7789917, -257.5599976, 857.7371826, -1148.2397461, 1224.3389893

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6018676, upper bound: 3084.5995175
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6057936, upper bound: 3084.6043509
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1391.0571289, 2470.1950684, -1328.9526367, 2349.9777832, -3741.0346680, 3799.1477051
1: -435.8998413, 977.9138794, -414.6787415, 931.5876465, -1367.4875488, 1392.5926514
2: -273.7055359, 975.7056885, -260.4193726, 927.9988403, -1201.7043457, 1236.1250000
3: -580.9333496, 1164.3685303, -551.6552734, 1109.4232178, -1690.3565674, 1716.0236816
4: -302.1361389, 1011.6669312, -287.5363464, 959.5765991, -1261.7124023, 1299.2032471

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235527, upper bound: 3084.6235002
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232712, upper bound: 3084.6233478
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1397.2583008, 2484.6318359, -1327.7117920, 2347.8598633, -3745.1181641, 3812.3437500
1: -438.2741089, 983.2133179, -414.2945862, 930.6220093, -1368.8959961, 1397.5079346
2: -275.1958618, 981.3370361, -260.1601562, 927.1107788, -1202.3066406, 1241.4971924
3: -584.2955933, 1170.9490967, -551.1566772, 1108.3073730, -1692.6027832, 1722.1055908
4: -303.7447205, 1017.9458008, -287.2289734, 958.7293091, -1262.4739990, 1305.1745605

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248640, upper bound: 3084.6236094
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6246264, upper bound: 3084.6234691
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1399.6290283, 2482.4948730, -1416.9160156, 2512.8166504, -3912.4455566, 3899.4106445
1: -438.9302673, 985.7106323, -444.2115784, 997.4184570, -1436.3487549, 1429.9222412
2: -275.1228027, 982.6247559, -278.3289490, 994.3074341, -1269.4300537, 1260.9537354
3: -584.4451294, 1172.5196533, -591.1787720, 1186.4028320, -1770.8479004, 1763.6984863
4: -303.7804871, 1018.0590820, -307.4039612, 1029.9575195, -1333.7380371, 1325.4630127

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6200449, upper bound: 3084.6207717
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6223180, upper bound: 3084.6233418
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1358.3666992, 2389.6352539, -1375.3380127, 2435.7287598, -3794.0954590, 3764.9731445
1: -422.5007629, 945.1229858, -430.2428589, 964.8338013, -1387.3344727, 1375.3658447
2: -266.4551086, 941.9147339, -269.9299927, 961.8358765, -1228.2910156, 1211.8446045
3: -563.5856934, 1124.1033936, -572.7680054, 1147.5051270, -1711.0908203, 1696.8713379
4: -293.6250610, 977.0136719, -298.0107422, 996.7420654, -1290.3670654, 1275.0244141

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6200449, upper bound: 3084.6203276
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6223180, upper bound: 3084.6227558
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1399.6290283, 2482.4948730, -1477.5883789, 2611.7814941, -4011.4106445, 3960.0827637
1: -438.9302673, 985.7106323, -462.1839905, 1036.4958496, -1475.4261475, 1447.8946533
2: -275.1228027, 982.6247559, -289.9262695, 1032.2145996, -1307.3372803, 1272.5510254
3: -584.4451294, 1172.5196533, -614.8312988, 1233.0949707, -1817.5400391, 1787.3509521
4: -303.7804871, 1018.0590820, -320.0946960, 1068.7520752, -1372.5324707, 1338.1536865

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6200544, upper bound: 3084.6201815
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6223837, upper bound: 3084.6229336
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1358.3666992, 2389.6352539, -1445.2509766, 2553.4721680, -3911.8388672, 3834.8862305
1: -422.5007629, 945.1229858, -451.5534058, 1012.4468384, -1434.9475098, 1396.6762695
2: -266.4551086, 941.9147339, -283.4258728, 1008.0917358, -1274.5468750, 1225.3402100
3: -563.5856934, 1124.1033936, -600.5936279, 1204.4677734, -1768.0534668, 1724.6970215
4: -293.6250610, 977.0136719, -312.8372192, 1043.2073975, -1336.8323975, 1289.8508301

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6009792, upper bound: 3084.5996755
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6200544, upper bound: 3084.6197853
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6223836, upper bound: 3084.6225000
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1505.4884033, 2660.1623535, -1419.4995117, 2516.0407715, -4021.5285645, 4079.6618652
1: -472.2309265, 1056.6666260, -444.8162537, 997.9965210, -1470.2274170, 1501.4829102
2: -295.5944519, 1052.8326416, -278.7385559, 995.0392456, -1290.6336670, 1331.5711670
3: -627.5159912, 1255.6586914, -591.8652954, 1187.1546631, -1814.6706543, 1847.5238037
4: -326.3293152, 1089.9542236, -307.8480225, 1030.9916992, -1357.3208008, 1397.8020020

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6233680, upper bound: 3084.6233680
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6233680, upper bound: 3084.6236782
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1600.8133545, 2823.7729492, -1392.4468994, 2468.8801270, -4069.6933594, 4216.2192383
1: -502.4445190, 1123.8677979, -436.4711914, 979.4906616, -1481.9351807, 1560.3389893
2: -314.6341553, 1118.8626709, -273.4716187, 976.5356445, -1291.1696777, 1392.3342285
3: -666.8442383, 1335.1634521, -580.7581177, 1165.0332031, -1831.8771973, 1915.9216309
4: -347.1289368, 1157.4825439, -302.0348511, 1011.7858276, -1358.9147949, 1459.5173340

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6200604, upper bound: 3084.6188939
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6188950, upper bound: 3084.6188950
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1505.4884033, 2660.1623535, -1485.4754639, 2626.3981934, -4131.8867188, 4145.6376953
1: -472.2309265, 1056.6666260, -464.6314697, 1042.0144043, -1514.2453613, 1521.2980957
2: -295.5944519, 1052.8326416, -291.4127808, 1037.7237549, -1333.3181152, 1344.2453613
3: -627.5159912, 1255.6586914, -617.8671265, 1239.7424316, -1867.2584229, 1873.5258789
4: -326.3293152, 1089.9542236, -321.7563782, 1074.2810059, -1400.6099854, 1411.7105713

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1600.8133545, 2823.7729492, -1461.8375244, 2584.4985352, -4185.3120117, 4285.6098633
1: -502.4445190, 1123.8677979, -457.2882385, 1025.7773438, -1528.2219238, 1581.1560059
2: -314.6341553, 1118.8626709, -286.7425537, 1021.4227295, -1336.0568848, 1405.6051025
3: -666.8442383, 1335.1634521, -608.0307007, 1220.2574463, -1887.1015625, 1943.1940918
4: -347.1289368, 1157.4825439, -316.6098022, 1057.2678223, -1404.3967285, 1474.0922852

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6172019, upper bound: 3084.6175979
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6194286, upper bound: 3084.6204614
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1297.3508301, 2285.1806641, -1235.7729492, 2181.5710449, -3478.9211426, 3520.9533691
1: -403.6644287, 905.7175293, -384.6885986, 860.1389160, -1263.8033447, 1290.4061279
2: -254.0503082, 901.9217529, -242.1268158, 858.2480469, -1112.2982178, 1144.0484619
3: -537.5778809, 1078.3177490, -512.6699829, 1025.1608887, -1562.7387695, 1590.9876709
4: -280.2217102, 932.8769531, -267.1941833, 889.6706543, -1169.8923340, 1200.0711670

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257187, upper bound: 3084.6254049
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249301, upper bound: 3084.6257153
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1406.2770996, 2482.6535645, -1235.7729492, 2181.5710449, -3587.8481445, 3718.4262695
1: -439.0044861, 982.5837402, -384.6885986, 860.1389160, -1299.1434326, 1367.2723389
2: -275.4435730, 978.7155762, -242.1268158, 858.2480469, -1133.6914062, 1220.8424072
3: -583.6162109, 1169.1649170, -512.6699829, 1025.1608887, -1608.7769775, 1681.8348389
4: -304.0253296, 1013.3541260, -267.1941833, 889.6706543, -1193.6960449, 1280.5483398

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6257187, upper bound: 3084.6254049
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249301, upper bound: 3084.6257162
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1297.3508301, 2285.1806641, -1338.3470459, 2360.5834961, -3657.9335938, 3623.5278320
1: -403.6644287, 905.7175293, -417.0802307, 936.7974854, -1340.4616699, 1322.7977295
2: -254.0503082, 901.9217529, -262.1675415, 932.7509155, -1186.8010254, 1164.0893555
3: -537.5778809, 1078.3177490, -554.9846191, 1115.1882324, -1652.7661133, 1633.3023682
4: -280.2217102, 932.8769531, -289.3200073, 964.2158203, -1244.4372559, 1222.1970215

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248337, upper bound: 3084.6242191
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248728, upper bound: 3084.6251800
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1406.2770996, 2482.6535645, -1338.3470459, 2360.5834961, -3766.8601074, 3821.0004883
1: -439.0044861, 982.5837402, -417.0802307, 936.7974854, -1375.8017578, 1399.6639404
2: -275.4435730, 978.7155762, -262.1675415, 932.7509155, -1208.1943359, 1240.8830566
3: -583.6162109, 1169.1649170, -554.9846191, 1115.1882324, -1698.8043213, 1724.1495361
4: -304.0253296, 1013.3541260, -289.3200073, 964.2158203, -1268.2409668, 1302.6740723

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248337, upper bound: 3084.6242191
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248728, upper bound: 3084.6252095
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1387.3601074, 2453.1020508, -1241.6719971, 2197.3815918, -3584.7414551, 3694.7739258
1: -432.6055908, 969.6902466, -386.8347778, 865.7549438, -1298.3604736, 1356.5250244
2: -271.9018250, 967.2741699, -243.4109344, 864.0536499, -1135.9554443, 1210.6849365
3: -575.5704956, 1155.8178711, -515.6751709, 1032.2830811, -1607.8535156, 1671.4930420
4: -300.1176453, 1001.4158325, -268.7325439, 896.1911011, -1196.3087158, 1270.1484375

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242076
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6241347, upper bound: 3084.6241391
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1522.1719971, 2691.2746582, -1241.6719971, 2197.3815918, -3719.5537109, 3932.9462891
1: -475.3417969, 1064.1807861, -386.8347778, 865.7549438, -1341.0966797, 1451.0156250
2: -298.3453674, 1060.9649658, -243.4109344, 864.0536499, -1162.3986816, 1304.3756104
3: -632.2144775, 1267.0095215, -515.6751709, 1032.2830811, -1664.4974365, 1782.6846924
4: -329.3215942, 1099.2470703, -268.7325439, 896.1911011, -1225.5125732, 1367.9796143

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242076
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6241347, upper bound: 3084.6241391
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1387.3601074, 2453.1020508, -1341.0338135, 2370.2185059, -3757.5786133, 3794.1357422
1: -432.6055908, 969.6902466, -418.3566895, 939.6682129, -1372.2738037, 1388.0468750
2: -271.9018250, 967.2741699, -262.6870422, 936.0248413, -1207.9263916, 1229.9611816
3: -575.5704956, 1155.8178711, -556.5010376, 1118.9609375, -1694.5311279, 1712.3188477
4: -300.1176453, 1001.4158325, -290.0379944, 967.8363037, -1267.9539795, 1291.4538574

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6240556, upper bound: 3084.6227882
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6241074, upper bound: 3084.6234796
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1522.1719971, 2691.2746582, -1341.0338135, 2370.2185059, -3892.3906250, 4032.3085938
1: -475.3417969, 1064.1807861, -418.3566895, 939.6682129, -1415.0100098, 1482.5374756
2: -298.3453674, 1060.9649658, -262.6870422, 936.0248413, -1234.3696289, 1323.6519775
3: -632.2144775, 1267.0095215, -556.5010376, 1118.9609375, -1751.1750488, 1823.5104980
4: -329.3215942, 1099.2470703, -290.0379944, 967.8363037, -1297.1579590, 1389.2849121

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6240556, upper bound: 3084.6227882
time: 3.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6241074, upper bound: 3084.6234796
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1474.7172852, 2608.0402832, -1434.1300049, 2543.2868652, -4018.0041504, 4042.1704102
1: -461.3870544, 1034.5942383, -449.4657898, 1008.8870850, -1470.2739258, 1484.0600586
2: -289.5460815, 1030.3859863, -281.6678162, 1005.8175659, -1295.3636475, 1312.0533447
3: -613.9439087, 1231.0495605, -598.1407471, 1200.1405029, -1814.0841064, 1829.1903076
4: -319.6275330, 1066.9720459, -311.0916748, 1041.9277344, -1361.5552979, 1378.0637207

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232951, upper bound: 3084.6243881
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232951, upper bound: 3084.6243881
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1545.5056152, 2728.8027344, -1419.4995117, 2516.0407715, -4061.5463867, 4148.3022461
1: -483.9100342, 1083.1859131, -444.8162537, 997.9965210, -1481.9064941, 1528.0021973
2: -303.4372864, 1078.7272949, -278.7385559, 995.0392456, -1298.4763184, 1357.4658203
3: -643.2431030, 1287.8352051, -591.8652954, 1187.1546631, -1830.3977051, 1879.7003174
4: -334.9207458, 1116.4824219, -307.8480225, 1030.9916992, -1365.9123535, 1424.3302002

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6230047, upper bound: 3084.6236058
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6230047, upper bound: 3084.6236058
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1474.7172852, 2608.0402832, -1499.0737305, 2650.6645508, -4125.3818359, 4107.1137695
1: -461.3870544, 1034.5942383, -468.9116516, 1051.7418213, -1513.1287842, 1503.5058594
2: -289.5460815, 1030.3859863, -294.1164551, 1047.3389893, -1336.8850098, 1324.5021973
3: -613.9439087, 1231.0495605, -623.6436157, 1251.3006592, -1865.2445068, 1854.6931152
4: -319.6275330, 1066.9720459, -324.7414246, 1084.0917969, -1403.7192383, 1391.7135010

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1545.5056152, 2728.8027344, -1485.4754639, 2626.3981934, -4171.9038086, 4214.2778320
1: -483.9100342, 1083.1859131, -464.6314697, 1042.0144043, -1525.9244385, 1547.8173828
2: -303.4372864, 1078.7272949, -291.4127808, 1037.7237549, -1341.1607666, 1370.1398926
3: -643.2431030, 1287.8352051, -617.8671265, 1239.7424316, -1882.9855957, 1905.7023926
4: -334.9207458, 1116.4824219, -321.7563782, 1074.2810059, -1409.2015381, 1438.2387695

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1526.7098389, 2696.9853516, -1415.3221436, 2510.4343262, -4037.1440430, 4112.3076172
1: -478.3357849, 1073.4028320, -443.6344910, 995.9380493, -1474.2736816, 1517.0372314
2: -300.2337341, 1068.0781250, -277.9474792, 992.8273926, -1293.0610352, 1346.0256348
3: -636.3912964, 1276.2258301, -590.3085938, 1184.5935059, -1820.9844971, 1866.5343018
4: -331.2103882, 1105.1285400, -307.0035706, 1028.3703613, -1359.5806885, 1412.1320801

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6155308, upper bound: 3084.6174473
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6178474, upper bound: 3084.6182381
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2223.8762207, 3840.0700684, -1372.1340332, 2433.3144531, -4657.1904297, 5212.2041016
1: -684.4851074, 1521.0748291, -429.8007812, 965.1634521, -1649.6484375, 1950.8756104
2: -429.4893188, 1512.9028320, -269.2540894, 962.1371460, -1391.6264648, 1782.1569824
3: -905.3745117, 1803.3977051, -571.5205688, 1147.4676514, -2052.8420410, 2374.9182129
4: -473.2775574, 1563.7294922, -297.4954224, 996.0777588, -1469.3552246, 1861.2248535

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6154767, upper bound: 3084.6134686
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6175961, upper bound: 3084.6161846
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1526.7098389, 2696.9853516, -1483.7840576, 2623.8076172, -4150.5175781, 4180.7695312
1: -478.3357849, 1073.4028320, -464.1591492, 1041.2960205, -1519.6315918, 1537.5618896
2: -300.2337341, 1068.0781250, -291.0116272, 1036.7889404, -1337.0227051, 1359.0897217
3: -636.3912964, 1276.2258301, -617.1450806, 1238.6921387, -1875.0832520, 1893.3708496
4: -331.2103882, 1105.1285400, -321.3525391, 1072.9230957, -1404.1334229, 1426.4810791

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2223.8762207, 3840.0700684, -1448.1148682, 2561.9042969, -4785.7802734, 5288.1850586
1: -684.4851074, 1521.0748291, -452.9858704, 1016.2014160, -1700.6864014, 1974.0606689
2: -429.4893188, 1512.9028320, -283.6442261, 1011.9388428, -1441.4279785, 1796.5468750
3: -905.3745117, 1803.3977051, -601.3883057, 1208.6948242, -2114.0693359, 2404.7861328
4: -473.2775574, 1563.7294922, -313.3135986, 1046.3823242, -1519.6599121, 1877.0430908

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6134060, upper bound: 3084.6122908
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6152378, upper bound: 3084.6152378
time: 0.95 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.72 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6235341, upper bound: 3084.6237898
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6231059, upper bound: 3084.6237468
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6248990, upper bound: 3084.6239059
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6244562, upper bound: 3084.6238832
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6235341, upper bound: 3084.6237898
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6232802, upper bound: 3084.6236628
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6249827, upper bound: 3084.6239059
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6246464, upper bound: 3084.6238799
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6018676, upper bound: 3084.6012415
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6057936, upper bound: 3084.6052237
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6018676, upper bound: 3084.5995175
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6057936, upper bound: 3084.6043509
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6235527, upper bound: 3084.6235002
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6232712, upper bound: 3084.6233478
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6248640, upper bound: 3084.6236094
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6246264, upper bound: 3084.6234691
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6200449, upper bound: 3084.6207717
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6223180, upper bound: 3084.6233418
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6200449, upper bound: 3084.6203276
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6223180, upper bound: 3084.6227558
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6200544, upper bound: 3084.6201815
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6223837, upper bound: 3084.6229336
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6200544, upper bound: 3084.6197853
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6223836, upper bound: 3084.6225000
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6233680, upper bound: 3084.6233680
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6233680, upper bound: 3084.6236782
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6200604, upper bound: 3084.6188939
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6188950, upper bound: 3084.6188950
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6195534, upper bound: 3084.6205028
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6172019, upper bound: 3084.6175979
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6194286, upper bound: 3084.6204614
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6257187, upper bound: 3084.6254049
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6249301, upper bound: 3084.6257153
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6257187, upper bound: 3084.6254049
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6249301, upper bound: 3084.6257162
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6248337, upper bound: 3084.6242191
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6248728, upper bound: 3084.6251800
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6248337, upper bound: 3084.6242191
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6248728, upper bound: 3084.6252095
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242076
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6241347, upper bound: 3084.6241391
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6249212, upper bound: 3084.6242076
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6241347, upper bound: 3084.6241391
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6240556, upper bound: 3084.6227882
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6241074, upper bound: 3084.6234796
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6240556, upper bound: 3084.6227882
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6241074, upper bound: 3084.6234796
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6232951, upper bound: 3084.6243881
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6232951, upper bound: 3084.6243881
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6230047, upper bound: 3084.6236058
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6230047, upper bound: 3084.6236058
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6170094, upper bound: 3084.6219431
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6198672, upper bound: 3084.6220675
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6155308, upper bound: 3084.6174473
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6178474, upper bound: 3084.6182381
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6154767, upper bound: 3084.6134686
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6175961, upper bound: 3084.6161846
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6153104, upper bound: 3084.6153104
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6134060, upper bound: 3084.6122908
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.72
Output dim: 0, lower bound: -3084.6152378, upper bound: 3084.6152378

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1224.3294678, 2165.1845703, -1163.5842285, 2047.0651855, -3271.3945312, 3328.7687988
1: -382.8482361, 857.6068115, -361.4118958, 808.4509277, -1191.2991943, 1219.0186768
2: -240.6857147, 855.0173340, -227.7287445, 806.1965332, -1046.8822021, 1082.7459717
3: -510.3619995, 1020.3186035, -481.5076904, 963.3287354, -1473.6905518, 1501.8262939
4: -265.3926086, 886.9333496, -251.1193542, 835.0061646, -1100.3988037, 1138.0527344

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235341, upper bound: 3084.6237729
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235341, upper bound: 3084.6237898
time: 1.39 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1232.9141846, 2181.9252930, -1174.3288574, 2073.7871094, -3306.7011719, 3356.2541504
1: -385.6006470, 863.7543945, -365.7147827, 818.3128052, -1203.9134521, 1229.4692383
2: -242.3236847, 861.3636475, -230.3162384, 816.4970093, -1058.8206787, 1091.6798096
3: -514.0200195, 1027.6851807, -487.9074097, 975.1178589, -1489.1376953, 1515.5924072
4: -267.3245850, 893.7033081, -254.1867523, 846.7164917, -1114.0410156, 1147.8897705

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6231059, upper bound: 3084.6237302
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6231059, upper bound: 3084.6237468
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1236.1840820, 2189.3181152, -1166.0581055, 2050.8496094, -3287.0332031, 3355.3759766
1: -386.9400635, 866.8619385, -362.1226196, 809.8128662, -1196.7529297, 1228.9846191
2: -243.2536163, 864.5471191, -228.1729736, 807.5958252, -1050.8492432, 1092.7200928
3: -516.0191650, 1031.6086426, -482.4382324, 965.0391846, -1481.0583496, 1514.0468750
4: -268.1986389, 897.2293701, -251.5920563, 836.4388428, -1104.6374512, 1148.8214111

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248990, upper bound: 3084.6238767
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248990, upper bound: 3084.6239059
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1240.8251953, 2199.1843262, -1175.8565674, 2075.8410645, -3316.6655273, 3375.0407715
1: -388.4917908, 870.3182373, -366.1103516, 819.1219482, -1207.6137695, 1236.4284668
2: -244.1409149, 868.1865845, -230.5486298, 817.3228760, -1061.4637451, 1098.7349854
3: -518.0836792, 1035.7591553, -488.4491272, 976.0908203, -1494.1745605, 1524.2081299
4: -269.2933655, 901.2156372, -254.4335327, 847.5248413, -1116.8182373, 1155.6490479

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6244458, upper bound: 3084.6238219
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6244458, upper bound: 3084.6238832
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1224.3294678, 2165.1845703, -1263.9903564, 2220.8732910, -3445.2021484, 3429.1748047
1: -382.8482361, 857.6068115, -392.8782959, 882.8933105, -1265.7415771, 1250.4851074
2: -240.6857147, 855.0173340, -247.2625122, 878.5684814, -1119.2541504, 1102.2797852
3: -510.3619995, 1020.3186035, -523.0654297, 1050.4300537, -1560.7919922, 1543.3839111
4: -265.3926086, 886.9333496, -272.7680664, 907.9382324, -1173.3308105, 1159.7014160

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6236714, upper bound: 3084.6237729
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6236714, upper bound: 3084.6237898
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1232.9141846, 2181.9252930, -1275.8408203, 2252.5600586, -3485.4741211, 3457.7661133
1: -385.6006470, 863.7543945, -397.9328613, 894.2180786, -1279.8187256, 1261.6872559
2: -242.3236847, 861.3636475, -250.2439423, 890.3892212, -1132.7128906, 1111.6075439
3: -514.0200195, 1027.6851807, -529.7904663, 1064.4626465, -1578.4825439, 1557.4752197
4: -267.3245850, 893.7033081, -276.1093140, 920.6082764, -1187.9328613, 1169.8125000

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232802, upper bound: 3084.6236229
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232802, upper bound: 3084.6236628
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1236.1840820, 2189.3181152, -1266.2689209, 2224.9445801, -3461.1286621, 3455.5866699
1: -386.9400635, 866.8619385, -393.5972290, 884.3414307, -1271.2814941, 1260.4591064
2: -243.2536163, 864.5471191, -247.7086334, 880.0877686, -1123.3414307, 1112.2557373
3: -516.0191650, 1031.6086426, -524.0774536, 1052.2335205, -1568.2524414, 1555.6859131
4: -268.1986389, 897.2293701, -273.2427368, 909.5617065, -1177.7603760, 1170.4721680

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249827, upper bound: 3084.6238714
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6249827, upper bound: 3084.6239059
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1240.8251953, 2199.1843262, -1278.2662354, 2256.5334473, -3497.3581543, 3477.4504395
1: -388.4917908, 870.3182373, -398.6369324, 895.7224731, -1284.2142334, 1268.9552002
2: -244.1409149, 868.1865845, -250.6626434, 891.9328003, -1136.0737305, 1118.8487549
3: -518.0836792, 1035.7591553, -530.7884521, 1066.2730713, -1584.3566895, 1566.5472412
4: -269.2933655, 901.2156372, -276.5642395, 922.2454834, -1191.5388184, 1177.7799072

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6246464, upper bound: 3084.6237228
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232802, upper bound: 3084.6238799
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1339.1026611, 2375.0251465, -1157.3884277, 2039.8717041, -3378.9743652, 3532.4130859
1: -419.6904602, 941.9899902, -359.4335327, 804.1202393, -1223.8105469, 1301.4235840
2: -263.5607910, 939.5025024, -226.4833984, 802.2995605, -1065.8602295, 1165.9855957
3: -559.5919800, 1121.1684570, -479.1689148, 958.7432861, -1518.3352051, 1600.3374023
4: -290.8291931, 974.0764160, -249.8387299, 831.9215088, -1122.7507324, 1223.9145508

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6016162, upper bound: 3084.6012387
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6016162, upper bound: 3084.6012415
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1344.0288086, 2384.6804199, -1160.1624756, 2053.0402832, -3397.0688477, 3544.8427734
1: -421.2429504, 945.5714722, -361.4695129, 809.1632690, -1230.4061279, 1307.0410156
2: -264.4287415, 943.2175903, -227.6151276, 807.7247314, -1072.1531982, 1170.8327637
3: -561.6322021, 1125.3006592, -482.5465698, 964.6783447, -1526.3105469, 1607.8470459
4: -291.8854675, 977.9937134, -251.2431946, 838.4223022, -1130.3077393, 1229.2369385

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6036020, upper bound: 3084.6044126
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6052284, upper bound: 3084.6044684
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1303.7808838, 2295.1491699, -1132.3414307, 1993.6395264, -3297.4201660, 3427.4899902
1: -405.4780273, 906.4467163, -351.1042175, 785.0952148, -1190.5729980, 1257.5509033
2: -256.1780701, 903.9825439, -221.4401855, 783.2058716, -1039.3839111, 1125.4227295
3: -541.7026978, 1078.7299805, -468.1689758, 935.8079834, -1477.5107422, 1546.8989258
4: -282.0691223, 938.6146851, -244.2489471, 812.3202515, -1094.3892822, 1182.8634033

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3084.5975092, upper bound: 3084.5967254
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.5975092, upper bound: 3084.5995175
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1310.4692383, 2306.5878906, -1138.3725586, 2010.5656738, -3321.0349121, 3444.9599609
1: -407.4566040, 910.8181763, -353.9262085, 791.7839966, -1199.2406006, 1264.7443848
2: -257.3684082, 908.2921753, -223.1387482, 790.0574341, -1047.4255371, 1131.4307861
3: -544.1362915, 1083.8608398, -472.4956055, 943.6626587, -1487.7989502, 1556.3563232
4: -283.4613342, 943.0181885, -246.2607117, 820.1475830, -1103.6088867, 1189.2789307

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6007107, upper bound: 3084.6007107
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6007107, upper bound: 3084.6043509
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1349.6871338, 2395.4553223, -1267.2377930, 2230.7377930, -3580.4243164, 3662.6931152
1: -422.9457703, 949.1395874, -394.2250061, 886.0729980, -1309.0187988, 1343.3645020
2: -265.6991882, 946.7713013, -247.8638306, 882.0465698, -1147.7457275, 1194.6351318
3: -563.9449463, 1129.8476562, -524.6801758, 1054.5408936, -1618.4857178, 1654.5278320
4: -293.2066345, 981.6852417, -273.5513916, 911.7367554, -1204.9429932, 1255.2365723

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235527, upper bound: 3084.6235002
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6235527, upper bound: 3084.6235002
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1359.4550781, 2413.5634766, -1275.3000488, 2256.5732422, -3616.0280762, 3688.8635254
1: -426.0147400, 956.1885986, -398.1658020, 894.9007568, -1320.9155273, 1354.3538818
2: -267.5087891, 953.9053345, -250.0732117, 891.4890747, -1158.9976807, 1203.9782715
3: -567.9570312, 1138.1805420, -529.9584351, 1065.6013184, -1633.5583496, 1668.1389160
4: -295.3214111, 989.0850220, -276.1065674, 922.0335083, -1217.3549805, 1265.1914062

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232712, upper bound: 3084.6233478
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6232712, upper bound: 3084.6233478
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1357.3536377, 2412.3212891, -1268.1574707, 2232.4475098, -3589.8010254, 3680.4787598
1: -425.7401123, 955.4356689, -394.5112915, 886.5204468, -1312.2604980, 1349.9470215
2: -267.4463501, 953.3821411, -248.0384064, 882.5827026, -1150.0290527, 1201.4205322
3: -567.8638306, 1137.5997314, -525.0526733, 1055.1563721, -1623.0202637, 1662.6523438
4: -295.1054077, 988.9479370, -273.7200623, 912.3506470, -1207.4560547, 1262.6679688

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248640, upper bound: 3084.6236094
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6248640, upper bound: 3084.6236094
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1365.7186279, 2427.9497070, -1275.9897461, 2257.5686035, -3623.2871094, 3703.9392090
1: -428.3966064, 961.4821167, -398.3435364, 895.1528320, -1323.5491943, 1359.8251953
2: -269.0096436, 959.5209351, -250.1656342, 891.8203735, -1160.8300781, 1209.6862793
3: -571.3215942, 1144.7591553, -530.1906738, 1065.9403076, -1637.2615967, 1674.9497070
4: -296.9343872, 995.3493652, -276.1887817, 922.4495850, -1219.3839111, 1271.5380859

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6246263, upper bound: 3084.6234691
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6246264, upper bound: 3084.6234691
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1389.3413086, 2465.2502441, -1397.9438477, 2481.1044922, -3870.4458008, 3863.1940918
1: -435.8145447, 978.8427124, -438.4651489, 984.7653198, -1420.5795898, 1417.3077393
2: -273.1932983, 975.7948608, -274.7723999, 981.7225952, -1254.9157715, 1250.5672607
3: -580.3403320, 1164.4072266, -583.6068115, 1171.4567871, -1751.7971191, 1748.0140381
4: -301.6513672, 1011.0193481, -303.4818726, 1016.9839478, -1318.6352539, 1314.5012207

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6185328, upper bound: 3084.6187919
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6188095, upper bound: 3084.6191884
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1383.5599365, 2455.9575195, -1404.9532471, 2496.0908203, -3879.6508789, 3860.9104004
1: -434.1076660, 975.0277100, -440.9637451, 990.2761230, -1424.3837891, 1415.9913330
2: -272.1289368, 972.1170654, -276.3456421, 987.5413818, -1259.6702881, 1248.4626465
3: -578.1461792, 1159.9653320, -587.1627808, 1178.3176270, -1756.4637451, 1747.1281738
4: -300.4545288, 1007.3775024, -305.1719360, 1023.5190430, -1323.9736328, 1312.5494385

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6208916, upper bound: 3084.6217465
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6212294, upper bound: 3084.6216458
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1348.1613770, 2372.2373047, -1355.9750977, 2403.2019043, -3751.3632812, 3728.2121582
1: -419.3551636, 938.1928711, -424.3449707, 951.8211670, -1371.1760254, 1362.5377197
2: -264.5062866, 935.0141602, -266.2830505, 948.8921509, -1213.3984375, 1201.2972412
3: -559.4331665, 1115.8956299, -564.9993896, 1132.1230469, -1691.5561523, 1680.8950195
4: -291.4755249, 969.8793945, -293.9888611, 983.4129028, -1274.8884277, 1263.8682861

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6195055, upper bound: 3084.6200666
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6192707, upper bound: 3084.6190840
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1340.9099121, 2360.9370117, -1364.2736816, 2420.5178223, -3761.4277344, 3725.2106934
1: -417.2884521, 933.5311279, -427.2537231, 958.2282715, -1375.5167236, 1360.7849121
2: -263.2182312, 930.4975586, -268.1240845, 955.6221313, -1218.8402100, 1198.6215820
3: -556.7454834, 1110.4521484, -569.0973511, 1140.0802002, -1696.8256836, 1679.5495605
4: -290.0259399, 965.3955688, -295.9719543, 990.8956909, -1280.9216309, 1261.3675537

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6212091, upper bound: 3084.6223163
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6209508, upper bound: 3084.6214536
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1389.3413086, 2465.2502441, -1457.7464600, 2578.3715820, -3967.7128906, 3922.9965820
1: -435.8145447, 978.8427124, -456.1336975, 1023.1235352, -1458.9381104, 1434.9764404
2: -273.1932983, 975.7948608, -286.2083740, 1018.9902344, -1292.1835938, 1262.0031738
3: -580.3403320, 1164.4072266, -606.9022827, 1217.3676758, -1797.7080078, 1771.3095703
4: -301.6513672, 1011.0193481, -315.9923401, 1055.1340332, -1356.7854004, 1327.0117188

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6184535, upper bound: 3084.6199133
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6196250, upper bound: 3084.6200119
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1383.5599365, 2455.9575195, -1465.8989258, 2595.4448242, -3979.0048828, 3921.8559570
1: -434.1076660, 975.0277100, -458.9978638, 1029.4744873, -1463.5821533, 1434.0252686
2: -272.1289368, 972.1170654, -287.9740906, 1025.6223145, -1297.7512207, 1260.0911865
3: -578.1461792, 1159.9653320, -610.8864746, 1225.1932373, -1803.3393555, 1770.8516846
4: -300.4545288, 1007.3775024, -317.8994141, 1062.5919189, -1363.0462646, 1325.2768555

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6211378, upper bound: 3084.6226473
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6217806, upper bound: 3084.6227348
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1348.1613770, 2372.2373047, -1424.9781494, 2519.2534180, -3867.4145508, 3797.2153320
1: -419.3551636, 938.1928711, -445.3424683, 998.6915283, -1418.0466309, 1383.5351562
2: -264.5062866, 935.0141602, -279.6121216, 994.4382324, -1258.9445801, 1214.6262207
3: -559.4331665, 1115.8956299, -592.4503174, 1188.2412109, -1747.6741943, 1708.3459473
4: -291.4755249, 969.8793945, -308.6266479, 1029.1912842, -1320.6667480, 1278.5061035

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6189780, upper bound: 3084.6195581
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6196250, upper bound: 3084.6194127
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1340.9099121, 2360.9370117, -1434.1003418, 2538.1108398, -3879.0207520, 3795.0371094
1: -417.2884521, 933.5311279, -448.5162659, 1005.7407227, -1423.0291748, 1382.0473633
2: -263.2182312, 930.4975586, -281.5789185, 1001.7963867, -1265.0142822, 1212.0764160
3: -556.7454834, 1110.4521484, -596.8532715, 1196.9116211, -1753.6571045, 1707.3052979
4: -290.0259399, 965.3955688, -310.7531738, 1037.3730469, -1327.3989258, 1276.1486816

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6214311, upper bound: 3084.6222404
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6217787, upper bound: 3084.6220901
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1505.4884033, 2660.1623535, -1398.6323242, 2480.0043945, -3985.4921875, 4058.7944336
1: -472.2309265, 1056.6666260, -438.4456177, 983.9061890, -1456.1370850, 1495.1123047
2: -295.5944519, 1052.8326416, -274.7594299, 981.0062866, -1276.6007080, 1327.5920410
3: -627.5159912, 1255.6586914, -583.4970093, 1170.4082031, -1797.9240723, 1839.1557617
4: -326.3293152, 1089.9542236, -303.4528198, 1016.5487671, -1342.8780518, 1393.4069824

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6222887, upper bound: 3084.6228782
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3084.6223893, upper bound: 3084.6224424
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1505.4884033, 2660.1623535, -1490.0766602, 2636.8054199, -4142.2939453, 4150.2392578
1: -472.2309265, 1056.6666260, -467.4607849, 1049.0161133, -1521.2470703, 1524.1270752
2: -295.5944519, 1052.8326416, -293.1042175, 1044.7712402, -1340.3657227, 1345.9367676
3: -627.5159912, 1255.6586914, -621.4726562, 1247.3742676, -1874.8901367, 1877.1311035
4: -326.3293152, 1089.9542236, -323.4012146, 1081.6102295, -1407.9393311, 1413.3554688

Time for backsubstitution: 2.28 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.51 + 416.65 = 421.16 seconds
