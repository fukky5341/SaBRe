## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 2561.1064622435756


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1091.8393555, 1704.6987305, -1091.8393555, 1704.6987305, -2796.5380859, 2796.5380859)
1: (-844.7684937, 1571.1506348, -844.7684937, 1571.1506348, -2415.9191895, 2415.9191895)
2: (-735.8467407, 1621.3187256, -735.8467407, 1621.3187256, -2357.1655273, 2357.1655273)
3: (-1145.6840820, 1614.1608887, -1145.6840820, 1614.1608887, -2759.8449707, 2759.8449707)
4: (-904.0911865, 1719.4404297, -904.0911865, 1719.4404297, -2623.5314941, 2623.5314941)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 2.10 = 3.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2561.2089106, upper bound: 2561.2089106

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1925588, upper bound: 2561.1958448
time: 0.77 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1929244, upper bound: 2561.1929258
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -2561.1925588, upper bound: 2561.1958448
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -2561.1929244, upper bound: 2561.1929258

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1013.0983887, 1580.1373291, -1039.2489014, 1621.4108887, -2634.5087891, 2619.3862305
1: -783.4617920, 1456.0682373, -803.7858887, 1494.2131348, -2277.6748047, 2259.8540039
2: -682.3781128, 1503.1348877, -700.0997925, 1542.2817383, -2224.6594238, 2203.2346191
3: -1063.0198975, 1496.4930420, -1090.4299316, 1535.5026855, -2598.5224609, 2586.9228516
4: -838.3773804, 1594.5090332, -860.1717529, 1635.8898926, -2474.2673340, 2454.6806641

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1914796, upper bound: 2561.1898800
time: 0.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1925579, upper bound: 2561.1899245
time: 0.82 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1157.0312500, 1803.8189697, -1044.2753906, 1630.0021973, -2787.0334473, 2848.0942383
1: -895.8355713, 1665.1497803, -808.0269775, 1502.6341553, -2398.4697266, 2473.1767578
2: -780.8422852, 1716.3956299, -703.7556152, 1550.2906494, -2331.1328125, 2420.1513672
3: -1213.8613281, 1712.2392578, -1095.6400146, 1543.8585205, -2757.7197266, 2807.8793945
4: -959.7330933, 1820.7717285, -864.8302002, 1643.9642334, -2603.6972656, 2685.6013184

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1922870, upper bound: 2561.1911538
time: 0.79 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1911047, upper bound: 2561.1911061
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.22 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -2561.1914796, upper bound: 2561.1898800
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -2561.1925579, upper bound: 2561.1899245
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -2561.1922870, upper bound: 2561.1911538
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.22
Output dim: 0, lower bound: -2561.1911047, upper bound: 2561.1911061

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -978.4204712, 1527.6555176, -923.7575073, 1444.3864746, -2422.8068848, 2451.4130859
1: -757.5373535, 1407.3720703, -716.0883789, 1331.2619629, -2088.7993164, 2123.4604492
2: -659.8286743, 1453.0770264, -623.6978760, 1373.9039307, -2033.7326660, 2076.7741699
3: -1027.7298584, 1446.3842773, -970.8195801, 1367.7448730, -2395.4746094, 2417.2038574
4: -810.6600342, 1541.4945068, -766.2642822, 1457.0456543, -2267.7055664, 2307.7585449

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1889811, upper bound: 2561.1893202
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1914796, upper bound: 2561.1898800
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1914796, upper bound: 2561.1898800
time: 0.87 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1006.8182373, 1570.2542725, -1031.6381836, 1609.4306641, -2616.2487793, 2601.8918457
1: -778.5737915, 1446.9348145, -797.8744507, 1483.1542969, -2261.7280273, 2244.8093262
2: -678.1070557, 1493.7564697, -694.9273071, 1530.9268799, -2209.0332031, 2188.6833496
3: -1056.4110107, 1487.1347656, -1082.4324951, 1524.1674805, -2580.5783691, 2569.5666504
4: -833.1312866, 1584.5776367, -853.8099976, 1623.8688965, -2456.9997559, 2438.3876953

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1904768, upper bound: 2561.1894072
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1924745, upper bound: 2561.1899231
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1924745, upper bound: 2561.1899231
time: 0.79 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1145.5443115, 1785.5177002, -987.6728516, 1546.4301758, -2691.9743652, 2773.1899414
1: -886.8579712, 1648.4224854, -767.0957642, 1425.8654785, -2312.7233887, 2415.5183105
2: -773.0132446, 1699.0631104, -668.0527344, 1471.0953369, -2244.1081543, 2367.1157227
3: -1201.7686768, 1694.9760742, -1040.2103271, 1464.5305176, -2666.2993164, 2735.1865234
4: -950.1185303, 1802.4516602, -820.8394775, 1560.4514160, -2510.5698242, 2623.2910156

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1917613, upper bound: 2561.1908472
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1922855, upper bound: 2561.1911099
time: 0.84 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1111.8826904, 1737.0926514, -1061.7513428, 1654.4648438, -2766.3476562, 2798.8439941
1: -862.1172485, 1602.8111572, -820.6848755, 1525.0666504, -2387.1838379, 2423.4958496
2: -751.3426514, 1652.6844482, -714.3743896, 1574.3714600, -2325.7141113, 2367.0583496
3: -1168.1716309, 1647.3054199, -1113.5040283, 1566.9583740, -2735.1298828, 2760.8093262
4: -923.2442627, 1752.9420166, -877.5718384, 1669.9929199, -2593.2370605, 2630.5131836

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1906235, upper bound: 2561.1908001
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1910837, upper bound: 2561.1910837
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.17 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1914796, upper bound: 2561.1898800
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1914796, upper bound: 2561.1898800
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1924745, upper bound: 2561.1899231
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1924745, upper bound: 2561.1899231
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1917613, upper bound: 2561.1908472
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1922855, upper bound: 2561.1911099
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1906235, upper bound: 2561.1908001
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 0, lower bound: -2561.1910837, upper bound: 2561.1910837

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -894.4981689, 1398.4339600, -923.7575073, 1444.3864746, -2338.8847656, 2322.1909180
1: -693.4399414, 1288.7373047, -716.0883789, 1331.2619629, -2024.7019043, 2004.8256836
2: -603.9234009, 1330.2990723, -623.6978760, 1373.9039307, -1977.8272705, 1953.9968262
3: -940.3646851, 1324.1655273, -970.8195801, 1367.7448730, -2308.1096191, 2294.9851074
4: -741.9331665, 1410.9464111, -766.2642822, 1457.0456543, -2198.9785156, 2177.2106934

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894783, upper bound: 2561.1882717
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894433, upper bound: 2561.1886235
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1004.1997070, 1566.1195068, -923.7575073, 1444.3864746, -2448.5861816, 2489.8769531
1: -776.5411987, 1443.1130371, -716.0883789, 1331.2619629, -2107.8032227, 2159.2014160
2: -676.3283081, 1489.8371582, -623.6978760, 1373.9039307, -2050.2321777, 2113.5344238
3: -1053.6616211, 1483.2260742, -970.8195801, 1367.7448730, -2421.4064941, 2454.0456543
4: -830.9431763, 1580.4350586, -766.2642822, 1457.0456543, -2287.9887695, 2346.6992188

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894783, upper bound: 2561.1882717
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894418, upper bound: 2561.1886221
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -894.4981689, 1398.4339600, -1031.6381836, 1609.4306641, -2503.9287109, 2430.0715332
1: -693.4399414, 1288.7373047, -797.8744507, 1483.1542969, -2176.5942383, 2086.6113281
2: -603.9234009, 1330.2990723, -694.9273071, 1530.9268799, -2134.8493652, 2025.2263184
3: -940.3646851, 1324.1655273, -1082.4324951, 1524.1674805, -2464.5322266, 2406.5976562
4: -741.9331665, 1410.9464111, -853.8099976, 1623.8688965, -2365.8015137, 2264.7563477

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1898318, upper bound: 2561.1898550
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1898318, upper bound: 2561.1899245
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1004.1997070, 1566.1195068, -1031.6381836, 1609.4306641, -2613.6301270, 2597.7575684
1: -776.5411987, 1443.1130371, -797.8744507, 1483.1542969, -2259.6955566, 2240.9875488
2: -676.3283081, 1489.8371582, -694.9273071, 1530.9268799, -2207.2543945, 2184.7641602
3: -1053.6616211, 1483.2260742, -1082.4324951, 1524.1674805, -2577.8291016, 2565.6584473
4: -830.9431763, 1580.4350586, -853.8099976, 1623.8688965, -2454.8117676, 2434.2451172

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1898333, upper bound: 2561.1898489
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1898333, upper bound: 2561.1898800
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1040.9541016, 1621.5966797, -947.9879761, 1485.7703857, -2526.7246094, 2569.5847168
1: -805.5348511, 1497.4191895, -737.2028198, 1369.6607666, -2175.1955566, 2234.6220703
2: -702.3331299, 1543.0114746, -642.0382690, 1413.2987061, -2115.6318359, 2185.0498047
3: -1091.2789307, 1539.8852539, -999.6093140, 1406.7923584, -2498.0705566, 2539.4946289
4: -863.2918701, 1636.9182129, -788.8409424, 1499.3017578, -2362.5935059, 2425.7592773

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1889797
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1908421
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1121.6572266, 1750.7678223, -984.2741699, 1541.1357422, -2662.7929688, 2735.0419922
1: -869.7238159, 1616.1365967, -764.4819336, 1420.9842529, -2290.7080078, 2380.6186523
2: -758.0205078, 1665.9106445, -665.7603760, 1466.0677490, -2224.0881348, 2331.6708984
3: -1178.4450684, 1661.6157227, -1036.6892090, 1459.5010986, -2637.9462891, 2698.3049316
4: -931.6533813, 1767.3630371, -818.0151367, 1555.1267090, -2486.7800293, 2585.3781738

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893160, upper bound: 2561.1904782
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1911092
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1007.9951782, 1574.3934326, -1021.1679077, 1592.7851562, -2600.7802734, 2595.5610352
1: -781.3174438, 1452.8100586, -790.2053833, 1467.9050293, -2249.2221680, 2243.0153809
2: -681.1659546, 1497.7023926, -687.8261719, 1515.6156006, -2196.7810059, 2185.5280762
3: -1058.3326416, 1493.2441406, -1072.1464844, 1508.1730957, -2566.5058594, 2565.3901367
4: -837.0563965, 1588.4678955, -844.9099731, 1607.8056641, -2444.8620605, 2433.3779297

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1883393, upper bound: 2561.1853193
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1883168, upper bound: 2561.1853436
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1089.8995361, 1705.2595215, -1058.9443359, 1650.0021973, -2739.9016113, 2764.2038574
1: -846.4125366, 1573.1939697, -818.4849854, 1520.9293213, -2367.3417969, 2391.6787109
2: -737.5918579, 1622.3205566, -712.4451294, 1570.1265869, -2307.7180176, 2334.7651367
3: -1146.8228760, 1616.7169189, -1110.5263672, 1562.6995850, -2709.5224609, 2727.2431641
4: -906.2974854, 1720.8232422, -875.1895752, 1665.5025635, -2571.7995605, 2596.0126953

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884842, upper bound: 2561.1886907
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1887308, upper bound: 2561.1887308
time: 0.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.08 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1894783, upper bound: 2561.1882717
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1894433, upper bound: 2561.1886235
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1894783, upper bound: 2561.1882717
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1894418, upper bound: 2561.1886221
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1898318, upper bound: 2561.1898550
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1898318, upper bound: 2561.1899245
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1898333, upper bound: 2561.1898489
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1898333, upper bound: 2561.1898800
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1889797
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1908421
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1893160, upper bound: 2561.1904782
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1911092
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1883393, upper bound: 2561.1853193
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1883168, upper bound: 2561.1853436
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1884842, upper bound: 2561.1886907
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -2561.1887308, upper bound: 2561.1887308

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -848.5320435, 1327.4982910, -814.6217041, 1272.2756348, -2120.8076172, 2142.1201172
1: -658.0446777, 1223.4365234, -631.4286499, 1172.9543457, -1830.9990234, 1854.8652344
2: -573.0209351, 1262.7781982, -549.9509888, 1210.6616211, -1783.6822510, 1812.7291260
3: -892.1120605, 1256.8941650, -856.0576782, 1205.6270752, -2097.7390137, 2112.9514160
4: -704.0721436, 1339.0577393, -675.7687378, 1284.3123779, -1988.3845215, 2014.8262939

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1881003, upper bound: 2561.1921428
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1880708, upper bound: 2561.1903119
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -878.7708740, 1374.0717773, -895.1587524, 1399.8914795, -2278.6621094, 2269.2299805
1: -681.3923950, 1266.1662598, -694.0921021, 1290.0372314, -1971.4296875, 1960.2583008
2: -593.4102783, 1307.1085205, -604.5012817, 1331.5418701, -1924.9521484, 1911.6098633
3: -924.0927734, 1300.9371338, -941.1090698, 1325.3607178, -2249.4536133, 2242.0461426
4: -728.9783936, 1386.4133301, -742.6105347, 1412.2385254, -2141.2167969, 2129.0239258

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1880468, upper bound: 2561.1937275
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1880201, upper bound: 2561.1909705
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -958.2484131, 1494.3383789, -814.6217041, 1272.2756348, -2230.5239258, 2308.9599609
1: -740.8281860, 1377.1527100, -631.4286499, 1172.9543457, -1913.7824707, 2008.5812988
2: -645.1989746, 1421.6130371, -549.9509888, 1210.6616211, -1855.8603516, 1971.5639648
3: -1005.1506958, 1415.4056396, -856.0576782, 1205.6270752, -2210.7778320, 2271.4628906
4: -792.7559204, 1507.8997803, -675.7687378, 1284.3123779, -2077.0678711, 2183.6679688

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1882029, upper bound: 2561.1878594
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1882089, upper bound: 2561.1875689
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -984.4857788, 1535.7641602, -895.1587524, 1399.8914795, -2384.3771973, 2430.9226074
1: -761.5208130, 1415.0258789, -694.0921021, 1290.0372314, -2051.5581055, 2109.1179199
2: -663.2230225, 1460.9743652, -604.5012817, 1331.5418701, -1994.7648926, 2065.4755859
3: -1033.3729248, 1454.3122559, -941.1090698, 1325.3607178, -2358.7336426, 2395.4211426
4: -814.7877197, 1549.8989258, -742.6105347, 1412.2385254, -2227.0263672, 2292.5095215

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884006, upper bound: 2561.1883932
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1883721, upper bound: 2561.1875781
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -894.4981689, 1398.4339600, -1004.1997070, 1566.1195068, -2460.6176758, 2402.6335449
1: -693.4399414, 1288.7373047, -776.5411987, 1443.1130371, -2136.5529785, 2065.2780762
2: -603.9234009, 1330.2990723, -676.3283081, 1489.8371582, -2093.7602539, 2006.6274414
3: -940.3646851, 1324.1655273, -1053.6616211, 1483.2260742, -2423.5908203, 2377.8269043
4: -741.9331665, 1410.9464111, -830.9431763, 1580.4350586, -2322.3681641, 2241.8896484

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1896783, upper bound: 2561.1912897
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1898900, upper bound: 2561.1933500
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -894.4981689, 1398.4339600, -1133.5378418, 1769.7250977, -2664.2231445, 2531.9716797
1: -693.4399414, 1288.7373047, -879.0056763, 1633.4289551, -2326.8688965, 2167.7429199
2: -603.9234009, 1330.2990723, -766.1158447, 1683.8416748, -2287.7648926, 2096.4150391
3: -940.3646851, 1324.1655273, -1190.9626465, 1679.4595947, -2619.8237305, 2515.1281738
4: -741.9331665, 1410.9464111, -941.6044922, 1786.3165283, -2528.2492676, 2352.5507812

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1896783, upper bound: 2561.1913160
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1898885, upper bound: 2561.1933560
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1004.1997070, 1566.1195068, -1004.1997070, 1566.1195068, -2570.3193359, 2570.3193359
1: -776.5411987, 1443.1130371, -776.5411987, 1443.1130371, -2219.6542969, 2219.6542969
2: -676.3283081, 1489.8371582, -676.3283081, 1489.8371582, -2166.1655273, 2166.1655273
3: -1053.6616211, 1483.2260742, -1053.6616211, 1483.2260742, -2536.8874512, 2536.8874512
4: -830.9431763, 1580.4350586, -830.9431763, 1580.4350586, -2411.3781738, 2411.3781738

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601479, upper bound: 2561.1612538
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601412
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1004.1997070, 1566.1195068, -1133.5378418, 1769.7250977, -2773.9248047, 2699.6572266
1: -776.5411987, 1443.1130371, -879.0056763, 1633.4289551, -2409.9702148, 2322.1186523
2: -676.3283081, 1489.8371582, -766.1158447, 1683.8416748, -2360.1696777, 2255.9531250
3: -1053.6616211, 1483.2260742, -1190.9626465, 1679.4595947, -2733.1206055, 2674.1887207
4: -830.9431763, 1580.4350586, -941.6044922, 1786.3165283, -2617.2595215, 2522.0395508

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601479, upper bound: 2561.1745497
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1603831
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1040.9541016, 1621.5966797, -929.4432373, 1455.3427734, -2496.2968750, 2551.0400391
1: -805.5348511, 1497.4191895, -722.1805420, 1340.8956299, -2146.4301758, 2219.5996094
2: -702.3331299, 1543.0114746, -628.9479980, 1384.5209961, -2086.8540039, 2171.9594727
3: -1091.2789307, 1539.8852539, -979.9127808, 1377.7153320, -2468.9936523, 2519.7976074
4: -863.2918701, 1636.9182129, -772.6450195, 1469.3321533, -2332.6237793, 2409.5632324

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1892702, upper bound: 2561.1889811
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1892702, upper bound: 2561.1889811
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1040.9541016, 1621.5966797, -1101.9798584, 1717.4613037, -2758.4155273, 2723.5766602
1: -805.5348511, 1497.4191895, -853.0021362, 1585.4559326, -2390.9907227, 2350.4213867
2: -702.3331299, 1543.0114746, -743.5178833, 1634.1606445, -2336.4936523, 2286.5292969
3: -1091.2789307, 1539.8852539, -1156.1501465, 1630.2432861, -2721.5214844, 2696.0354004
4: -863.2918701, 1636.9182129, -914.0596924, 1733.6628418, -2596.9545898, 2550.9780273

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1908421
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1908421
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1121.6572266, 1750.7678223, -955.8089600, 1495.2904053, -2616.9477539, 2706.5766602
1: -869.7238159, 1616.1365967, -741.8988037, 1378.0211182, -2247.7448730, 2358.0354004
2: -758.0205078, 1665.9106445, -646.0741577, 1422.6616211, -2180.6818848, 2311.9848633
3: -1178.4450684, 1661.6157227, -1006.7634888, 1415.9141846, -2594.3591309, 2668.3789062
4: -931.6533813, 1767.3630371, -793.6965942, 1509.7103271, -2441.3632812, 2561.0595703

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1904782
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1904782
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1121.6572266, 1750.7678223, -1105.6322021, 1725.1694336, -2846.8266602, 2856.3999023
1: -869.7238159, 1616.1365967, -856.8534546, 1592.5567627, -2462.2805176, 2472.9899902
2: -758.0205078, 1665.9106445, -746.8388672, 1641.4439697, -2399.4643555, 2412.7490234
3: -1178.4450684, 1661.6157227, -1161.1483154, 1637.2781982, -2815.7229004, 2822.7634277
4: -931.6533813, 1767.3630371, -918.0913086, 1741.3260498, -2672.9787598, 2685.4543457

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1911077
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1911077
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -992.9307861, 1550.7280273, -989.3981323, 1542.5078125, -2535.4384766, 2540.1257324
1: -769.5979004, 1430.9705811, -765.4710693, 1421.3428955, -2190.9409180, 2196.4416504
2: -670.9459229, 1475.2093506, -666.2046509, 1467.8709717, -2138.8168945, 2141.4138184
3: -1042.5401611, 1470.8261719, -1038.9339600, 1460.4382324, -2502.9780273, 2509.7602539
4: -824.5123291, 1564.6608887, -818.3239136, 1557.3892822, -2381.9016113, 2382.9848633

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1840471, upper bound: 2561.1760107
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1860923, upper bound: 2561.1838917
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -974.7413940, 1522.3131104, -980.9848633, 1531.0187988, -2505.7602539, 2503.2976074
1: -755.3185425, 1404.5622559, -758.7159424, 1411.0264893, -2166.3449707, 2163.2783203
2: -658.4776001, 1447.9696045, -660.5104980, 1456.5229492, -2115.0002441, 2108.4799805
3: -1022.8614502, 1443.5164795, -1028.7946777, 1449.0749512, -2471.9365234, 2472.3110352
4: -809.2201538, 1535.4659424, -811.4834595, 1544.3918457, -2353.6120605, 2346.9487305

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1840866, upper bound: 2561.1760338
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1867829, upper bound: 2561.1839471
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1075.6632080, 1682.8492432, -1026.0256348, 1598.1320801, -2673.7954102, 2708.8740234
1: -835.2994995, 1552.4826660, -792.9296265, 1472.9177246, -2308.2172852, 2345.4123535
2: -727.9036255, 1601.0074463, -690.0905151, 1520.8669434, -2248.7700195, 2291.0979004
3: -1131.8464355, 1595.4748535, -1076.1571045, 1513.4464111, -2645.2929688, 2671.6318359
4: -894.4129028, 1698.2519531, -847.7145386, 1613.4436035, -2507.8557129, 2545.9663086

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884783, upper bound: 2561.1886917
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884773, upper bound: 2561.1886701
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1052.9321289, 1647.2911377, -1012.0733032, 1578.1823730, -2631.1142578, 2659.3645020
1: -817.6118774, 1519.5849609, -782.0404053, 1454.7324219, -2272.3442383, 2301.6254883
2: -712.4432983, 1567.0430908, -680.8456421, 1501.3906250, -2213.8339844, 2247.8886719
3: -1107.6450195, 1561.4593506, -1060.4498291, 1494.1226807, -2601.7673340, 2621.9091797
4: -875.4610596, 1662.0205078, -836.5147095, 1591.8884277, -2467.3488770, 2498.5349121

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1887258, upper bound: 2561.1887298
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1887258, upper bound: 2561.1887258
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.06 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1881003, upper bound: 2561.1921428
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1880708, upper bound: 2561.1903119
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1880468, upper bound: 2561.1937275
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1880201, upper bound: 2561.1909705
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1882029, upper bound: 2561.1878594
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1882089, upper bound: 2561.1875689
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1884006, upper bound: 2561.1883932
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1883721, upper bound: 2561.1875781
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1896783, upper bound: 2561.1912897
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1898900, upper bound: 2561.1933500
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1896783, upper bound: 2561.1913160
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1898885, upper bound: 2561.1933560
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1601479, upper bound: 2561.1612538
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601412
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1601479, upper bound: 2561.1745497
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1603831
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1892702, upper bound: 2561.1889811
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1892702, upper bound: 2561.1889811
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1908421
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1892687, upper bound: 2561.1908421
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1904782
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1904782
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1911077
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1893145, upper bound: 2561.1911077
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1840471, upper bound: 2561.1760107
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1860923, upper bound: 2561.1838917
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1840866, upper bound: 2561.1760338
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1867829, upper bound: 2561.1839471
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1884783, upper bound: 2561.1886917
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1884773, upper bound: 2561.1886701
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1887258, upper bound: 2561.1887298
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -2561.1887258, upper bound: 2561.1887258

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -793.6140137, 1243.0518799, -786.9316406, 1230.0692139, -2023.6832275, 2029.9835205
1: -615.9049683, 1145.4624023, -610.2899170, 1133.7927246, -1749.6977539, 1755.7523193
2: -536.2837524, 1182.5036621, -531.5203857, 1170.4167480, -1706.7004395, 1714.0240479
3: -834.8047485, 1176.2753906, -827.3508911, 1165.0848389, -1999.8895264, 2003.6262207
4: -658.7698975, 1253.6990967, -653.0420532, 1241.5249023, -1900.2947998, 1906.7412109

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1880763, upper bound: 2561.1921260
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1879543, upper bound: 2561.1917177
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -837.8941650, 1310.8898926, -804.8860474, 1256.9272461, -2094.8212891, 2115.7758789
1: -649.8330078, 1208.1105957, -623.8716431, 1158.7044678, -1808.5374756, 1831.9821777
2: -565.8596802, 1247.0177002, -543.3500977, 1196.0352783, -1761.8950195, 1790.3677979
3: -881.0083008, 1241.1702881, -845.9449463, 1191.0451660, -2072.0534668, 2087.1152344
4: -695.2955322, 1322.3697510, -667.6806030, 1268.8662109, -1964.1616211, 1990.0502930

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1880475, upper bound: 2561.1900377
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1879244, upper bound: 2561.1902699
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -822.9994507, 1288.1007080, -861.5941162, 1348.0270996, -2171.0266113, 2149.6948242
1: -638.5651855, 1186.7825928, -668.3927002, 1242.1646729, -1880.7297363, 1855.1751709
2: -556.0558472, 1225.4071045, -582.0722046, 1282.3319092, -1838.3876953, 1807.4788818
3: -865.8097534, 1218.9666748, -906.2877808, 1276.0067139, -2141.8164062, 2125.2543945
4: -682.9207153, 1299.5500488, -714.9594116, 1360.0604248, -2042.9812012, 2014.5093994

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871896, upper bound: 2561.1937275
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871910, upper bound: 2561.1937275
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -864.8134155, 1352.6260986, -886.2507935, 1386.1661377, -2250.9792480, 2238.8769531
1: -670.7463989, 1246.3753662, -687.2669678, 1277.3886719, -1948.1350098, 1933.6423340
2: -584.1116943, 1286.7117920, -598.5417480, 1318.5122070, -1902.6239014, 1885.2535400
3: -909.7012939, 1280.6313477, -931.9186401, 1312.3555908, -2222.0568848, 2212.5500488
4: -717.6157837, 1364.8006592, -735.3113403, 1398.4366455, -2116.0520020, 2100.1115723

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871603, upper bound: 2561.1909687
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871617, upper bound: 2561.1909705
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -922.2970581, 1438.0576172, -793.4987793, 1239.2000732, -2161.4970703, 2231.5563965
1: -713.1995239, 1324.8621826, -615.1879272, 1142.2683105, -1855.4677734, 1940.0500488
2: -620.9397583, 1368.0718994, -535.7113647, 1179.1738281, -1800.1135254, 1903.7829590
3: -967.9686890, 1361.5716553, -834.1707153, 1174.0355225, -2142.0041504, 2195.7424316
4: -762.9187012, 1451.3853760, -658.2642212, 1251.0429688, -2013.9616699, 2109.6496582

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1868327, upper bound: 2561.1870780
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1881660, upper bound: 2561.1857420
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1855531, upper bound: 2561.1852882
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -905.5288086, 1412.5815430, -789.4217529, 1232.7650146, -2138.2934570, 2202.0029297
1: -699.7106323, 1301.7523193, -611.6545410, 1136.4721680, -1836.1828613, 1913.4067383
2: -609.4241943, 1343.4572754, -532.6876831, 1172.9793701, -1782.4031982, 1876.1448975
3: -949.0551147, 1337.7303467, -829.1975098, 1168.0994873, -2117.1545410, 2166.9277344
4: -748.9844971, 1424.5627441, -654.6281738, 1244.1838379, -1993.1680908, 2079.1904297

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1868444, upper bound: 2561.1865931
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1880466, upper bound: 2561.1834469
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1881815, upper bound: 2561.1875440
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1881458, upper bound: 2561.1875626
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1882104, upper bound: 2561.1875689
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -953.6580811, 1487.6230469, -875.3457642, 1368.6254883, -2322.2834473, 2362.9687500
1: -737.9324951, 1370.2918701, -678.7377319, 1260.9852295, -1998.9177246, 2049.0292969
2: -642.5144043, 1415.1524658, -591.0437622, 1301.7626953, -1944.2770996, 2006.1962891
3: -1001.6232300, 1408.2766113, -920.4338989, 1295.5150146, -2297.1379395, 2328.7104492
4: -789.3620605, 1501.5465088, -726.0992432, 1380.7844238, -2170.1464844, 2227.6457520

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1873227, upper bound: 2561.1883824
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1873227, upper bound: 2561.1883932
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -930.2824097, 1451.3697510, -869.3905640, 1359.7872314, -2290.0695801, 2320.7602539
1: -719.1380005, 1337.1285400, -674.0536499, 1253.1182861, -1972.2563477, 2011.1821289
2: -626.3148804, 1380.3217773, -586.9448242, 1293.3366699, -1919.6510010, 1967.2666016
3: -975.6902466, 1373.9935303, -913.7591553, 1287.2507324, -2262.9409180, 2287.7526855
4: -769.5977173, 1463.9731445, -721.0656128, 1371.5336914, -2141.1311035, 2185.0385742

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1872513, upper bound: 2561.1875540
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1872528, upper bound: 2561.1875781
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -875.5201416, 1369.5260010, -989.3584595, 1542.9080811, -2418.4282227, 2358.8845215
1: -679.6177368, 1262.0584717, -764.9985962, 1421.7218018, -2101.3395996, 2027.0567627
2: -591.8907471, 1302.9324951, -666.2789307, 1467.7194824, -2059.6103516, 1969.2113037
3: -921.6289673, 1297.0440674, -1038.0251465, 1461.2673340, -2382.8962402, 2335.0693359
4: -727.1287842, 1382.2132568, -818.6164551, 1556.9243164, -2284.0532227, 2200.8295898

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894195, upper bound: 2561.1896349
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1889053, upper bound: 2561.1896311
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -884.0866089, 1382.0074463, -999.0402832, 1557.9995117, -2442.0861816, 2381.0478516
1: -685.2991333, 1273.6333008, -772.5260010, 1435.6439209, -2120.9428711, 2046.1590576
2: -596.8110352, 1314.6678467, -672.8173218, 1482.1037598, -2078.9147949, 1987.4851074
3: -929.3988647, 1308.5666504, -1048.2421875, 1475.5125732, -2404.9113770, 2356.8088379
4: -733.1756592, 1394.4033203, -826.6250000, 1572.2493896, -2305.4250488, 2221.0283203

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1853256, upper bound: 2561.1918715
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1900322, upper bound: 2561.1935883
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -875.5201416, 1369.5260010, -1119.0317383, 1747.3239746, -2622.8442383, 2488.5576172
1: -679.6177368, 1262.0584717, -867.8610229, 1612.7231445, -2292.3408203, 2129.9194336
2: -591.8907471, 1302.9324951, -756.4312744, 1662.4158936, -2254.3066406, 2059.3637695
3: -921.6289673, 1297.0440674, -1175.8602295, 1658.1542969, -2579.7832031, 2472.9042969
4: -727.1287842, 1382.2132568, -929.7142944, 1763.5704346, -2490.6992188, 2311.9274902

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1899006, upper bound: 2561.1895986
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1913050, upper bound: 2561.1896188
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -884.0866089, 1382.0074463, -1127.8984375, 1760.8249512, -2644.9116211, 2509.9057617
1: -685.2991333, 1273.6333008, -874.5998535, 1625.2600098, -2310.5590820, 2148.2331543
2: -596.8110352, 1314.6678467, -762.2598877, 1675.3984375, -2272.2094727, 2076.9277344
3: -929.3988647, 1308.5666504, -1185.0247803, 1671.0417480, -2600.4401855, 2493.5913086
4: -733.1756592, 1394.4033203, -936.8638306, 1777.3753662, -2510.5507812, 2331.2670898

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1918685, upper bound: 2561.1933492
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1869242, upper bound: 2561.1906572
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -976.1632080, 1521.8137207, -988.4547119, 1541.2719727, -2517.4350586, 2510.2685547
1: -754.7578735, 1402.0584717, -764.3248291, 1420.0871582, -2174.8447266, 2166.3833008
2: -657.2800903, 1447.7104492, -665.6465454, 1466.2034912, -2123.4836426, 2113.3566895
3: -1024.3293457, 1441.1303711, -1037.2066650, 1459.6072998, -2483.9365234, 2478.3369141
4: -807.5880127, 1535.9213867, -817.8492432, 1555.4570312, -2363.0449219, 2353.7705078

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601402
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601412
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -953.2823486, 1488.2750244, -968.0911865, 1509.7496338, -2463.0319824, 2456.3662109
1: -737.0429077, 1371.3309326, -748.3927002, 1391.1885986, -2128.2309570, 2119.7233887
2: -642.0244751, 1415.4780273, -651.7777710, 1436.1156006, -2078.1398926, 2067.2558594
3: -999.6423340, 1408.9439697, -1015.3685913, 1429.7657471, -2429.4082031, 2424.3125000
4: -788.9846191, 1500.8854980, -800.8502808, 1523.2218018, -2312.2062988, 2301.7353516

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601412, upper bound: 2561.1601402
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601402
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -976.1632080, 1521.8137207, -1119.8232422, 1748.1040039, -2724.2670898, 2641.6367188
1: -754.7578735, 1402.0584717, -868.3094482, 1613.4648438, -2368.2221680, 2270.3679199
2: -657.2800903, 1447.7104492, -756.7977295, 1663.3046875, -2320.5847168, 2204.5083008
3: -1024.3293457, 1441.1303711, -1176.5512695, 1658.9898682, -2683.3193359, 2617.6816406
4: -807.5880127, 1535.9213867, -930.1829834, 1764.5770264, -2572.1650391, 2466.1044922

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1803570, upper bound: 2561.1745180
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1808412, upper bound: 2561.1743203
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -953.2823486, 1488.2750244, -1096.1040039, 1711.4820557, -2664.7644043, 2584.3784180
1: -737.0429077, 1371.3309326, -849.9551392, 1579.6008301, -2316.6437988, 2221.2861328
2: -642.0244751, 1415.4780273, -740.7394409, 1628.2496338, -2270.2736816, 2156.2172852
3: -999.6423340, 1408.9439697, -1151.4296875, 1623.9351807, -2623.5769043, 2560.3735352
4: -788.9846191, 1500.8854980, -910.4813232, 1727.1926270, -2516.1772461, 2411.3664551

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1582723, upper bound: 2561.1595636
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1691468, upper bound: 2561.1598346
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1017.4556885, 1585.0762939, -929.4432373, 1455.3427734, -2472.7980957, 2514.5195312
1: -787.2538452, 1463.6813965, -722.1805420, 1340.8956299, -2128.1494141, 2185.8618164
2: -686.4097900, 1508.1789551, -628.9479980, 1384.5209961, -2070.9306641, 2137.1267090
3: -1066.6779785, 1505.0780029, -979.9127808, 1377.7153320, -2444.3933105, 2484.9907227
4: -843.8798218, 1599.9127197, -772.6450195, 1469.3321533, -2313.2119141, 2372.5576172

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893202, upper bound: 2561.1889811
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893202, upper bound: 2561.1889797
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1045.6462402, 1630.8074951, -929.4432373, 1455.3427734, -2500.9890137, 2560.2507324
1: -809.4635010, 1503.8724365, -722.1805420, 1340.8956299, -2150.3586426, 2226.0527344
2: -705.4038086, 1551.8363037, -628.9479980, 1384.5209961, -2089.9245605, 2180.7839355
3: -1097.2976074, 1545.3623047, -979.9127808, 1377.7153320, -2475.0129395, 2525.2746582
4: -866.2383423, 1646.3774414, -772.6450195, 1469.3321533, -2335.5705566, 2419.0224609

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893187, upper bound: 2561.1889811
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893202, upper bound: 2561.1889797
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1017.4556885, 1585.0762939, -1101.9798584, 1717.4613037, -2734.9169922, 2687.0561523
1: -787.2538452, 1463.6813965, -853.0021362, 1585.4559326, -2372.7097168, 2316.6835938
2: -686.4097900, 1508.1789551, -743.5178833, 1634.1606445, -2320.5703125, 2251.6967773
3: -1066.6779785, 1505.0780029, -1156.1501465, 1630.2432861, -2696.9211426, 2661.2280273
4: -843.8798218, 1599.9127197, -914.0596924, 1733.6628418, -2577.5427246, 2513.9721680

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909287, upper bound: 2561.1907770
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909287, upper bound: 2561.1908406
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1045.6462402, 1630.8074951, -1101.9798584, 1717.4613037, -2763.1074219, 2732.7873535
1: -809.4635010, 1503.8724365, -853.0021362, 1585.4559326, -2394.9191895, 2356.8742676
2: -705.4038086, 1551.8363037, -743.5178833, 1634.1606445, -2339.5644531, 2295.3537598
3: -1097.2976074, 1545.3623047, -1156.1501465, 1630.2432861, -2727.5410156, 2701.5124512
4: -866.2383423, 1646.3774414, -914.0596924, 1733.6628418, -2599.9011230, 2560.4370117

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909302, upper bound: 2561.1907756
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909302, upper bound: 2561.1908421
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1098.0722656, 1714.1361084, -955.8089600, 1495.2904053, -2593.3627930, 2669.9450684
1: -851.4404907, 1582.3031006, -741.8988037, 1378.0211182, -2229.4616699, 2324.2019043
2: -742.0964355, 1630.9265137, -646.0741577, 1422.6616211, -2164.7580566, 2277.0007324
3: -1153.8172607, 1626.6868896, -1006.7634888, 1415.9141846, -2569.7314453, 2633.4499512
4: -912.2368164, 1730.2082520, -793.6965942, 1509.7103271, -2421.9472656, 2523.9047852

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894057, upper bound: 2561.1904782
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894072, upper bound: 2561.1904782
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1133.3123779, 1770.6425781, -955.8089600, 1495.2904053, -2628.6027832, 2726.4516602
1: -878.9195557, 1632.2900391, -741.8988037, 1378.0211182, -2256.9406738, 2374.1889648
2: -765.6055298, 1684.9393311, -646.0741577, 1422.6616211, -2188.2670898, 2331.0134277
3: -1191.8031006, 1677.1815186, -1006.7634888, 1415.9141846, -2607.7172852, 2683.9440918
4: -940.0958862, 1787.7607422, -793.6965942, 1509.7103271, -2449.8061523, 2581.4570312

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894057, upper bound: 2561.1904782
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1894072, upper bound: 2561.1904767
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1098.0722656, 1714.1361084, -1105.6322021, 1725.1694336, -2823.2416992, 2819.7683105
1: -851.4404907, 1582.3031006, -856.8534546, 1592.5567627, -2443.9973145, 2439.1564941
2: -742.0964355, 1630.9265137, -746.8388672, 1641.4439697, -2383.5402832, 2377.7648926
3: -1153.8172607, 1626.6868896, -1161.1483154, 1637.2781982, -2791.0954590, 2787.8347168
4: -912.2368164, 1730.2082520, -918.0913086, 1741.3260498, -2653.5627441, 2648.2995605

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909331, upper bound: 2561.1910170
time: 1.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909316, upper bound: 2561.1910937
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1133.3123779, 1770.6425781, -1105.6322021, 1725.1694336, -2858.4819336, 2876.2749023
1: -878.9195557, 1632.2900391, -856.8534546, 1592.5567627, -2471.4763184, 2489.1435547
2: -765.6055298, 1684.9393311, -746.8388672, 1641.4439697, -2407.0495605, 2431.7778320
3: -1191.8031006, 1677.1815186, -1161.1483154, 1637.2781982, -2829.0812988, 2838.3288574
4: -940.0958862, 1787.7607422, -918.0913086, 1741.3260498, -2681.4216309, 2705.8520508

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909316, upper bound: 2561.1910185
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1909316, upper bound: 2561.1910937
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -975.0402832, 1521.9683838, -957.2369995, 1493.5552979, -2468.5957031, 2479.2050781
1: -755.4662476, 1404.4979248, -741.2057495, 1376.0126953, -2131.4790039, 2145.7033691
2: -658.6273804, 1447.9871826, -645.0354004, 1421.3714600, -2079.9987793, 2093.0222168
3: -1023.3255615, 1443.8897705, -1006.0847168, 1413.7005615, -2437.0261230, 2449.9746094
4: -809.3129272, 1535.8435059, -792.2606812, 1508.1044922, -2317.4169922, 2328.1037598

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1840304, upper bound: 2561.1760058
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1840304, upper bound: 2561.1760107
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -954.8056641, 1492.7701416, -978.5911865, 1525.5345459, -2480.3398438, 2471.3613281
1: -740.8449707, 1377.2956543, -757.0413818, 1405.6789551, -2146.5236816, 2134.3366699
2: -645.8282471, 1420.0772705, -658.8401489, 1451.7072754, -2097.5351562, 2078.9174805
3: -1003.7402344, 1415.4346924, -1027.5679932, 1444.3311768, -2448.0710449, 2443.0026855
4: -793.5856934, 1506.3010254, -809.2977295, 1540.2727051, -2333.8583984, 2315.5986328

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1853383, upper bound: 2561.1838837
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1853383, upper bound: 2561.1838917
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -956.7625732, 1493.3811035, -949.1369019, 1482.6191406, -2439.3813477, 2442.5180664
1: -741.0418091, 1377.9467773, -734.7777710, 1366.2341309, -2107.2758789, 2112.7246094
2: -646.0343018, 1420.5776367, -639.5552979, 1410.5748291, -2056.6091309, 2060.1328125
3: -1003.4223022, 1416.3874512, -996.4555054, 1402.8227539, -2406.2451172, 2412.8427734
4: -793.8917236, 1506.4350586, -785.6488647, 1495.7600098, -2289.6518555, 2292.0834961

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1835138, upper bound: 2561.1760061
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1835127, upper bound: 2561.1760338
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -937.8209839, 1466.3665771, -970.0645142, 1513.8831787, -2451.7038574, 2436.4311523
1: -727.5593872, 1352.7562256, -750.2008667, 1395.1770020, -2122.7363281, 2102.9567871
2: -634.2183228, 1394.7437744, -653.1117554, 1440.1693115, -2074.3876953, 2047.8553467
3: -985.3920898, 1390.0419922, -1017.2868652, 1432.8267822, -2418.2187500, 2407.3288574
4: -779.3424683, 1479.1406250, -802.4270630, 1527.0572510, -2306.3996582, 2281.5671387

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1838820, upper bound: 2561.1838832
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1838820, upper bound: 2561.1838832
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1085.1573486, 1693.9377441, -1026.0256348, 1598.1320801, -2683.2895508, 2719.9631348
1: -841.3966675, 1563.6069336, -792.9296265, 1472.9177246, -2314.3144531, 2356.5366211
2: -733.3482666, 1611.6884766, -690.0905151, 1520.8669434, -2254.2143555, 2301.7785645
3: -1140.2657471, 1607.5142822, -1076.1571045, 1513.4464111, -2653.7121582, 2683.6713867
4: -901.5358887, 1709.8212891, -847.7145386, 1613.4436035, -2514.9787598, 2557.5358887

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884014, upper bound: 2561.1886734
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1865561, upper bound: 2561.1886089
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1119.5059814, 1749.0172119, -1026.0256348, 1598.1320801, -2717.6381836, 2775.0429688
1: -868.1608887, 1612.2731934, -792.9296265, 1472.9177246, -2341.0783691, 2405.2023926
2: -756.2301636, 1664.3443604, -690.0905151, 1520.8669434, -2277.0966797, 2354.4345703
3: -1177.2941895, 1656.6550293, -1076.1571045, 1513.4464111, -2690.7407227, 2732.8120117
4: -928.5960693, 1765.9332275, -847.7145386, 1613.4436035, -2542.0385742, 2613.6474609

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884014, upper bound: 2561.1886471
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1865561, upper bound: 2561.1886079
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1059.9357910, 1654.4644775, -1012.0733032, 1578.1823730, -2638.1171875, 2666.5378418
1: -821.7730713, 1527.2720947, -782.0404053, 1454.7324219, -2276.5053711, 2309.3125000
2: -716.1726074, 1574.0754395, -680.8456421, 1501.3906250, -2217.5632324, 2254.9206543
3: -1113.4594727, 1569.9078369, -1060.4498291, 1494.1226807, -2607.5820312, 2630.3576660
4: -880.4017944, 1669.7854004, -836.5147095, 1591.8884277, -2472.2902832, 2506.3000488

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1885953, upper bound: 2561.1886927
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1886705, upper bound: 2561.1886715
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1095.4847412, 1711.0939941, -1012.0733032, 1578.1823730, -2673.6665039, 2723.1672363
1: -849.3831787, 1577.3101807, -782.0404053, 1454.7324219, -2304.1157227, 2359.3505859
2: -739.8094482, 1628.1781006, -680.8456421, 1501.3906250, -2241.2001953, 2309.0231934
3: -1151.7282715, 1620.4654541, -1060.4498291, 1494.1226807, -2645.8510742, 2680.9152832
4: -908.4486084, 1727.4857178, -836.5147095, 1591.8884277, -2500.3361816, 2564.0004883

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1885953, upper bound: 2561.1886760
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1886705, upper bound: 2561.1886705
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.10 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1880763, upper bound: 2561.1921260
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1879543, upper bound: 2561.1917177
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1880475, upper bound: 2561.1900377
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1879244, upper bound: 2561.1902699
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1871896, upper bound: 2561.1937275
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1871910, upper bound: 2561.1937275
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1871603, upper bound: 2561.1909687
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1871617, upper bound: 2561.1909705
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1881660, upper bound: 2561.1857420
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1855531, upper bound: 2561.1852882
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1881458, upper bound: 2561.1875626
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1882104, upper bound: 2561.1875689
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1873227, upper bound: 2561.1883824
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1873227, upper bound: 2561.1883932
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1872513, upper bound: 2561.1875540
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1872528, upper bound: 2561.1875781
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1894195, upper bound: 2561.1896349
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1889053, upper bound: 2561.1896311
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1853256, upper bound: 2561.1918715
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1900322, upper bound: 2561.1935883
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1899006, upper bound: 2561.1895986
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1913050, upper bound: 2561.1896188
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1918685, upper bound: 2561.1933492
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1869242, upper bound: 2561.1906572
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601402
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601412
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1601412, upper bound: 2561.1601402
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1601402, upper bound: 2561.1601402
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1803570, upper bound: 2561.1745180
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1808412, upper bound: 2561.1743203
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1582723, upper bound: 2561.1595636
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1691468, upper bound: 2561.1598346
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1893202, upper bound: 2561.1889811
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1893202, upper bound: 2561.1889797
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1893187, upper bound: 2561.1889811
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1893202, upper bound: 2561.1889797
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909287, upper bound: 2561.1907770
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909287, upper bound: 2561.1908406
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909302, upper bound: 2561.1907756
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909302, upper bound: 2561.1908421
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1894057, upper bound: 2561.1904782
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1894072, upper bound: 2561.1904782
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1894057, upper bound: 2561.1904782
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1894072, upper bound: 2561.1904767
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909331, upper bound: 2561.1910170
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909316, upper bound: 2561.1910937
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909316, upper bound: 2561.1910185
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1909316, upper bound: 2561.1910937
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1840304, upper bound: 2561.1760058
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1840304, upper bound: 2561.1760107
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1853383, upper bound: 2561.1838837
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1853383, upper bound: 2561.1838917
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1835138, upper bound: 2561.1760061
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1835127, upper bound: 2561.1760338
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1838820, upper bound: 2561.1838832
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1838820, upper bound: 2561.1838832
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1884014, upper bound: 2561.1886734
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1865561, upper bound: 2561.1886089
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1884014, upper bound: 2561.1886471
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1865561, upper bound: 2561.1886079
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1885953, upper bound: 2561.1886927
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1886705, upper bound: 2561.1886715
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1885953, upper bound: 2561.1886760
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -2561.1886705, upper bound: 2561.1886705

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -778.7077637, 1219.9942627, -766.3720093, 1198.3653564, -1977.0731201, 1986.3662109
1: -604.4338989, 1124.2310791, -595.2329712, 1104.6389160, -1709.0727539, 1719.4639893
2: -526.2888184, 1160.4781494, -518.3723755, 1140.5113525, -1666.8001709, 1678.8504639
3: -819.1279907, 1154.4558105, -807.0855713, 1135.4263916, -1954.5543213, 1961.5413818
4: -646.6029663, 1230.2416992, -636.7488403, 1210.2380371, -1856.8409424, 1866.9904785

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1863909, upper bound: 2561.1880005
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1863998, upper bound: 2561.1865551
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -788.9088135, 1235.6342773, -776.9471436, 1214.4188232, -2003.3276367, 2012.5814209
1: -612.2158813, 1138.6396484, -602.5399780, 1119.3852539, -1731.6010742, 1741.1796875
2: -533.0640869, 1175.4414062, -524.7375488, 1155.5000000, -1688.5640869, 1700.1789551
3: -829.8481445, 1169.2250977, -816.8955688, 1150.1916504, -1980.0396729, 1986.1206055
4: -654.7972412, 1246.2248535, -644.6833496, 1225.7542725, -1880.5515137, 1890.9082031

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1848294, upper bound: 2561.1909048
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1859963, upper bound: 2561.1877298
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1859714, upper bound: 2561.1863640
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -822.9893799, 1287.7930908, -787.1723022, 1229.7783203, -2052.7673340, 2074.9653320
1: -638.3457642, 1186.8385010, -610.9445801, 1133.6546631, -1772.0002441, 1797.7830811
2: -555.8436279, 1224.9672852, -532.0792236, 1170.3201904, -1726.1638184, 1757.0465088
3: -865.3250122, 1219.3221436, -828.5171509, 1165.5576172, -2030.8825684, 2047.8393555
4: -683.0860596, 1298.8902588, -653.7359619, 1241.9310303, -1925.0168457, 1952.6262207

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1862816, upper bound: 2561.1859288
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1863748, upper bound: 2561.1859780
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -833.2814331, 1303.6208496, -794.7288208, 1240.9852295, -2074.2661133, 2098.3496094
1: -646.2193604, 1201.4224854, -615.9840088, 1144.0372314, -1790.2564697, 1817.4062500
2: -562.7064819, 1240.0947266, -536.4460449, 1180.8535156, -1743.5600586, 1776.5407715
3: -876.1498413, 1234.2572021, -835.3126831, 1175.8868408, -2052.0366211, 2069.5698242
4: -691.4088745, 1315.0422363, -659.1748047, 1252.8179932, -1944.2268066, 1974.2169189

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1859662, upper bound: 2561.1863543
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1859684, upper bound: 2561.1860669
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -822.9994507, 1288.1007080, -833.0891113, 1303.4150391, -2126.4145508, 2121.1896973
1: -638.5651855, 1186.7825928, -646.4056396, 1200.8569336, -1839.4219971, 1833.1882324
2: -556.0558472, 1225.4071045, -562.8672485, 1240.0125732, -1796.0683594, 1788.2744141
3: -865.8097534, 1218.9666748, -876.7321167, 1233.6342773, -2099.4436035, 2095.6984863
4: -682.9207153, 1299.5500488, -691.3189697, 1315.3205566, -1998.2412109, 1990.8684082

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1866165, upper bound: 2561.1937041
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871896, upper bound: 2561.1937261
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -822.9994507, 1288.1007080, -1003.4908447, 1562.6499023, -2385.6494141, 2291.5915527
1: -638.5651855, 1186.7825928, -776.4390869, 1443.0109863, -2081.5761719, 1963.2216797
2: -556.0558472, 1225.4071045, -676.9482422, 1487.1148682, -2043.1706543, 1902.3549805
3: -865.8097534, 1218.9666748, -1051.9945068, 1484.3157959, -2350.1254883, 2270.9611816
4: -682.9207153, 1299.5500488, -832.1176758, 1577.7927246, -2260.7133789, 2131.6672363

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1866165, upper bound: 2561.1937041
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871896, upper bound: 2561.1937275
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -864.8134155, 1352.6260986, -857.1425171, 1340.5996094, -2205.4128418, 2209.7685547
1: -670.7463989, 1246.3753662, -664.8116455, 1235.2135010, -1905.9598389, 1911.1870117
2: -584.1116943, 1286.7117920, -578.9421997, 1275.2886963, -1859.4003906, 1865.6540527
3: -909.7012939, 1280.6313477, -901.7282715, 1269.1293945, -2178.8305664, 2182.3596191
4: -717.6157837, 1364.8006592, -711.1937866, 1352.7412109, -2070.3569336, 2075.9943848

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1869117, upper bound: 2561.1865643
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871501, upper bound: 2561.1909287
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -864.8134155, 1352.6260986, -994.9884644, 1550.5953369, -2415.4086914, 2347.6145020
1: -670.7463989, 1246.3753662, -770.3479004, 1431.6972656, -2102.4436035, 2016.7232666
2: -584.1116943, 1286.7117920, -671.6307983, 1475.5472412, -2059.6589355, 1958.3422852
3: -909.7012939, 1280.6313477, -1043.7287598, 1472.4738770, -2382.1750488, 2324.3601074
4: -717.6157837, 1364.8006592, -825.5682373, 1565.5111084, -2283.1269531, 2190.3688965

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1869117, upper bound: 2561.1866033
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1871501, upper bound: 2561.1909307
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -922.2970581, 1438.0576172, -791.0933228, 1235.3785400, -2157.6755371, 2229.1508789
1: -713.1995239, 1324.8621826, -613.3131104, 1138.7753906, -1851.9748535, 1938.1752930
2: -620.9397583, 1368.0718994, -534.0698853, 1175.5683594, -1796.5079346, 1902.1416016
3: -967.9686890, 1361.5716553, -831.6323242, 1170.4610596, -2138.4296875, 2193.2041016
4: -762.9187012, 1451.3853760, -656.2417603, 1247.2197266, -2010.1384277, 2107.6271973

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1855546, upper bound: 2561.1852897
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1855546, upper bound: 2561.1852897
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -922.2970581, 1438.0576172, -794.2044678, 1240.7830811, -2163.0800781, 2232.2617188
1: -713.1995239, 1324.8621826, -615.7592163, 1143.7160645, -1856.9155273, 1940.6210938
2: -620.9397583, 1368.0718994, -536.2487183, 1180.6359863, -1801.5756836, 1904.3204346
3: -967.9686890, 1361.5716553, -834.8885498, 1175.3841553, -2143.3527832, 2196.4602051
4: -762.9187012, 1451.3853760, -658.9702148, 1252.4627686, -2015.3814697, 2110.3554688

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1855546, upper bound: 2561.1852897
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1855546, upper bound: 2561.1852897
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -904.0618286, 1410.3288574, -778.5257568, 1216.2800293, -2120.3415527, 2188.8544922
1: -698.5768433, 1299.6732178, -603.3046875, 1121.2443848, -1819.8211670, 1902.9775391
2: -608.4352417, 1341.3100586, -525.4249268, 1157.2647705, -1765.6998291, 1866.7349854
3: -947.5156250, 1335.5794678, -817.8933105, 1152.3164062, -2099.8320312, 2153.4726562
4: -747.7700806, 1422.2730713, -645.7112427, 1227.4232178, -1975.1932373, 2067.9843750

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -905.5288086, 1412.5815430, -784.7340088, 1225.4952393, -2131.0239258, 2197.3154297
1: -699.7106323, 1301.7523193, -608.0712891, 1129.7701416, -1829.4807129, 1909.8236084
2: -609.4241943, 1343.4572754, -529.5659180, 1166.0725098, -1775.4963379, 1873.0231934
3: -949.0551147, 1337.7303467, -824.3584595, 1161.1805420, -2110.2355957, 2162.0888672
4: -748.9844971, 1424.5627441, -650.7810669, 1236.8778076, -1985.8621826, 2075.3432617

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -953.6580811, 1487.6230469, -847.6884155, 1325.3602295, -2279.0183105, 2335.3115234
1: -737.9324951, 1370.2918701, -657.4199829, 1220.9418945, -1958.8743896, 2027.7116699
2: -642.5144043, 1415.1524658, -572.4392090, 1260.7333984, -1903.2478027, 1987.5914307
3: -1001.6232300, 1408.2766113, -891.7814331, 1254.4429932, -2256.0661621, 2300.0573730
4: -789.3620605, 1501.5465088, -703.1925049, 1337.4093018, -2126.7709961, 2204.7390137

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1858416, upper bound: 2561.1869532
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1858288, upper bound: 2561.1869731
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -953.6580811, 1487.6230469, -1002.7229614, 1561.4544678, -2515.1125488, 2490.3459473
1: -737.9324951, 1370.2918701, -775.8852539, 1441.8416748, -2179.7741699, 2146.1772461
2: -642.5144043, 1415.1524658, -676.4671631, 1485.9366455, -2128.4511719, 2091.6196289
3: -1001.6232300, 1408.2766113, -1051.2578125, 1483.1335449, -2484.7565918, 2459.5344238
4: -789.3620605, 1501.5465088, -831.5626221, 1576.5869141, -2365.9487305, 2333.1091309

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1858416, upper bound: 2561.1869560
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1858288, upper bound: 2561.1869793
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -930.2824097, 1451.3697510, -839.1701660, 1312.5148926, -2242.7971191, 2290.5400391
1: -719.1380005, 1337.1285400, -650.6942749, 1209.3586426, -1928.4965820, 1987.8226318
2: -626.3148804, 1380.3217773, -566.5584106, 1248.4772949, -1874.7917480, 1946.8801270
3: -975.6902466, 1373.9935303, -882.3711548, 1242.3793945, -2218.0695801, 2256.3642578
4: -769.5977173, 1463.9731445, -695.9736328, 1324.0905762, -2093.6882324, 2159.9467773

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1857215, upper bound: 2561.1808413
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1857652, upper bound: 2561.1862579
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -930.2824097, 1451.3697510, -981.5560913, 1529.2658691, -2459.5483398, 2432.9257812
1: -719.1380005, 1337.1285400, -759.7170410, 1411.8038330, -2130.9409180, 2096.8457031
2: -626.3148804, 1380.3217773, -662.2661743, 1455.0257568, -2081.3400879, 2042.5878906
3: -975.6902466, 1373.9935303, -1029.1752930, 1451.9176025, -2427.6076660, 2403.1687012
4: -769.5977173, 1463.9731445, -814.0558472, 1543.6684570, -2313.2661133, 2278.0288086

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1857215, upper bound: 2561.1808449
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1857652, upper bound: 2561.1862781
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -854.3876953, 1336.2618408, -956.6550903, 1491.8840332, -2346.2717285, 2292.9167480
1: -663.3094482, 1231.1848145, -739.9660034, 1374.3365479, -2037.6458740, 1971.1507568
2: -577.6033936, 1271.2659912, -644.3016968, 1419.1469727, -1996.7502441, 1915.5676270
3: -899.6892090, 1265.3083496, -1004.2934570, 1412.4963379, -2312.1850586, 2269.6013184
4: -709.6018677, 1348.7854004, -791.6270142, 1505.6437988, -2215.2456055, 2140.4123535

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1874205, upper bound: 2561.1895968
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1889067, upper bound: 2561.1896311
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1889067, upper bound: 2561.1896311
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -850.1747437, 1330.1697998, -937.5272217, 1462.4716797, -2312.6464844, 2267.6970215
1: -659.8184204, 1225.7363281, -724.5243530, 1347.5041504, -2007.3223877, 1950.2606201
2: -574.5490723, 1265.3925781, -631.0734863, 1390.8096924, -1965.3587646, 1896.4660645
3: -894.6235962, 1259.5495605, -982.8662109, 1384.7177734, -2279.3410645, 2242.4157715
4: -705.8524780, 1342.1700439, -775.5258179, 1474.9306641, -2180.7832031, 2117.6958008

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1867867, upper bound: 2561.1879486
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1889053, upper bound: 2561.1896311
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1889067, upper bound: 2561.1896297
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -834.8958740, 1306.5069580, -930.8140869, 1454.6306152, -2289.5263672, 2237.3210449
1: -648.2257080, 1203.9090576, -721.5299072, 1340.2999268, -1988.5255127, 1925.4389648
2: -564.5039673, 1242.9708252, -628.4691772, 1383.7258301, -1948.2297363, 1871.4399414
3: -879.2385254, 1236.8714600, -978.6797485, 1377.6104736, -2256.8486328, 2215.5512695
4: -693.3407593, 1318.7237549, -771.9751587, 1468.0916748, -2161.4323730, 2090.6989746

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1840206, upper bound: 2561.1914998
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1840039, upper bound: 2561.1904884
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -879.5588379, 1375.0814209, -988.8067017, 1542.1716309, -2421.7304688, 2363.8881836
1: -681.7717285, 1267.2449951, -764.5329590, 1421.0440674, -2102.8156738, 2031.7779541
2: -593.6818237, 1308.1101074, -665.7849731, 1467.0812988, -2060.7631836, 1973.8950195
3: -924.6383057, 1301.8835449, -1037.4969482, 1460.3605957, -2384.9990234, 2339.3803711
4: -729.2858887, 1387.3944092, -817.9240112, 1556.2517090, -2285.5375977, 2205.3183594

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1891897, upper bound: 2561.1930889
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1889969, upper bound: 2561.1896788
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -854.3876953, 1336.2618408, -1093.7562256, 1707.3549805, -2561.7426758, 2430.0180664
1: -663.3094482, 1231.1848145, -848.1606445, 1575.8684082, -2239.1777344, 2079.3454590
2: -577.6033936, 1271.2659912, -739.2467651, 1624.4769287, -2202.0800781, 2010.5126953
3: -899.6892090, 1265.3083496, -1149.2863770, 1620.3581543, -2520.0466309, 2414.5947266
4: -709.6018677, 1348.7854004, -908.6389160, 1723.4281006, -2433.0297852, 2257.4243164

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1846245, upper bound: 2561.1892194
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1878323, upper bound: 2561.1895643
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1899021, upper bound: 2561.1895986
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1899021, upper bound: 2561.1895986
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -850.1747437, 1330.1697998, -1054.4378662, 1647.2143555, -2497.3889160, 2384.6071777
1: -659.8184204, 1225.7363281, -818.3558350, 1520.1064453, -2179.9248047, 2044.0921631
2: -574.5490723, 1265.3925781, -712.9569702, 1567.0093994, -2141.5585938, 1978.3496094
3: -894.6235962, 1259.5495605, -1108.7319336, 1562.3048096, -2456.9284668, 2368.2814941
4: -705.8524780, 1342.1700439, -876.2176514, 1662.4686279, -2368.3210449, 2218.3876953

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1895976, upper bound: 2561.1892761
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1912934, upper bound: 2561.1896203
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1912934, upper bound: 2561.1896203
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -859.2481689, 1343.2362061, -1073.2609863, 1675.1875000, -2534.4355469, 2416.4970703
1: -665.9456177, 1237.9163818, -831.6425781, 1545.7889404, -2211.7346191, 2069.5590820
2: -579.9724731, 1277.6491699, -725.0896606, 1593.2270508, -2173.1992188, 2002.7387695
3: -903.0352173, 1271.9118652, -1126.6848145, 1589.7066650, -2492.7419434, 2398.5964355
4: -712.6232300, 1354.9720459, -891.5709229, 1689.8856201, -2402.5087891, 2246.5424805

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1910742, upper bound: 2561.1931028
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1901422, upper bound: 2561.1920647
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1892169, upper bound: 2561.1908962
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1893178, upper bound: 2561.1878515
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -873.2086792, 1364.6947021, -1104.9804688, 1724.4450684, -2597.6538086, 2469.6745605
1: -676.7703857, 1257.6350098, -856.6831665, 1591.7458496, -2268.5161133, 2114.3181152
2: -589.3847656, 1298.1903076, -746.6190796, 1640.8636475, -2230.2485352, 2044.8092041
3: -917.8849487, 1292.2003174, -1160.8822021, 1636.6738281, -2554.5585938, 2453.0825195
4: -724.0686035, 1376.9770508, -917.6831055, 1740.8486328, -2464.9172363, 2294.6601562

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1862085, upper bound: 2561.1903913
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1855879, upper bound: 2561.1896587
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1858987, upper bound: 2561.1889873
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1868948, upper bound: 2561.1906386
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -976.1632080, 1521.8137207, -976.1632080, 1521.8137207, -2497.9768066, 2497.9768066
1: -754.7578735, 1402.0584717, -754.7578735, 1402.0584717, -2156.8164062, 2156.8164062
2: -657.2800903, 1447.7104492, -657.2800903, 1447.7104492, -2104.9904785, 2104.9904785
3: -1024.3293457, 1441.1303711, -1024.3293457, 1441.1303711, -2465.4597168, 2465.4597168
4: -807.5880127, 1535.9213867, -807.5880127, 1535.9213867, -2343.5092773, 2343.5092773

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1416055, upper bound: 2561.1453085
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1414370, upper bound: 2561.1414906
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -976.1632080, 1521.8137207, -953.2823486, 1488.2750244, -2464.4382324, 2475.0961914
1: -754.7578735, 1402.0584717, -737.0429077, 1371.3309326, -2126.0883789, 2139.1013184
2: -657.2800903, 1447.7104492, -642.0244751, 1415.4780273, -2072.7580566, 2089.7346191
3: -1024.3293457, 1441.1303711, -999.6423340, 1408.9439697, -2433.2734375, 2440.7727051
4: -807.5880127, 1535.9213867, -788.9846191, 1500.8854980, -2308.4731445, 2324.9060059

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1416054, upper bound: 2561.1453075
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1414370, upper bound: 2561.1414904
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -953.2823486, 1488.2750244, -976.0103760, 1521.5788574, -2474.8613281, 2464.2854004
1: -737.0429077, 1371.3309326, -754.6422119, 1401.8427734, -2138.8857422, 2125.9729004
2: -642.0244751, 1415.4780273, -657.1799927, 1447.4862061, -2089.5107422, 2072.6579590
3: -999.6423340, 1408.9439697, -1024.1715088, 1440.9102783, -2440.5527344, 2433.1152344
4: -788.9846191, 1500.8854980, -807.4657593, 1535.6831055, -2324.6677246, 2308.3510742

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1595175, upper bound: 2561.1567814
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1595679, upper bound: 2561.1595679
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -953.2823486, 1488.2750244, -953.2823486, 1488.2750244, -2441.5573730, 2441.5573730
1: -737.0429077, 1371.3309326, -737.0429077, 1371.3309326, -2108.3735352, 2108.3735352
2: -642.0244751, 1415.4780273, -642.0244751, 1415.4780273, -2057.5021973, 2057.5021973
3: -999.6423340, 1408.9439697, -999.6423340, 1408.9439697, -2408.5864258, 2408.5864258
4: -788.9846191, 1500.8854980, -788.9846191, 1500.8854980, -2289.8696289, 2289.8696289

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1595166, upper bound: 2561.1567805
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1595688, upper bound: 2561.1595688
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -950.0498047, 1480.7639160, -1066.0002441, 1663.8382568, -2613.8881836, 2546.7641602
1: -734.3095093, 1364.2473145, -825.9937134, 1535.1873779, -2269.4965820, 2190.2409668
2: -639.4960938, 1408.5468750, -720.1893921, 1582.3856201, -2221.8818359, 2128.7353516
3: -996.5464478, 1402.3671875, -1119.0744629, 1578.8870850, -2575.4335938, 2521.4416504
4: -785.8632202, 1494.2445068, -885.5877686, 1678.4027100, -2464.2658691, 2379.8322754

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -965.7675781, 1505.3173828, -1097.1602783, 1712.1324463, -2677.8999023, 2602.4775391
1: -746.6239624, 1386.8105469, -850.5893555, 1580.3242188, -2326.9479980, 2237.3999023
2: -650.1917114, 1432.0162354, -741.3272705, 1629.1556396, -2279.3471680, 2173.3435059
3: -1013.3558350, 1425.5120850, -1152.6741943, 1625.0007324, -2638.3564453, 2578.1857910
4: -798.8980713, 1519.3250732, -911.2078247, 1728.4595947, -2527.3576660, 2430.5329590

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1767678, upper bound: 2561.1704298
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1755528, upper bound: 2561.1688692
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -926.2637939, 1446.8006592, -1060.8154297, 1655.4340820, -2581.6977539, 2507.6159668
1: -716.6834106, 1333.1643066, -822.2328491, 1528.0776367, -2244.7609863, 2155.3967285
2: -624.3098755, 1376.1907959, -716.6336670, 1574.8680420, -2199.1779785, 2092.8244629
3: -972.1969604, 1369.6818848, -1113.8774414, 1571.0373535, -2543.2343750, 2483.5593262
4: -767.1339111, 1459.4760742, -881.0083008, 1670.5784912, -2437.7124023, 2340.4843750

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1526698, upper bound: 2561.1556448
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1526119, upper bound: 2561.1535259
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -936.1203613, 1461.1422119, -1060.3055420, 1656.7171631, -2592.8374023, 2521.4477539
1: -723.5245361, 1346.2982178, -823.2644043, 1528.9926758, -2252.5170898, 2169.5625000
2: -630.2501831, 1389.6479492, -717.4339600, 1576.7292480, -2206.9794922, 2107.0820312
3: -981.3450317, 1383.2795410, -1115.9732666, 1571.9757080, -2553.3208008, 2499.2526855
4: -774.4719238, 1473.4427490, -881.5534058, 1673.1530762, -2447.6247559, 2354.9960938

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1681586, upper bound: 2561.1597813
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1655051, upper bound: 2561.1551612
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1017.4556885, 1585.0762939, -843.9020386, 1323.8022461, -2341.2575684, 2428.9782715
1: -787.2538452, 1463.6813965, -656.9483643, 1220.1994629, -2007.4532471, 2120.6298828
2: -686.4097900, 1508.1789551, -572.0822144, 1259.5561523, -1945.9658203, 2080.2609863
3: -1066.6779785, 1505.0780029, -890.9157104, 1253.3331299, -2320.0107422, 2395.9936523
4: -843.8798218, 1599.9127197, -702.7177124, 1336.4271240, -2180.3068848, 2302.6298828

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884538, upper bound: 2561.1875732
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1887769, upper bound: 2561.1887532
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1017.4556885, 1585.0762939, -953.1518555, 1491.0954590, -2508.5505371, 2538.2280273
1: -787.2538452, 1463.6813965, -739.8281860, 1374.1557617, -2161.4096680, 2203.5095215
2: -686.4097900, 1508.1789551, -644.2612305, 1418.6929932, -2105.1027832, 2152.4399414
3: -1066.6779785, 1505.0780029, -1003.9645996, 1411.9536133, -2478.6315918, 2509.0424805
4: -843.8798218, 1599.9127197, -791.4631348, 1505.5098877, -2349.3894043, 2391.3757324

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1884552, upper bound: 2561.1875732
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.1887769, upper bound: 2561.1887517
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1045.6462402, 1630.8074951, -843.9020386, 1323.8022461, -2369.4484863, 2474.7094727
1: -809.4635010, 1503.8724365, -656.9483643, 1220.1994629, -2029.6629639, 2160.8208008
2: -705.4038086, 1551.8363037, -572.0822144, 1259.5561523, -1964.9599609, 2123.9184570
3: -1097.2976074, 1545.3623047, -890.9157104, 1253.3331299, -2350.6306152, 2436.2775879
4: -866.2383423, 1646.3774414, -702.7177124, 1336.4271240, -2202.6655273, 2349.0949707

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.67 + 416.49 = 420.16 seconds
