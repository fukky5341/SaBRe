## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 9177.495374428498


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3932.7343750, 6716.3618164, -3932.7343750, 6716.3618164, -10649.0947266, 10649.0947266)
1: (-677.7418213, 871.9747925, -677.7418213, 871.9747925, -1549.7165527, 1549.7165527)
2: (-520.9997559, 1149.9592285, -520.9997559, 1149.9592285, -1670.9588623, 1670.9589844)
3: (-550.2935181, 1535.1552734, -550.2935181, 1535.1552734, -2085.4487305, 2085.4487305)
4: (-452.5996704, 1453.1818848, -452.5996704, 1453.1818848, -1905.7814941, 1905.7814941)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.40 + 2.11 = 4.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9177.5871503, upper bound: 9177.5871501

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861427, upper bound: 9177.5868403
time: 0.77 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861130
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -9177.5861427, upper bound: 9177.5868403
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861130

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3861.6340332, 6607.7148438, -3913.2919922, 6686.7998047, -10548.4306641, 10521.0058594
1: -667.9061890, 857.4456177, -675.0779419, 868.0106201, -1535.9167480, 1532.5235596
2: -512.4591064, 1130.6225586, -518.6737671, 1144.6767578, -1657.1357422, 1649.2962646
3: -540.5580444, 1508.8596191, -547.6389771, 1527.9803467, -2068.5383301, 2056.4985352
4: -444.9967651, 1428.4781494, -450.5288391, 1446.4333496, -1891.4301758, 1879.0069580

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861124
time: 0.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861131
time: 0.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4071.1005859, 6909.0424805, -3815.3251953, 6514.6420898, -10585.7421875, 10724.3671875
1: -696.0803833, 897.3737183, -656.5891724, 845.8390503, -1541.9194336, 1553.9628906
2: -536.0514526, 1185.8249512, -505.2734070, 1116.2990723, -1652.3505859, 1691.0983887
3: -569.6372681, 1582.8814697, -534.1718750, 1489.5866699, -2059.2238770, 2117.0529785
4: -466.6393738, 1498.3332520, -439.1775208, 1410.6595459, -1877.2989502, 1937.5106201

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861132
time: 0.87 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861132
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.51 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861124
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861131
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861132
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -9177.5861132, upper bound: 9177.5861132

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3861.6340332, 6607.7148438, -3861.6340332, 6607.7148438, -10469.3466797, 10469.3466797
1: -667.9061890, 857.4456177, -667.9061890, 857.4456177, -1525.3516846, 1525.3516846
2: -512.4591064, 1130.6225586, -512.4591064, 1130.6225586, -1643.0816650, 1643.0816650
3: -540.5580444, 1508.8596191, -540.5580444, 1508.8596191, -2049.4177246, 2049.4177246
4: -444.9967651, 1428.4781494, -444.9967651, 1428.4781494, -1873.4748535, 1873.4748535

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843615, upper bound: 9177.5862165
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843436, upper bound: 9177.5853571
time: 0.71 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3861.6340332, 6607.7148438, -4071.1005859, 6909.0424805, -10770.6757812, 10678.8134766
1: -667.9061890, 857.4456177, -696.0803833, 897.3737183, -1565.2799072, 1553.5260010
2: -512.4591064, 1130.6225586, -536.0514526, 1185.8249512, -1698.2840576, 1666.6740723
3: -540.5580444, 1508.8596191, -569.6372681, 1582.8814697, -2123.4394531, 2078.4968262
4: -444.9967651, 1428.4781494, -466.6393738, 1498.3332520, -1943.3300781, 1895.1175537

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843622, upper bound: 9177.5862165
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5843436, upper bound: 9177.5853571
time: 0.79 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4071.1005859, 6909.0424805, -3861.6340332, 6607.7148438, -10678.8134766, 10770.6757812
1: -696.0803833, 897.3737183, -667.9061890, 857.4456177, -1553.5260010, 1565.2799072
2: -536.0514526, 1185.8249512, -512.4591064, 1130.6225586, -1666.6740723, 1698.2840576
3: -569.6372681, 1582.8814697, -540.5580444, 1508.8596191, -2078.4968262, 2123.4394531
4: -466.6393738, 1498.3332520, -444.9967651, 1428.4781494, -1895.1175537, 1943.3300781

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850607, upper bound: 9177.5853955
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477
time: 0.79 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4071.1005859, 6909.0424805, -4071.1005859, 6909.0424805, -10980.1425781, 10980.1425781
1: -696.0803833, 897.3737183, -696.0803833, 897.3737183, -1593.4541016, 1593.4541016
2: -536.0514526, 1185.8249512, -536.0514526, 1185.8249512, -1721.8764648, 1721.8764648
3: -569.6372681, 1582.8814697, -569.6372681, 1582.8814697, -2152.5185547, 2152.5185547
4: -466.6393738, 1498.3332520, -466.6393738, 1498.3332520, -1964.9725342, 1964.9725342

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850607, upper bound: 9177.5853953
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.80 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5843615, upper bound: 9177.5862165
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5843436, upper bound: 9177.5853571
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5843622, upper bound: 9177.5862165
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5843436, upper bound: 9177.5853571
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5850607, upper bound: 9177.5853955
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5850607, upper bound: 9177.5853953
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.80
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3779.1152344, 6473.2001953, -3818.3261719, 6537.2275391, -10316.3427734, 10291.5263672
1: -655.0988159, 839.7410889, -661.2041016, 848.1538696, -1503.2526855, 1500.9451904
2: -501.9861145, 1107.4720459, -506.9684448, 1118.4945068, -1620.4804688, 1614.4403076
3: -529.2615967, 1477.5150146, -534.6397095, 1492.4255371, -2021.6870117, 2012.1545410
4: -435.8514404, 1398.9156494, -440.2049866, 1412.9775391, -1848.8287354, 1839.1206055

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5842104, upper bound: 9177.5861517
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862504, upper bound: 9177.5868143
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4535.4140625, 7801.3769531, -3770.3820801, 6466.2968750, -11001.7089844, 11571.7587891
1: -791.5411987, 1008.3140869, -654.0256348, 838.0330811, -1629.5742188, 1662.3397217
2: -604.0177002, 1329.6524658, -501.0592957, 1106.3953857, -1710.4130859, 1830.7116699
3: -636.1485596, 1776.3583984, -528.3242188, 1475.6331787, -2111.7817383, 2304.6826172
4: -524.5239868, 1679.3609619, -434.9992676, 1397.8247070, -1922.3486328, 2114.3603516

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841886, upper bound: 9177.5852426
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862424, upper bound: 9177.5862434
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3779.1152344, 6473.2001953, -4028.1757812, 6840.5117188, -10619.6269531, 10501.3740234
1: -655.0988159, 839.7410889, -689.7568359, 888.3043213, -1543.4030762, 1529.4979248
2: -501.9861145, 1107.4720459, -530.7199097, 1173.9815674, -1675.9676514, 1638.1918945
3: -529.2615967, 1477.5150146, -563.8520508, 1566.7109375, -2095.9724121, 2041.3670654
4: -435.8514404, 1398.9156494, -461.9811096, 1483.2014160, -1919.0526123, 1860.8967285

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5838976, upper bound: 9177.5859030
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829649, upper bound: 9177.5857725
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4535.4140625, 7801.3769531, -3987.5976562, 6781.6015625, -11317.0156250, 11788.9726562
1: -791.5411987, 1008.3140869, -683.3646240, 879.4713745, -1671.0125732, 1691.6787109
2: -604.0177002, 1329.6524658, -525.5456543, 1164.0430908, -1768.0607910, 1855.1979980
3: -636.1485596, 1776.3583984, -558.5406494, 1552.7724609, -2188.9208984, 2334.8989258
4: -524.5239868, 1679.3609619, -457.4834595, 1470.6552734, -1995.1791992, 2136.8444824

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5838710, upper bound: 9177.5850122
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829393, upper bound: 9177.5849103
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3938.9438477, 6697.7949219, -3793.8203125, 6497.3408203, -10436.2822266, 10491.6152344
1: -676.1566162, 869.5827026, -657.4161377, 842.9569702, -1519.1135254, 1526.9986572
2: -519.7559204, 1147.1933594, -503.9284668, 1110.6746826, -1630.4304199, 1651.1217041
3: -551.8447876, 1532.5855713, -531.3256836, 1482.8438721, -2034.6883545, 2063.9111328
4: -452.3707275, 1449.1911621, -437.5340271, 1403.0098877, -1855.3806152, 1886.7252197

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860439, upper bound: 9177.5850883
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5860439, upper bound: 9177.5850890
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4368.1933594, 7493.6669922, -3724.3674316, 6390.4482422, -10758.6396484, 11218.0341797
1: -758.4396362, 968.8934326, -646.5141602, 828.1467896, -1586.5863037, 1615.4073486
2: -580.1182861, 1278.1990967, -495.3045654, 1092.4066162, -1672.5249023, 1773.5032959
3: -612.5208740, 1707.6627197, -521.9797974, 1457.5371094, -2070.0581055, 2229.6425781
4: -503.8991394, 1613.9832764, -429.9768677, 1379.9089355, -1883.8079834, 2043.9602051

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5859552, upper bound: 9177.5843566
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5853570, upper bound: 9177.5843426
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3938.9438477, 6697.7949219, -3996.2167969, 6789.3750000, -10728.3154297, 10694.0107422
1: -676.1566162, 869.5827026, -684.7976074, 881.6251831, -1557.7817383, 1554.3802490
2: -519.7559204, 1147.1933594, -526.8085327, 1163.9731445, -1683.7287598, 1674.0018311
3: -551.8447876, 1532.5855713, -559.5319824, 1554.3970947, -2106.2419434, 2092.1176758
4: -452.3707275, 1449.1911621, -458.5469055, 1470.5371094, -1922.9078369, 1907.7380371

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4368.1933594, 7493.6669922, -3941.7646484, 6706.9365234, -11075.1279297, 11435.4316406
1: -758.4396362, 968.8934326, -676.0520020, 869.8145142, -1628.2539062, 1644.9453125
2: -580.1182861, 1278.1990967, -519.9447632, 1150.3358154, -1730.4541016, 1798.1436768
3: -612.5208740, 1707.6627197, -552.1091919, 1534.9809570, -2147.5019531, 2259.7719727
4: -503.8991394, 1613.9832764, -452.5838928, 1453.2764893, -1957.1756592, 2066.5671387

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850114, upper bound: 9177.5843183
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5842812, upper bound: 9177.5842812
time: 0.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.94 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5842104, upper bound: 9177.5861517
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5862504, upper bound: 9177.5868143
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5841886, upper bound: 9177.5852426
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5862424, upper bound: 9177.5862434
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5838976, upper bound: 9177.5859030
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5829649, upper bound: 9177.5857725
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5838710, upper bound: 9177.5850122
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5829393, upper bound: 9177.5849103
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5860439, upper bound: 9177.5850883
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5860439, upper bound: 9177.5850890
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5859552, upper bound: 9177.5843566
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5853570, upper bound: 9177.5843426
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5850477, upper bound: 9177.5850477
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5850114, upper bound: 9177.5843183
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.94
Output dim: 0, lower bound: -9177.5842812, upper bound: 9177.5842812

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3715.9001465, 6364.4643555, -3698.8361816, 6340.8247070, -10056.7246094, 10063.3007812
1: -643.8072510, 825.7962646, -642.9371338, 822.6820679, -1466.4892578, 1468.7333984
2: -493.5046692, 1089.0859375, -491.7440186, 1083.1137695, -1576.6184082, 1580.8298340
3: -520.2738647, 1452.8648682, -518.1416016, 1445.9764404, -1966.2500000, 1971.0063477
4: -428.4457092, 1375.8280029, -426.8108521, 1367.6956787, -1796.1413574, 1802.6385498

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828093, upper bound: 9177.5858434
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5835636, upper bound: 9177.5848112
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841174, upper bound: 9177.5860584
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3772.5869141, 6462.0195312, -3788.8186035, 6486.6235352, -10259.2099609, 10250.8378906
1: -653.9963379, 838.2731323, -656.2157593, 841.5127563, -1495.5090332, 1494.4888916
2: -501.1090393, 1105.5986328, -502.9973755, 1110.0296631, -1611.1386719, 1608.5959473
3: -528.3354492, 1474.9716797, -530.4546509, 1480.9245605, -2009.2600098, 2005.4262695
4: -435.0898743, 1396.5607910, -436.7606506, 1402.3378906, -1837.4274902, 1833.3214111

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5852546, upper bound: 9177.5852102
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5852546, upper bound: 9177.5868139
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4468.7387695, 7687.1982422, -3662.1289062, 6287.0400391, -10755.7773438, 11349.3271484
1: -779.8642578, 993.6373291, -637.2210693, 814.7689209, -1594.6331787, 1630.8582764
2: -595.1331177, 1310.3150635, -487.1593018, 1073.8502197, -1668.9833984, 1797.4739990
3: -626.7661743, 1750.4678955, -513.3401489, 1433.1845703, -2059.9506836, 2263.8081055
4: -516.6903076, 1655.0358887, -422.7897339, 1356.1890869, -1872.8791504, 2077.8254395

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827887, upper bound: 9177.5851037
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4528.9501953, 7790.0336914, -3740.6538086, 6415.1108398, -10944.0566406, 11530.6875000
1: -790.4243164, 1006.8434448, -648.9752197, 831.3243408, -1621.7486572, 1655.8186035
2: -603.1349487, 1327.7362061, -497.0469055, 1097.8048096, -1700.9396973, 1824.7829590
3: -635.2205200, 1773.8111572, -524.1040039, 1464.0109863, -2099.2314453, 2297.9150391
4: -523.7615967, 1676.9588623, -431.5219116, 1387.0301514, -1910.7917480, 2108.4804688

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5852426, upper bound: 9177.5841878
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5852426, upper bound: 9177.5862434
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3715.9001465, 6364.4643555, -3910.0954590, 6647.1577148, -10363.0576172, 10274.5595703
1: -643.8072510, 825.7962646, -671.8306274, 863.0271606, -1506.8344727, 1497.6265869
2: -493.5046692, 1089.0859375, -515.9346313, 1139.2639160, -1632.7683105, 1605.0205078
3: -520.2738647, 1452.8648682, -547.7094727, 1521.2109375, -2041.4847412, 2000.5742188
4: -428.4457092, 1375.8280029, -449.0249634, 1438.8017578, -1867.2474365, 1824.8530273

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5824477, upper bound: 9177.5856375
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5836824, upper bound: 9177.5852611
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837962, upper bound: 9177.5857980
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3772.5869141, 6462.0195312, -4000.5305176, 6793.6293945, -10566.2158203, 10462.5498047
1: -653.9963379, 838.2731323, -685.1307983, 882.1505127, -1536.1468506, 1523.4039307
2: -501.1090393, 1105.5986328, -527.0250854, 1166.1124268, -1667.2214355, 1632.6235352
3: -528.3354492, 1474.9716797, -559.9563599, 1556.0206299, -2084.3557129, 2034.9278564
4: -435.0898743, 1396.5607910, -458.7833557, 1473.3112793, -1908.4008789, 1855.3441162

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5852162
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829418, upper bound: 9177.5857725
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4468.7387695, 7687.1982422, -3870.3464355, 6594.0527344, -11062.7910156, 11557.5449219
1: -779.8642578, 993.6373291, -665.9492188, 855.1146851, -1634.9790039, 1659.5864258
2: -595.1331177, 1310.3150635, -511.2473450, 1129.9057617, -1725.0388184, 1821.5622559
3: -626.7661743, 1750.4678955, -542.5671387, 1507.8698730, -2134.6359863, 2293.0351562
4: -516.6903076, 1655.0358887, -444.8381653, 1427.0556641, -1943.7458496, 2099.8732910

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5824261, upper bound: 9177.5849062
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5823949, upper bound: 9177.5837592
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4528.9501953, 7790.0336914, -3960.7109375, 6735.7006836, -11264.6484375, 11750.7441406
1: -790.4243164, 1006.8434448, -678.8225708, 873.4548340, -1663.8791504, 1685.6660156
2: -603.1349487, 1327.7362061, -521.9298096, 1156.3598633, -1759.4948730, 1849.6660156
3: -635.2205200, 1773.8111572, -554.7399902, 1542.3375244, -2177.5581055, 2328.5512695
4: -523.7615967, 1676.9588623, -454.3600769, 1461.0109863, -1984.7724609, 2131.3186035

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5840546
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5849103
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3938.9438477, 6697.7949219, -3740.8774414, 6411.9614258, -10350.9042969, 10438.6699219
1: -676.1566162, 869.5827026, -649.3037720, 831.7255249, -1507.8820801, 1518.8864746
2: -519.7559204, 1147.1933594, -497.3157043, 1095.1785889, -1614.9343262, 1644.5089111
3: -551.8447876, 1532.5855713, -524.1222534, 1462.6403809, -2014.4848633, 2056.7075195
4: -452.3707275, 1449.1911621, -431.7399292, 1383.2468262, -1835.6175537, 1880.9311523

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5857741, upper bound: 9177.5850807
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5852878, upper bound: 9177.5852829
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5859511, upper bound: 9177.5852645
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3938.9438477, 6697.7949219, -4094.3361816, 7080.5703125, -11019.5136719, 10792.1308594
1: -676.1566162, 869.5827026, -719.4486084, 914.4120483, -1590.5686035, 1589.0312500
2: -519.7559204, 1147.1933594, -547.8181152, 1203.4143066, -1723.1700439, 1695.0114746
3: -551.8447876, 1532.5855713, -574.5559692, 1608.2004395, -2160.0451660, 2107.1416016
4: -452.3707275, 1449.1911621, -475.1710815, 1519.7420654, -1972.1127930, 1924.3623047

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5857741, upper bound: 9177.5850807
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5852878, upper bound: 9177.5852829
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5859511, upper bound: 9177.5852645
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4323.2495117, 7420.6196289, -3624.4736328, 6227.8984375, -10551.1484375, 11045.0937500
1: -751.6095581, 959.3450317, -631.1663208, 806.7566528, -1558.3662109, 1590.5113525
2: -574.4973755, 1265.5528564, -482.6267395, 1064.5180664, -1639.0151367, 1748.1794434
3: -606.4149170, 1690.6735840, -508.2644043, 1419.7342529, -2026.1490479, 2198.9379883
4: -498.9666138, 1597.8590088, -418.8901367, 1344.3907471, -1843.3574219, 2016.7490234

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5856148, upper bound: 9177.5838894
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5855140, upper bound: 9177.5829577
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4297.7773438, 7381.7290039, -4447.3828125, 7653.3334961, -11951.1103516, 11829.1083984
1: -747.3599243, 953.6936035, -776.7753296, 989.0999756, -1736.4598389, 1730.4689941
2: -571.1375122, 1259.1805420, -592.5501709, 1304.1907959, -1875.3282471, 1851.7305908
3: -602.7319946, 1681.7371826, -623.8907471, 1742.5041504, -2345.2360840, 2305.6279297
4: -495.9942932, 1590.0904541, -514.5271606, 1647.1973877, -2143.1914062, 2104.6171875

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5850122, upper bound: 9177.5838718
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849103, upper bound: 9177.5829393
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3938.9438477, 6697.7949219, -3938.9438477, 6697.7949219, -10636.7363281, 10636.7363281
1: -676.1566162, 869.5827026, -676.1566162, 869.5827026, -1545.7392578, 1545.7392578
2: -519.7559204, 1147.1933594, -519.7559204, 1147.1933594, -1666.9490967, 1666.9490967
3: -551.8447876, 1532.5855713, -551.8447876, 1532.5855713, -2084.4304199, 2084.4304199
4: -452.3707275, 1449.1911621, -452.3707275, 1449.1911621, -1901.5618896, 1901.5618896

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849563, upper bound: 9177.5852827
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849562, upper bound: 9177.5852637
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3938.9438477, 6697.7949219, -4368.1933594, 7493.6669922, -11432.6103516, 11065.9853516
1: -676.1566162, 869.5827026, -758.4396362, 968.8934326, -1645.0500488, 1628.0222168
2: -519.7559204, 1147.1933594, -580.1182861, 1278.1990967, -1797.9547119, 1727.3116455
3: -551.8447876, 1532.5855713, -612.5208740, 1707.6627197, -2259.5075684, 2145.1064453
4: -452.3707275, 1449.1911621, -503.8991394, 1613.9832764, -2066.3540039, 1953.0903320

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849563, upper bound: 9177.5852829
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849562, upper bound: 9177.5852637
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4323.2495117, 7420.6196289, -3853.8293457, 6564.1376953, -10887.3867188, 11274.4482422
1: -751.6095581, 959.3450317, -662.9356079, 851.0617676, -1602.6712646, 1622.2806396
2: -574.4973755, 1265.5528564, -508.8819580, 1125.8593750, -1700.3566895, 1774.4345703
3: -606.4149170, 1690.6735840, -540.1511841, 1501.6351318, -2108.0500488, 2230.8247070
4: -498.9666138, 1597.8590088, -442.9009399, 1422.0192871, -1920.9858398, 2040.7598877

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5831010, upper bound: 9177.5837575
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4297.7773438, 7381.7290039, -4607.1909180, 7888.8090820, -12186.5849609, 11988.9189453
1: -747.3599243, 953.6936035, -797.4747925, 1019.3844604, -1766.7443848, 1751.1684570
2: -571.1375122, 1259.1805420, -610.5800781, 1348.8946533, -1920.0321045, 1869.7606201
3: -602.7319946, 1681.7371826, -646.4379272, 1799.4555664, -2402.1872559, 2328.1745605
4: -495.9942932, 1590.0904541, -530.7012939, 1702.8721924, -2198.8662109, 2120.7912598

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828619, upper bound: 9177.5837488
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828197, upper bound: 9177.5828197
time: 0.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.42 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5835636, upper bound: 9177.5848112
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5841174, upper bound: 9177.5860584
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5852546, upper bound: 9177.5852102
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5852546, upper bound: 9177.5868139
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5827887, upper bound: 9177.5851037
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5852426, upper bound: 9177.5841878
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5852426, upper bound: 9177.5862434
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5836824, upper bound: 9177.5852611
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5837962, upper bound: 9177.5857980
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5852162
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5829418, upper bound: 9177.5857725
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5824261, upper bound: 9177.5849062
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5823949, upper bound: 9177.5837592
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5840546
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5849103
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5852878, upper bound: 9177.5852829
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5859511, upper bound: 9177.5852645
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5852878, upper bound: 9177.5852829
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5859511, upper bound: 9177.5852645
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5856148, upper bound: 9177.5838894
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5855140, upper bound: 9177.5829577
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5850122, upper bound: 9177.5838718
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5849103, upper bound: 9177.5829393
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5849563, upper bound: 9177.5852827
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5849562, upper bound: 9177.5852637
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5849563, upper bound: 9177.5852829
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5849562, upper bound: 9177.5852637
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5831010, upper bound: 9177.5837575
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5828619, upper bound: 9177.5837488
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 0, lower bound: -9177.5828197, upper bound: 9177.5828197

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3690.0249023, 6320.3115234, -3683.7172852, 6315.2006836, -10005.2255859, 10004.0292969
1: -639.4639893, 820.0898438, -640.4197388, 819.3718262, -1458.8358154, 1460.5093994
2: -490.0565796, 1081.2921143, -489.7442322, 1078.5928955, -1568.6491699, 1571.0363770
3: -516.6007690, 1442.7359619, -515.9887085, 1440.0897217, -1956.6904297, 1958.7246094
4: -425.4356079, 1366.0732422, -425.0618896, 1362.0373535, -1787.4727783, 1791.1351318

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5835636, upper bound: 9177.5848112
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5835629, upper bound: 9177.5848112
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3748.7114258, 6428.3271484, -3675.9973145, 6302.9042969, -10051.6152344, 10104.3242188
1: -651.5828247, 833.1761475, -639.2084961, 817.6456909, -1469.2281494, 1472.3845215
2: -498.4365540, 1097.5628662, -488.7815857, 1075.9642334, -1574.4007568, 1586.3444824
3: -525.4647827, 1466.3040771, -515.0266724, 1437.0908203, -1962.5552979, 1981.3308105
4: -432.7620239, 1386.4736328, -424.2238770, 1358.6774902, -1791.4394531, 1810.6975098

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841166, upper bound: 9177.5860584
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841174, upper bound: 9177.5860584
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3655.5886230, 6271.7954102, -3788.8186035, 6486.6235352, -10142.2109375, 10060.6142578
1: -636.4227295, 813.5534058, -656.2157593, 841.5127563, -1477.9355469, 1469.7689209
2: -486.3359985, 1071.2492676, -502.9973755, 1110.0296631, -1596.3654785, 1574.2465820
3: -512.2059326, 1429.7664795, -530.4546509, 1480.9245605, -1993.1304932, 1960.2210693
4: -422.1175537, 1352.5257568, -436.7606506, 1402.3378906, -1824.4552002, 1789.2863770

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5852105
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5852102
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3749.2917480, 6422.1757812, -3788.8186035, 6486.6235352, -10235.9150391, 10210.9941406
1: -650.0700684, 833.0407104, -656.2157593, 841.5127563, -1491.5827637, 1489.2563477
2: -497.9827576, 1098.9267578, -502.9973755, 1110.0296631, -1608.0124512, 1601.9239502
3: -525.0321655, 1465.9018555, -530.4546509, 1480.9245605, -2005.9567871, 1996.3564453
4: -432.3747253, 1388.1757812, -436.7606506, 1402.3378906, -1834.7126465, 1824.9364014

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5868139
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5868143
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4369.2983398, 7527.1455078, -3612.6870117, 6206.5151367, -10575.8125000, 11139.8310547
1: -764.4841309, 972.7909546, -629.6001587, 804.3834229, -1568.8675537, 1602.3911133
2: -582.5541992, 1282.9584961, -480.8674316, 1060.3984375, -1642.9526367, 1763.8259277
3: -612.7703247, 1713.1683350, -506.4031372, 1414.5455322, -2027.3156738, 2219.5710449
4: -505.8053894, 1620.3533936, -417.2857666, 1339.1103516, -1844.9157715, 2037.6390381

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5421.3022461, 9373.8183594, -3569.2758789, 6143.4794922, -11564.7812500, 12943.0937500
1: -950.5028076, 1212.4326172, -622.5816650, 796.1206055, -1746.6234131, 1835.0141602
2: -725.6915283, 1602.3067627, -475.6535034, 1050.2974854, -1775.9887695, 2077.9599609
3: -760.6499023, 2129.4760742, -500.3911743, 1399.3499756, -2160.0000000, 2629.8671875
4: -629.9783936, 2021.2227783, -412.4781494, 1326.3923340, -1956.3707275, 2433.7009277

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4414.9814453, 7609.6201172, -3740.6538086, 6415.1108398, -10830.0898438, 11350.2734375
1: -773.5969238, 983.4029541, -648.9752197, 831.3243408, -1604.9212646, 1632.3781738
2: -589.2892456, 1295.1297607, -497.0469055, 1097.8048096, -1687.0939941, 1792.1765137
3: -619.6710815, 1730.8128662, -524.1040039, 1464.0109863, -2083.6821289, 2254.9169922
4: -511.6530151, 1635.2827148, -431.5219116, 1387.0301514, -1898.6831055, 2066.8044434

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5840519, upper bound: 9177.5827887
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5827617
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4505.7573242, 7749.3828125, -3740.6538086, 6415.1108398, -10920.8642578, 11490.0371094
1: -786.4260254, 1001.5693970, -648.9752197, 831.3243408, -1617.7503662, 1650.5446777
2: -599.9754028, 1320.8731689, -497.0469055, 1097.8048096, -1697.7802734, 1817.9199219
3: -631.8952026, 1764.6744385, -524.1040039, 1464.0109863, -2095.9062500, 2288.7778320
4: -521.0288696, 1668.3464355, -431.5219116, 1387.0301514, -1908.0589600, 2099.8684082

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5840519, upper bound: 9177.5853433
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5853254
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3690.0249023, 6320.3115234, -3895.6464844, 6622.6523438, -10312.6777344, 10215.9580078
1: -639.4639893, 820.0898438, -669.4351196, 859.8519897, -1499.3159180, 1489.5246582
2: -490.0565796, 1081.2921143, -514.0304565, 1134.9146729, -1624.9709473, 1595.3225098
3: -516.6007690, 1442.7359619, -545.6843262, 1515.5764160, -2032.1772461, 1988.4202881
4: -425.4356079, 1366.0732422, -447.3638611, 1433.3511963, -1858.7867432, 1813.4370117

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5836824, upper bound: 9177.5852611
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5836824, upper bound: 9177.5852611
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3748.7114258, 6428.3271484, -3886.1057129, 6607.6523438, -10356.3623047, 10314.4326172
1: -651.5828247, 833.1761475, -667.9417725, 857.8045044, -1509.3873291, 1501.1176758
2: -498.4365540, 1097.5628662, -512.8441772, 1131.8670654, -1630.3034668, 1610.4069824
3: -525.4647827, 1466.3040771, -544.4286499, 1511.8878174, -2037.3522949, 2010.7326660
4: -432.7620239, 1386.4736328, -446.3135986, 1429.5015869, -1862.2636719, 1832.7872314

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837962, upper bound: 9177.5857980
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837972, upper bound: 9177.5857980
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3655.5886230, 6271.7954102, -4000.5305176, 6793.6293945, -10449.2167969, 10272.3261719
1: -636.4227295, 813.5534058, -685.1307983, 882.1505127, -1518.5732422, 1498.6842041
2: -486.3359985, 1071.2492676, -527.0250854, 1166.1124268, -1652.4483643, 1598.2742920
3: -512.2059326, 1429.7664795, -559.9563599, 1556.0206299, -2068.2263184, 1989.7227783
4: -422.1175537, 1352.5257568, -458.7833557, 1473.3112793, -1895.4285889, 1811.3090820

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829418, upper bound: 9177.5852162
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5852164
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3749.2917480, 6422.1757812, -4000.5305176, 6793.6293945, -10542.9208984, 10422.7060547
1: -650.0700684, 833.0407104, -685.1307983, 882.1505127, -1532.2205811, 1518.1715088
2: -497.9827576, 1098.9267578, -527.0250854, 1166.1124268, -1664.0952148, 1625.9515381
3: -525.0321655, 1465.9018555, -559.9563599, 1556.0206299, -2081.0524902, 2025.8579102
4: -432.3747253, 1388.1757812, -458.7833557, 1473.3112793, -1905.6860352, 1846.9591064

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5857725
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5857723
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4369.2983398, 7527.1455078, -3828.1516113, 6524.2910156, -10893.5898438, 11355.2968750
1: -764.4841309, 972.7909546, -659.3768921, 846.1026001, -1610.5865479, 1632.1678467
2: -582.5541992, 1282.9584961, -505.7956238, 1118.4320068, -1700.9860840, 1788.7541504
3: -612.7703247, 1713.1683350, -536.6308594, 1491.8276367, -2104.5979004, 2249.7985840
4: -505.8053894, 1620.3533936, -440.0963440, 1412.5333252, -1918.3387451, 2060.4497070

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5421.3022461, 9373.8183594, -3761.8457031, 6420.9174805, -11842.2197266, 13135.6630859
1: -950.5028076, 1212.4326172, -648.3282471, 833.2429199, -1783.7457275, 1860.7607422
2: -725.6915283, 1602.3067627, -497.5271606, 1101.5344238, -1827.2259521, 2099.8339844
3: -760.6499023, 2129.4760742, -527.1614380, 1467.6970215, -2228.3469238, 2656.6374512
4: -629.9783936, 2021.2227783, -432.5283508, 1390.9626465, -2020.9410400, 2453.7512207

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4414.9814453, 7609.6201172, -3960.7109375, 6735.7006836, -11150.6816406, 11570.3310547
1: -773.5969238, 983.4029541, -678.8225708, 873.4548340, -1647.0517578, 1662.2255859
2: -589.2892456, 1295.1297607, -521.9298096, 1156.3598633, -1745.6491699, 1817.0595703
3: -619.6710815, 1730.8128662, -554.7399902, 1542.3375244, -2162.0085449, 2285.5527344
4: -511.6530151, 1635.2827148, -454.3600769, 1461.0109863, -1972.6640625, 2089.6420898

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5840543
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5840545
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4505.7573242, 7749.3828125, -3960.7109375, 6735.7006836, -11241.4560547, 11710.0937500
1: -786.4260254, 1001.5693970, -678.8225708, 873.4548340, -1659.8807373, 1680.3919678
2: -599.9754028, 1320.8731689, -521.9298096, 1156.3598633, -1756.3352051, 1842.8029785
3: -631.8952026, 1764.6744385, -554.7399902, 1542.3375244, -2174.2324219, 2319.4140625
4: -521.0288696, 1668.3464355, -454.3600769, 1461.0109863, -1982.0397949, 2122.7062988

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5848992
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828829, upper bound: 9177.5848992
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3912.0847168, 6652.1025391, -3726.3051758, 6386.9858398, -10299.0703125, 10378.4033203
1: -671.6961060, 863.6645508, -646.8442993, 828.4860229, -1500.1821289, 1510.5085449
2: -516.2010498, 1139.0972900, -495.3624268, 1090.8062744, -1607.0073242, 1634.4593506
3: -548.0826416, 1522.0594482, -522.0504150, 1456.9323730, -2005.0150146, 2044.1098633
4: -449.2771606, 1439.0180664, -430.0390320, 1377.7747803, -1827.0520020, 1869.0571289

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853120
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853120
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3985.9565430, 6785.6152344, -3716.9052734, 6371.9306641, -10357.8847656, 10502.5205078
1: -686.2785034, 880.1289062, -645.3723145, 826.4227295, -1512.7011719, 1525.5012207
2: -526.5716553, 1159.6820068, -494.1877441, 1087.6439209, -1614.2155762, 1653.8696289
3: -559.0259399, 1551.4614258, -520.8403320, 1453.2947998, -2012.3208008, 2072.3015137
4: -458.3278198, 1464.8522949, -429.0118408, 1373.7744141, -1832.1022949, 1893.8641357

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853120
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853126
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3912.0847168, 6652.1025391, -4083.1394043, 7061.4521484, -10973.5361328, 10735.2382812
1: -671.6961060, 863.6645508, -717.5787964, 911.9306030, -1583.6267090, 1581.2434082
2: -516.2010498, 1139.0972900, -546.3137817, 1200.0180664, -1716.2191162, 1685.4108887
3: -548.0826416, 1522.0594482, -572.9432983, 1603.8269043, -2151.9094238, 2095.0026855
4: -449.2771606, 1439.0180664, -473.8738708, 1515.4968262, -1964.7739258, 1912.8916016

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3985.9565430, 6785.6152344, -4069.7236328, 7039.2172852, -11025.1728516, 10855.3388672
1: -686.2785034, 880.1289062, -715.3843994, 908.9240112, -1595.2025146, 1595.5133057
2: -526.5716553, 1159.6820068, -544.6140137, 1195.6232910, -1722.1949463, 1704.2958984
3: -559.0259399, 1551.4614258, -571.2146606, 1598.5433350, -2157.5693359, 2122.6757812
4: -458.3278198, 1464.8522949, -472.3907471, 1509.9290771, -1968.2568359, 1937.2430420

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5859028, upper bound: 9177.5851216
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5859511, upper bound: 9177.5852645
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4206.3657227, 7227.0673828, -3566.6962891, 6126.3671875, -10332.7324219, 10793.7636719
1: -733.4389648, 934.1078491, -620.5828247, 793.7888184, -1527.2277832, 1554.6903076
2: -559.7365112, 1230.7972412, -474.7431030, 1047.3519287, -1607.0882568, 1705.5402832
3: -590.3714600, 1645.2766113, -500.0307617, 1396.8916016, -1987.2630615, 2145.3073730
4: -486.1708374, 1553.4758301, -412.0265808, 1322.8547363, -1809.0256348, 1965.5023193

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5855108, upper bound: 9177.5824444
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5836287, upper bound: 9177.5832572
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5852754, upper bound: 9177.5833589
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4297.9340820, 7378.9501953, -3618.1660156, 6217.1040039, -10515.0351562, 10997.1162109
1: -747.4999390, 953.8648682, -630.1066284, 805.3339844, -1552.8339844, 1583.9714355
2: -571.2108154, 1258.4233398, -481.7787781, 1062.6937256, -1633.9042969, 1740.2017822
3: -602.8869019, 1681.0587158, -507.3707886, 1417.2720947, -2020.1589355, 2188.4294434
4: -496.0987244, 1588.8769531, -418.1547852, 1342.0986328, -1838.1972656, 2007.0316162

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849022, upper bound: 9177.5829281
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849022, upper bound: 9177.5829577
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4187.1049805, 7200.5332031, -4379.6562500, 7537.7592773, -11724.8642578, 11580.1894531
1: -730.4904785, 930.1587524, -764.9629517, 974.2111206, -1704.7015381, 1695.1217041
2: -557.3704834, 1226.3956299, -583.5557861, 1284.5863037, -1841.9567871, 1809.9514160
3: -587.6904297, 1638.9481201, -614.3767700, 1716.2526855, -2303.9428711, 2253.3249512
4: -484.0057983, 1548.2297363, -506.5951538, 1622.5002441, -2106.5061035, 2054.8249512

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849062, upper bound: 9177.5824261
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5837592, upper bound: 9177.5823949
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4270.2866211, 7335.9028320, -4441.0625000, 7642.2856445, -11912.5722656, 11776.9619141
1: -742.8439941, 947.6719360, -775.6904297, 987.6656494, -1730.5096436, 1723.3620605
2: -567.5335083, 1251.3034668, -591.6903687, 1302.3247070, -1869.8580322, 1842.9935303
3: -598.8925781, 1671.2165527, -622.9841919, 1740.0183105, -2338.9106445, 2294.2006836
4: -492.8517456, 1580.1860352, -513.7841797, 1644.8535156, -2137.7053223, 2093.9702148

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5840546, upper bound: 9177.5828830
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5840546, upper bound: 9177.5829393
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3912.0847168, 6652.1025391, -3924.5551758, 6673.2919922, -10585.3769531, 10576.6542969
1: -671.6961060, 863.6645508, -673.7619629, 866.4098511, -1538.1059570, 1537.4263916
2: -516.2010498, 1139.0972900, -517.8503418, 1142.8515625, -1659.0526123, 1656.9473877
3: -548.0826416, 1522.0594482, -549.8272095, 1526.9415283, -2075.0241699, 2071.8864746
4: -449.2771606, 1439.0180664, -450.7122498, 1443.7380371, -1893.0151367, 1889.7301025

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5840919, upper bound: 9177.5840919
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847831, upper bound: 9177.5848793
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3985.9565430, 6785.6152344, -3915.4956055, 6658.6015625, -10644.5566406, 10701.1113281
1: -686.2785034, 880.1289062, -672.3023682, 864.3952026, -1550.6737061, 1552.4312744
2: -526.5716553, 1159.6820068, -516.7047119, 1139.8009033, -1666.3725586, 1676.3864746
3: -559.0259399, 1551.4614258, -548.6323242, 1523.4195557, -2082.4453125, 2100.0937500
4: -458.3278198, 1464.8522949, -449.7066345, 1439.9161377, -1898.2438965, 1914.5589600

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5840919, upper bound: 9177.5840919
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5847808, upper bound: 9177.5847808
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3912.0847168, 6652.1025391, -4356.7680664, 7474.2670898, -11386.3515625, 11008.8681641
1: -671.6961060, 863.6645508, -756.5440063, 966.3838501, -1638.0799561, 1620.2084961
2: -516.2010498, 1139.0972900, -578.6003418, 1274.7409668, -1790.9420166, 1717.6975098
3: -548.0826416, 1522.0594482, -610.8995972, 1703.2004395, -2251.2832031, 2132.9589844
4: -449.2771606, 1439.0180664, -502.5719910, 1609.6442871, -2058.9213867, 1941.5898438

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5825396, upper bound: 9177.5808159
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5844454, upper bound: 9177.5847837
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3985.9565430, 6785.6152344, -4340.4897461, 7447.4726562, -11433.4277344, 11126.1054688
1: -686.2785034, 880.1289062, -753.8855591, 962.7644043, -1649.0429688, 1634.0144043
2: -526.5716553, 1159.6820068, -576.5363770, 1269.5242920, -1796.0959473, 1736.2182617
3: -559.0259399, 1551.4614258, -608.7601318, 1696.8293457, -2255.8549805, 2160.2214355
4: -458.3278198, 1464.8522949, -500.7884216, 1603.1113281, -2061.4392090, 1965.6407471

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826074, upper bound: 9177.5826307
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5844453, upper bound: 9177.5847650
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4206.3657227, 7227.0673828, -3786.1271973, 6447.5000000, -10653.8652344, 11013.1943359
1: -733.4389648, 934.1078491, -650.8954468, 836.0250854, -1569.4641113, 1585.0030518
2: -559.7365112, 1230.7972412, -499.7656555, 1106.1151123, -1665.8515625, 1730.5628662
3: -590.3714600, 1645.2766113, -530.5650024, 1475.2648926, -2065.6362305, 2175.8415527
4: -486.1708374, 1553.4758301, -434.9726562, 1397.1856689, -1883.3564453, 1988.4484863

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4297.9340820, 7378.9501953, -3847.9973145, 6554.2480469, -10852.1796875, 11226.9472656
1: -747.4999390, 953.8648682, -661.9621582, 849.7623291, -1597.2622070, 1615.8270264
2: -571.2108154, 1258.4233398, -508.1026611, 1124.1955566, -1695.4063721, 1766.5256348
3: -602.8869019, 1681.0587158, -539.3273315, 1499.3795166, -2102.2663574, 2220.3859863
4: -496.0987244, 1588.8769531, -442.2261047, 1419.9285889, -1916.0273438, 2031.1029053

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4187.1049805, 7200.5332031, -4542.6811523, 7776.7573242, -11963.8613281, 11743.2138672
1: -730.4904785, 930.1587524, -786.0288696, 1004.9501343, -1735.4404297, 1716.1876221
2: -557.3704834, 1226.3956299, -601.8361206, 1329.8183594, -1887.1888428, 1828.2316895
3: -587.6904297, 1638.9481201, -637.2315063, 1774.1452637, -2361.8356934, 2276.1796875
4: -484.0057983, 1548.2297363, -523.0949097, 1678.9693604, -2162.9750977, 2071.3244629

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827972, upper bound: 9177.5823040
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5811765, upper bound: 9177.5822500
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4270.2866211, 7335.9028320, -4600.5083008, 7877.5532227, -12147.8398438, 11936.4091797
1: -742.8439941, 947.6719360, -796.3699341, 1017.9084473, -1760.7523193, 1744.0416260
2: -567.5335083, 1251.3034668, -609.6967163, 1346.9844971, -1914.5179443, 1861.0002441
3: -598.8925781, 1671.2165527, -645.5029297, 1796.8795166, -2395.7714844, 2316.7192383
4: -492.8517456, 1580.1860352, -529.9366455, 1700.4685059, -2193.3203125, 2110.1225586

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828197, upper bound: 9177.5828197
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5828198, upper bound: 9177.5828198
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.81 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5835636, upper bound: 9177.5848112
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5835629, upper bound: 9177.5848112
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5841166, upper bound: 9177.5860584
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5841174, upper bound: 9177.5860584
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5852105
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5852102
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5868139
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5841716, upper bound: 9177.5868143
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5827617, upper bound: 9177.5840332
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5840519, upper bound: 9177.5827887
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5827617
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5840519, upper bound: 9177.5853433
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5853254
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5836824, upper bound: 9177.5852611
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5836824, upper bound: 9177.5852611
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5837962, upper bound: 9177.5857980
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5837972, upper bound: 9177.5857980
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5829418, upper bound: 9177.5852162
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5852164
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5857725
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5829419, upper bound: 9177.5857723
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5823950, upper bound: 9177.5837592
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5840543
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5840545
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5828830, upper bound: 9177.5848992
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5828829, upper bound: 9177.5848992
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853120
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853120
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853120
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5863343, upper bound: 9177.5853126
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5859028, upper bound: 9177.5851216
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5859511, upper bound: 9177.5852645
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5836287, upper bound: 9177.5832572
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5852754, upper bound: 9177.5833589
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5849022, upper bound: 9177.5829281
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5849022, upper bound: 9177.5829577
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5849062, upper bound: 9177.5824261
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5837592, upper bound: 9177.5823949
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5840546, upper bound: 9177.5828830
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5840546, upper bound: 9177.5829393
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5840919, upper bound: 9177.5840919
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5847831, upper bound: 9177.5848793
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5840919, upper bound: 9177.5840919
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5847808, upper bound: 9177.5847808
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5825396, upper bound: 9177.5808159
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5844454, upper bound: 9177.5847837
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5826074, upper bound: 9177.5826307
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5844453, upper bound: 9177.5847650
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5830586, upper bound: 9177.5828284
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5827972, upper bound: 9177.5823040
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5811765, upper bound: 9177.5822500
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5828197, upper bound: 9177.5828197
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.81
Output dim: 0, lower bound: -9177.5828198, upper bound: 9177.5828198

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3690.0249023, 6320.3115234, -3640.4150391, 6246.1113281, -9936.1357422, 9960.7265625
1: -639.4639893, 820.0898438, -633.8971558, 810.2349854, -1449.6989746, 1453.9869385
2: -490.0565796, 1081.2921143, -484.3325195, 1066.7185059, -1556.7749023, 1565.6246338
3: -516.6007690, 1442.7359619, -510.0419312, 1423.8636475, -1940.4643555, 1952.7777100
4: -425.4356079, 1366.0732422, -420.3754272, 1346.8546143, -1772.2901611, 1786.4486084

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5835629, upper bound: 9177.5848112
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3690.0249023, 6320.3115234, -4402.4541016, 7588.7802734, -11278.8046875, 10722.7646484
1: -639.4639893, 820.0898438, -771.5627441, 980.6826782, -1620.1467285, 1591.6524658
2: -490.0565796, 1081.2921143, -587.6590576, 1291.3795166, -1781.4359131, 1668.9511719
3: -516.6007690, 1442.7359619, -617.8928833, 1725.9770508, -2242.5771484, 2060.6289062
4: -425.4356079, 1366.0732422, -510.2424316, 1630.5908203, -2056.0263672, 1876.3154297

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5835629, upper bound: 9177.5848112
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3748.7114258, 6428.3271484, -3632.1096191, 6232.8671875, -9981.5771484, 10060.4365234
1: -651.5828247, 833.1761475, -632.5969238, 808.3815308, -1459.9643555, 1465.7729492
2: -498.4365540, 1097.5628662, -483.2958984, 1063.9047852, -1562.3411865, 1580.8586426
3: -525.4647827, 1466.3040771, -509.0047302, 1420.6401367, -1946.1046143, 1975.3088379
4: -432.7620239, 1386.4736328, -419.4695129, 1343.2625732, -1776.0246582, 1805.9431152

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5835175, upper bound: 9177.5849454
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5836030, upper bound: 9177.5857799
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3748.7114258, 6428.3271484, -4390.1118164, 7567.5874023, -11316.2958984, 10818.4394531
1: -651.5828247, 833.1761475, -769.4498291, 977.8527222, -1629.4354248, 1602.6258545
2: -498.4365540, 1097.5628662, -586.0227661, 1287.2869873, -1785.7232666, 1683.5856934
3: -525.4647827, 1466.3040771, -616.2666626, 1721.0344238, -2246.4992676, 2082.5705566
4: -432.7620239, 1386.4736328, -508.8139954, 1625.4433594, -2058.2053223, 1895.2874756

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5835175, upper bound: 9177.5849455
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5836030, upper bound: 9177.5857799
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3655.5886230, 6271.7954102, -3749.2917480, 6422.1757812, -10077.7646484, 10021.0869141
1: -636.4227295, 813.5534058, -650.0700684, 833.0407104, -1469.4633789, 1463.6231689
2: -486.3359985, 1071.2492676, -497.9827576, 1098.9267578, -1585.2624512, 1569.2320557
3: -512.2059326, 1429.7664795, -525.0321655, 1465.9018555, -1978.1077881, 1954.7985840
4: -422.1175537, 1352.5257568, -432.3747253, 1388.1757812, -1810.2933350, 1784.9005127

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841614, upper bound: 9177.5848306
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5836463, upper bound: 9177.5847552
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3655.5886230, 6271.7954102, -4505.7573242, 7749.3828125, -11404.9716797, 10777.5527344
1: -636.4227295, 813.5534058, -786.4260254, 1001.5693970, -1637.9920654, 1599.9793701
2: -486.3359985, 1071.2492676, -599.9754028, 1320.8731689, -1807.2089844, 1671.2246094
3: -512.2059326, 1429.7664795, -631.8952026, 1764.6744385, -2276.8798828, 2061.6613770
4: -422.1175537, 1352.5257568, -521.0288696, 1668.3464355, -2090.4638672, 1873.5545654

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5841614, upper bound: 9177.5848306
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5849592, upper bound: 9177.5847552
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3749.2917480, 6422.1757812, -3749.2917480, 6422.1757812, -10171.4677734, 10171.4677734
1: -650.0700684, 833.0407104, -650.0700684, 833.0407104, -1483.1105957, 1483.1107178
2: -497.9827576, 1098.9267578, -497.9827576, 1098.9267578, -1596.9093018, 1596.9093018
3: -525.0321655, 1465.9018555, -525.0321655, 1465.9018555, -1990.9339600, 1990.9339600
4: -432.3747253, 1388.1757812, -432.3747253, 1388.1757812, -1820.5505371, 1820.5505371

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826635, upper bound: 9177.5845467
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862068, upper bound: 9177.5867697
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3749.2917480, 6422.1757812, -4505.7573242, 7749.3828125, -11498.6748047, 10927.9335938
1: -650.0700684, 833.0407104, -786.4260254, 1001.5693970, -1651.6392822, 1619.4666748
2: -497.9827576, 1098.9267578, -599.9754028, 1320.8731689, -1818.8558350, 1698.9020996
3: -525.0321655, 1465.9018555, -631.8952026, 1764.6744385, -2289.7058105, 2097.7971191
4: -432.3747253, 1388.1757812, -521.0288696, 1668.3464355, -2100.7211914, 1909.2045898

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826642, upper bound: 9177.5845467
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5862068, upper bound: 9177.5867702
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4369.2983398, 7527.1455078, -3556.5822754, 6116.1420898, -10485.4404297, 11083.7275391
1: -764.4841309, 972.7909546, -621.0281372, 792.7008667, -1557.1846924, 1593.8190918
2: -582.5541992, 1282.9584961, -473.7857056, 1045.2277832, -1627.7819824, 1756.7441406
3: -612.7703247, 1713.1683350, -498.5217896, 1393.5241699, -2006.2944336, 2211.6894531
4: -505.8053894, 1620.3533936, -411.1761780, 1319.8477783, -1825.6531982, 2031.5292969

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827200, upper bound: 9177.5840519
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827200, upper bound: 9177.5851037
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4369.2983398, 7527.1455078, -4389.9023438, 7575.9384766, -11945.2363281, 11917.0478516
1: -764.4841309, 972.7909546, -769.0201416, 981.9183350, -1746.4024658, 1741.8109131
2: -582.5541992, 1282.9584961, -586.9189453, 1297.3420410, -1879.8962402, 1869.8774414
3: -612.7703247, 1713.1683350, -615.6922607, 1723.6666260, -2336.4370117, 2328.8601074
4: -505.8053894, 1620.3533936, -509.2453918, 1637.1014404, -2142.9067383, 2129.5988770

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827199, upper bound: 9177.5840519
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5827200, upper bound: 9177.5851037
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5421.3022461, 9373.8183594, -3556.5822754, 6116.1420898, -11537.4443359, 12930.4003906
1: -950.5028076, 1212.4326172, -621.0281372, 792.7008667, -1743.2036133, 1833.4606934
2: -725.6915283, 1602.3067627, -473.7857056, 1045.2277832, -1770.9191895, 2076.0917969
3: -760.6499023, 2129.4760742, -498.5217896, 1393.5241699, -2154.1740723, 2627.9978027
4: -629.9783936, 2021.2227783, -411.1761780, 1319.8477783, -1949.8261719, 2432.3989258

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5826541
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5840332
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5421.3022461, 9373.8183594, -4389.9023438, 7575.9384766, -12997.2402344, 13763.7207031
1: -950.5028076, 1212.4326172, -769.0201416, 981.9183350, -1932.4211426, 1981.4525146
2: -725.6915283, 1602.3067627, -586.9189453, 1297.3420410, -2023.0333252, 2189.2255859
3: -760.6499023, 2129.4760742, -615.6922607, 1723.6666260, -2484.3164062, 2745.1684570
4: -629.9783936, 2021.2227783, -509.2453918, 1637.1014404, -2267.0793457, 2530.4680176

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5826540
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9177.5826541, upper bound: 9177.5840332
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4366.8369141, 7532.6772461, -3635.0761719, 6243.2392578, -10610.0751953, 11167.7539062
1: -766.1329956, 973.3587646, -632.5471802, 809.0769653, -1575.2099609, 1605.9058838
2: -583.3323975, 1282.0225830, -483.5997314, 1068.7369385, -1652.0693359, 1765.6220703
3: -612.9025269, 1712.8254395, -509.2717896, 1424.0888672, -2036.9914551, 2222.0971680
4: -506.3904419, 1618.6300049, -419.7204895, 1350.1463623, -1856.5367432, 2038.3503418

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.52 + 415.93 = 420.44 seconds
