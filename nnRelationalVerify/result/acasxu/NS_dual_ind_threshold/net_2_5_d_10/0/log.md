## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 3656.1557913764673


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1750.0124512, 3321.1867676, -1750.0124512, 3321.1867676, -5071.1992188, 5071.1992188)
1: (-591.4364624, 1249.1158447, -591.4364624, 1249.1158447, -1840.5522461, 1840.5522461)
2: (-302.9573364, 1257.5653076, -302.9573364, 1257.5653076, -1560.5225830, 1560.5225830)
3: (-690.8545532, 1531.0914307, -690.8545532, 1531.0914307, -2221.9460449, 2221.9460449)
4: (-393.2388916, 1293.9045410, -393.2388916, 1293.9045410, -1687.1434326, 1687.1434326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.73 + 2.13 = 4.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3656.1923533, upper bound: 3656.1923533

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1905663
time: 0.80 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1856194, upper bound: 3656.1856194
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.91 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1905663
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 0, lower bound: -3656.1856194, upper bound: 3656.1856194

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1580.9206543, 2982.5354004, -1631.7821045, 3097.4033203, -4678.3237305, 4614.3173828
1: -532.2230225, 1123.4039307, -551.4689941, 1164.8719482, -1697.0949707, 1674.8729248
2: -271.8260193, 1129.6441650, -281.9917297, 1172.4995117, -1444.3251953, 1411.6358643
3: -619.9239502, 1374.4746094, -643.1171265, 1427.0217285, -2046.9456787, 2017.5917969
4: -353.0870056, 1159.9466553, -366.1242371, 1204.8984375, -1557.9854736, 1526.0709229

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
time: 0.94 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892342
time: 0.90 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1651.6168213, 3135.9174805, -1686.8813477, 3202.0041504, -4853.6206055, 4822.7983398
1: -558.6760254, 1179.1304932, -570.4138794, 1204.1953125, -1762.8712158, 1749.5443115
2: -286.0073242, 1187.2133789, -292.0686951, 1212.3238525, -1498.3311768, 1479.2819824
3: -650.9782715, 1445.0185547, -665.1475220, 1475.7308350, -2126.7089844, 2110.1660156
4: -371.1179199, 1219.8610840, -379.0341492, 1246.0922852, -1617.2100830, 1598.8951416

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1328905, upper bound: 3656.1463442
time: 0.87 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1835579, upper bound: 3656.1835579
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.53 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.53
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.53
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892342
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 4.53
Output dim: 0, lower bound: -3656.1328905, upper bound: 3656.1463442
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.53
Output dim: 0, lower bound: -3656.1835579, upper bound: 3656.1835579

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1467.9260254, 2778.7185059, -1383.7115479, 2629.7697754, -4097.6958008, 4162.4296875
1: -494.9534912, 1045.7126465, -467.9188232, 989.3686523, -1484.3220215, 1513.6313477
2: -252.6454620, 1052.2827148, -239.2847900, 995.6418457, -1248.2873535, 1291.5672607
3: -576.5960693, 1280.2117920, -546.5562744, 1212.5380859, -1789.1341553, 1826.7680664
4: -328.1435547, 1079.8819580, -310.8585815, 1023.7635498, -1351.9071045, 1390.7404785

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
time: 0.76 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1571.4195557, 2964.6350098, -1613.1994629, 3062.5312500, -4633.9506836, 4577.8344727
1: -529.0820312, 1116.7486572, -545.3587646, 1151.9138184, -1680.9958496, 1662.1072998
2: -270.2056274, 1122.9215088, -278.8323669, 1159.4578857, -1429.6635742, 1401.7539062
3: -616.1264648, 1366.3358154, -635.7932129, 1411.1557617, -2027.2822266, 2002.1287842
4: -350.9929504, 1152.9600830, -362.0308228, 1191.3262939, -1542.3190918, 1514.9908447

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1617546, upper bound: 3656.1535309
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
time: 0.70 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1650.4305420, 3133.5646973, -1682.5670166, 3193.4055176, -4843.8344727, 4816.1313477
1: -558.2691040, 1178.2641602, -568.9160767, 1201.0330811, -1759.3021240, 1747.1801758
2: -285.7759399, 1186.3387451, -291.2162170, 1209.1437988, -1494.9196777, 1477.5549316
3: -650.4225464, 1443.9091797, -663.1286011, 1471.6832275, -2122.1057129, 2107.0378418
4: -370.8152466, 1218.8358154, -377.9198608, 1242.3779297, -1613.1929932, 1596.7556152

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834954, upper bound: 3656.1835579
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1835104, upper bound: 3656.1835104
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.40 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 0, lower bound: -3656.1617546, upper bound: 3656.1535309
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 0, lower bound: -3656.1834954, upper bound: 3656.1835579
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.40
Output dim: 0, lower bound: -3656.1835104, upper bound: 3656.1835104

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1396.6440430, 2642.3776855, -1356.0345459, 2579.4299316, -3976.0734863, 3998.4121094
1: -470.4030151, 993.7450562, -458.6696167, 969.6292114, -1440.0322266, 1452.4144287
2: -240.4356842, 1000.1814575, -234.5975952, 976.0795288, -1216.5151367, 1234.7784424
3: -548.8563232, 1216.7685547, -535.6669922, 1188.4787598, -1737.3350830, 1752.4355469
4: -312.1244507, 1027.7401123, -304.6391907, 1003.9230957, -1316.0476074, 1332.3792725

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1458.5491943, 2760.7680664, -1375.3345947, 2614.1193848, -4072.6684570, 4136.1025391
1: -491.8841553, 1039.2145996, -465.1409912, 983.6267700, -1475.5107422, 1504.3553467
2: -251.0446014, 1045.7264404, -237.8849945, 989.9069824, -1240.9512939, 1283.6113281
3: -572.9407959, 1272.2065430, -543.3688965, 1205.5059814, -1778.4466553, 1815.5754395
4: -326.0745850, 1073.0727539, -309.0349731, 1017.8513184, -1343.9259033, 1382.1076660

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
time: 1.35 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1501.5017090, 2831.1857910, -1578.9356689, 2993.4296875, -4494.9311523, 4410.1215820
1: -505.0660095, 1065.9953613, -533.4063110, 1126.5777588, -1631.6437988, 1599.4016113
2: -257.7775574, 1071.4672852, -272.7944946, 1133.5976562, -1391.3750000, 1344.2617188
3: -587.3356323, 1303.4781494, -621.7434692, 1379.9888916, -1967.3244629, 1925.2214355
4: -334.8744812, 1099.6273193, -354.2642517, 1164.6324463, -1499.5067139, 1453.8916016

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1617553, upper bound: 3656.1535309
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1617553, upper bound: 3656.1535309
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1565.5180664, 2953.4284668, -1611.2095947, 3058.7326660, -4624.2509766, 4564.6376953
1: -527.1376953, 1112.7050781, -544.7054443, 1150.5482178, -1677.6859131, 1657.4104004
2: -269.1836853, 1118.7882080, -278.4887390, 1158.0614014, -1427.2449951, 1397.2769775
3: -613.8220215, 1361.3262939, -635.0020752, 1409.4676514, -2023.2896729, 1996.3283691
4: -349.6910706, 1148.6240234, -361.5922241, 1189.8293457, -1539.5203857, 1510.2161865

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1398.9925537, 2659.0437012, -1567.8406982, 2988.1779785, -4387.1704102, 4226.8842773
1: -473.6305237, 1000.6710205, -531.5817261, 1122.7514648, -1596.3819580, 1532.2526855
2: -242.4832306, 1007.3726807, -271.6761169, 1130.8081055, -1373.2913818, 1279.0487061
3: -552.5869751, 1226.8209229, -618.9943237, 1376.0316162, -1928.6184082, 1845.8151855
4: -314.8037109, 1035.2404785, -352.5320435, 1161.0202637, -1475.8239746, 1387.7724609

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1813023
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1631.5279541, 3098.4653320, -1670.9743652, 3171.8217773, -4803.3496094, 4769.4394531
1: -552.0596313, 1165.1343994, -565.1031494, 1192.9719238, -1745.0314941, 1730.2375488
2: -282.5646057, 1173.1888428, -289.2411804, 1201.0563965, -1483.6209717, 1462.4299316
3: -642.9915161, 1427.8371582, -658.5553589, 1461.8055420, -2104.7966309, 2086.3925781
4: -366.6563416, 1205.1334229, -375.3639221, 1233.9422607, -1600.5983887, 1580.4973145

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1810074, upper bound: 3656.1807186
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1815520, upper bound: 3656.1815520
time: 0.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.78 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882940
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1617553, upper bound: 3656.1535309
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1617553, upper bound: 3656.1535309
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1813023
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1810074, upper bound: 3656.1807186
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.78
Output dim: 0, lower bound: -3656.1815520, upper bound: 3656.1815520

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1396.6440430, 2642.3776855, -1303.9255371, 2459.5505371, -3856.1938477, 3946.3032227
1: -470.4030151, 993.7450562, -438.7599182, 927.0606689, -1397.4636230, 1432.5050049
2: -240.4356842, 1000.1814575, -224.1307526, 932.3007202, -1172.7360840, 1224.3120117
3: -548.8563232, 1216.7685547, -511.9994202, 1135.0255127, -1683.8818359, 1728.7679443
4: -312.1244507, 1027.7401123, -291.1955872, 958.0763550, -1270.2008057, 1318.9356689

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830325
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1396.6440430, 2642.3776855, -1369.0336914, 2604.4516602, -4001.0954590, 4011.4113770
1: -470.4030151, 993.7450562, -463.5313110, 979.1362915, -1449.5393066, 1457.2761230
2: -240.4356842, 1000.1814575, -237.3527222, 986.0717163, -1226.5074463, 1237.5339355
3: -548.8563232, 1216.7685547, -540.8361816, 1200.6480713, -1749.5042725, 1757.6047363
4: -312.1244507, 1027.7401123, -308.0209961, 1013.6055908, -1325.7299805, 1335.7611084

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830325
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1458.5491943, 2760.7680664, -1334.9085693, 2516.3486328, -3974.8974609, 4095.6765137
1: -491.8841553, 1039.2145996, -449.2790833, 949.2331543, -1441.1173096, 1488.4935303
2: -251.0446014, 1045.7264404, -229.3490906, 954.2366943, -1205.2810059, 1275.0754395
3: -572.9407959, 1272.2065430, -524.0039062, 1161.7487793, -1734.6894531, 1796.2103271
4: -326.0745850, 1073.0727539, -298.0812378, 980.3115845, -1306.3862305, 1371.1540527

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882876
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1858894, upper bound: 3656.1872545
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1458.5491943, 2760.7680664, -1395.6121826, 2652.3808594, -4110.9296875, 4156.3798828
1: -491.8841553, 1039.2145996, -472.5339355, 998.4816895, -1490.3658447, 1511.7485352
2: -251.0446014, 1045.7264404, -241.9597473, 1005.1511230, -1256.1956787, 1287.6861572
3: -572.9407959, 1272.2065430, -551.4211426, 1224.1636963, -1797.1044922, 1823.6276855
4: -326.0745850, 1073.0727539, -314.1398621, 1033.0375977, -1359.1121826, 1387.2126465

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882876
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1858894, upper bound: 3656.1872545
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1501.5017090, 2831.1857910, -1533.7362061, 2888.7963867, -4390.2978516, 4364.9213867
1: -505.0660095, 1065.9953613, -516.1625366, 1089.3739014, -1594.4399414, 1582.1579590
2: -257.7775574, 1071.4672852, -263.5389404, 1095.1053467, -1352.8826904, 1335.0062256
3: -587.3356323, 1303.4781494, -600.6391602, 1332.3388672, -1919.6745605, 1904.1169434
4: -334.8744812, 1099.6273193, -342.3923950, 1124.1097412, -1458.9841309, 1442.0197754

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1551004, upper bound: 3656.1399276
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1573110, upper bound: 3656.1521046
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1501.5017090, 2831.1857910, -1595.3482666, 3025.9265137, -4527.4282227, 4426.5341797
1: -505.0660095, 1065.9953613, -539.7172852, 1138.4576416, -1643.5236816, 1605.7124023
2: -257.7775574, 1071.4672852, -276.2493896, 1146.0725098, -1403.8499756, 1347.7163086
3: -587.3356323, 1303.4781494, -628.2398071, 1395.0151367, -1982.3508301, 1931.7177734
4: -334.8744812, 1099.6273193, -358.4928894, 1177.3447266, -1512.2192383, 1458.1202393

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1551004, upper bound: 3656.1399276
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1573110, upper bound: 3656.1521046
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1565.5180664, 2953.4284668, -1564.1149902, 2950.7941895, -4516.3125000, 4517.5424805
1: -527.1376953, 1112.7050781, -526.6522827, 1111.6027832, -1638.7404785, 1639.3572998
2: -269.1836853, 1118.7882080, -268.9443970, 1117.7299805, -1386.9133301, 1387.7325439
3: -613.8220215, 1361.3262939, -613.2048340, 1360.0172119, -1973.8392334, 1974.5310059
4: -349.6910706, 1148.6240234, -349.3647461, 1147.5585938, -1497.2495117, 1497.9887695

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1881623
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1565.5180664, 2953.4284668, -1627.6057129, 3091.3364258, -4656.8544922, 4581.0341797
1: -527.1376953, 1112.7050781, -550.8072510, 1162.4875488, -1689.6252441, 1663.5122070
2: -269.1836853, 1118.7882080, -281.9015808, 1170.5052490, -1439.6885986, 1400.6898193
3: -613.8220215, 1361.3262939, -641.4719238, 1424.6007080, -2038.4227295, 2002.7982178
4: -349.6910706, 1148.6240234, -365.8082275, 1202.3164062, -1552.0073242, 1514.4322510

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1881623
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1351.2231445, 2571.3488770, -1493.7373047, 2853.5207520, -4204.7441406, 4065.0861816
1: -457.9662170, 967.4046631, -507.6636047, 1071.8144531, -1529.7806396, 1475.0682373
2: -234.2206726, 973.6732788, -258.8596802, 1079.1735840, -1313.3942871, 1232.5329590
3: -533.8497314, 1186.0107422, -589.7190552, 1313.2364502, -1847.0861816, 1775.7297363
4: -304.1994629, 999.5770874, -336.0385742, 1106.0561523, -1410.2556152, 1335.6156006

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1803842, upper bound: 3656.1800413
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1375.7465820, 2616.7416992, -1525.6141357, 2911.1472168, -4286.8930664, 4142.3559570
1: -466.0960083, 984.6516113, -517.7940063, 1093.3422852, -1559.4382324, 1502.4454346
2: -238.5605164, 991.4895630, -264.5579834, 1101.5556641, -1340.1162109, 1256.0471191
3: -543.5600586, 1207.2281494, -602.7034302, 1340.1770020, -1883.7370605, 1809.9316406
4: -309.7062988, 1018.6324463, -343.2530823, 1130.9416504, -1440.6479492, 1361.8854980

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1802688, upper bound: 3656.1769294
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1813023
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1583.4553223, 3011.2829590, -1592.6590576, 3029.2629395, -4612.7182617, 4603.9418945
1: -536.4425659, 1132.2745361, -539.5967407, 1139.0300293, -1675.4725342, 1671.8713379
2: -274.2811890, 1139.8764648, -275.6536865, 1146.5501709, -1420.8312988, 1415.5301514
3: -624.3874512, 1387.4620361, -627.7530518, 1395.4862061, -2019.8736572, 2015.2150879
4: -356.0687866, 1169.9016113, -357.9206543, 1176.2619629, -1532.3308105, 1527.8222656

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1037891, upper bound: 3656.1215763
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1037891, upper bound: 3656.1806922
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1605.0020752, 3050.2014160, -1626.8863525, 3091.2395020, -4696.2416992, 4677.0878906
1: -543.4299927, 1146.6519775, -550.6613159, 1162.1087646, -1705.5385742, 1697.3132324
2: -278.0640564, 1154.7448730, -281.7601013, 1170.2886963, -1448.3525391, 1436.5048828
3: -632.6001587, 1405.1821289, -641.4290771, 1424.0783691, -2056.6782227, 2046.6110840
4: -360.7867737, 1186.0296631, -365.6141663, 1202.4516602, -1563.2384033, 1551.6437988

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1211262, upper bound: 3656.1268684
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1211268, upper bound: 3656.1814611
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.30 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830325
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1865137
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830325
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882876
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1858894, upper bound: 3656.1872545
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1882876
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1858894, upper bound: 3656.1872545
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1551004, upper bound: 3656.1399276
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1573110, upper bound: 3656.1521046
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1551004, upper bound: 3656.1399276
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1573110, upper bound: 3656.1521046
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1881623
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1853315, upper bound: 3656.1881623
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1859767, upper bound: 3656.1892306
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1803842, upper bound: 3656.1800413
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1802688, upper bound: 3656.1769294
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1813023
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1037891, upper bound: 3656.1215763
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1037891, upper bound: 3656.1806922
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1211262, upper bound: 3656.1268684
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.30
Output dim: 0, lower bound: -3656.1211268, upper bound: 3656.1814611

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1353.6424561, 2562.9030762, -1283.4907227, 2422.7265625, -3776.3691406, 3846.3937988
1: -456.1769714, 964.5028687, -432.0809937, 913.1892700, -1369.3660889, 1396.5838623
2: -233.2864532, 970.4293823, -220.7514038, 918.3516235, -1151.6378174, 1191.1805420
3: -532.7540894, 1181.0201416, -504.3879395, 1118.2471924, -1651.0012207, 1685.4080811
4: -302.9057922, 997.2111206, -286.8348389, 943.7222290, -1246.6278076, 1284.0458984

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865481, upper bound: 3656.1872828
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1869181, upper bound: 3656.1881571
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1383.1528320, 2617.1611328, -1297.9973145, 2448.6118164, -3831.7646484, 3915.1584473
1: -465.9932861, 984.5905151, -436.8308716, 923.0695190, -1389.0627441, 1421.4212646
2: -238.1092834, 990.8218994, -223.1155701, 928.2371216, -1166.3464355, 1213.9375000
3: -543.5665283, 1205.3801270, -509.7022400, 1130.0762939, -1673.6427002, 1715.0823975
4: -309.1487732, 1017.8850708, -289.8964233, 953.8122559, -1262.9610596, 1307.7814941

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1862324, upper bound: 3656.1829470
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865945, upper bound: 3656.1830821
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1353.6424561, 2562.9030762, -1347.5605469, 2566.2539062, -3919.8964844, 3910.4633789
1: -456.1769714, 964.5028687, -456.5956726, 964.8018799, -1420.9788818, 1421.0985107
2: -233.2864532, 970.4293823, -233.7747955, 971.6654663, -1204.9519043, 1204.2039795
3: -532.7540894, 1181.0201416, -532.7736206, 1183.0798340, -1715.8339844, 1713.7934570
4: -302.9057922, 997.2111206, -303.4017334, 998.6290894, -1301.5349121, 1300.6127930

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1844113, upper bound: 3656.1843826
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1849457, upper bound: 3656.1864996
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1383.1528320, 2617.1611328, -1363.9157715, 2595.0476074, -3978.2004395, 3981.0764160
1: -465.9932861, 984.5905151, -461.8927917, 975.6931152, -1441.6864014, 1446.4832764
2: -238.1092834, 990.8218994, -236.4966278, 982.5590820, -1220.6683350, 1227.3184814
3: -543.5665283, 1205.3801270, -538.8803711, 1196.4099121, -1739.9764404, 1744.2603760
4: -309.1487732, 1017.8850708, -306.9211121, 1009.9287109, -1319.0775146, 1324.8061523

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1847650, upper bound: 3656.1828436
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1842312, upper bound: 3656.1826844
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1413.9539795, 2678.1398926, -1313.6009521, 2477.8652344, -3891.8190918, 3991.7407227
1: -477.0383301, 1008.8669434, -442.3482971, 934.8195190, -1411.8579102, 1451.2152100
2: -243.6025391, 1014.8198242, -225.8375092, 939.7601318, -1183.3626709, 1240.6573486
3: -556.1818237, 1235.0668945, -516.0974121, 1144.3383789, -1700.5200195, 1751.1643066
4: -316.4887390, 1041.3067627, -293.5532837, 965.4095459, -1281.8978271, 1334.8599854

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1871700, upper bound: 3656.1886417
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1875230, upper bound: 3656.1890957
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1446.2880859, 2738.0852051, -1329.6406250, 2506.5869141, -3952.8750000, 4067.7258301
1: -487.9005737, 1030.9630127, -447.5688171, 945.6953735, -1433.5959473, 1478.5316162
2: -248.9490814, 1037.2941895, -228.4419403, 950.6272583, -1199.5760498, 1265.7359619
3: -568.1857910, 1261.9468994, -521.9493408, 1157.3455811, -1725.5312500, 1783.8962402
4: -323.3947449, 1064.1920166, -296.9209290, 976.5051270, -1299.8999023, 1361.1129150

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1871398, upper bound: 3656.1872338
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1874936, upper bound: 3656.1873361
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1413.9539795, 2678.1398926, -1373.6166992, 2613.4877930, -4027.4416504, 4051.7565918
1: -477.0383301, 1008.8669434, -465.4027405, 983.7652588, -1460.8035889, 1474.2696533
2: -243.6025391, 1014.8198242, -238.3036957, 990.3956909, -1233.9981689, 1253.1234131
3: -556.1818237, 1235.0668945, -543.2139893, 1206.2106934, -1762.3923340, 1778.2808838
4: -316.4887390, 1041.3067627, -309.4267273, 1017.7080688, -1334.1966553, 1350.7333984

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1858009, upper bound: 3656.1880688
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1851758, upper bound: 3656.1879534
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1446.2880859, 2738.0852051, -1390.7651367, 2643.4047852, -4089.6928711, 4128.8505859
1: -487.9005737, 1030.9630127, -470.9761047, 995.1958618, -1483.0964355, 1501.9390869
2: -248.9490814, 1037.2941895, -241.1448669, 1001.8187866, -1250.7677002, 1278.4390869
3: -568.1857910, 1261.9468994, -549.5578613, 1220.1203613, -1788.3060303, 1811.5047607
4: -323.3947449, 1064.1920166, -313.0922241, 1029.5373535, -1352.9321289, 1377.2841797

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1857615, upper bound: 3656.1871326
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1851440, upper bound: 3656.1869603
time: 2.95 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1485.3950195, 2804.0144043, -1519.8488770, 2863.5207520, -4348.9155273, 4323.8632812
1: -500.3392944, 1056.2000732, -511.6992798, 1079.9851074, -1580.3240967, 1567.8994141
2: -255.4194336, 1061.6469727, -261.2549744, 1085.6528320, -1341.0718994, 1322.9016113
3: -581.8908081, 1291.9848633, -595.3821411, 1321.0350342, -1902.9257812, 1887.3669434
4: -331.8793335, 1089.1824951, -339.4638977, 1114.1856689, -1446.0646973, 1428.6463623

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1659748, upper bound: 3656.1558573
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1659748, upper bound: 3656.1558511
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1485.3950195, 2804.0144043, -1581.4129639, 3000.9858398, -4486.3803711, 4385.4272461
1: -500.3392944, 1056.2000732, -535.2450562, 1129.0754395, -1629.4144287, 1591.4450684
2: -255.4194336, 1061.6469727, -273.9298706, 1136.7139893, -1392.1330566, 1335.5765381
3: -581.8908081, 1291.9848633, -622.8640747, 1383.5880127, -1965.4787598, 1914.8488770
4: -331.8793335, 1089.1824951, -355.4888306, 1167.4416504, -1499.3209229, 1444.6712646

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1469601, upper bound: 3656.1499040
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1354497, upper bound: 3656.1325017
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1401591, upper bound: 3656.1340408
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1499.0382080, 2825.9675293, -1529.9364014, 2888.4375000, -4387.4746094, 4355.9038086
1: -504.0126648, 1064.1436768, -515.1243896, 1087.3414307, -1591.3541260, 1579.2678223
2: -257.7232361, 1070.3291016, -263.1750488, 1093.7302246, -1351.4532471, 1333.5040283
3: -587.7527466, 1302.2279053, -599.9333496, 1330.5972900, -1918.3500977, 1902.1612549
4: -334.6189270, 1100.3907471, -341.7505493, 1123.0081787, -1457.6268311, 1442.1413574

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1877392, upper bound: 3656.1834215
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865966, upper bound: 3656.1833526
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1558.0628662, 2938.7141113, -1560.0675049, 2942.8010254, -4500.8632812, 4498.7817383
1: -524.6494141, 1107.4379883, -525.3020630, 1108.7470703, -1633.3963623, 1632.7399902
2: -267.9169312, 1113.4915771, -268.2563171, 1114.8536377, -1382.7705078, 1381.7478027
3: -610.9157715, 1354.8485107, -611.6276855, 1356.5026855, -1967.4184570, 1966.4761963
4: -348.0439453, 1143.2000732, -348.4702148, 1144.6151123, -1492.6590576, 1491.6702881

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1897228, upper bound: 3656.1876233
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1875683, upper bound: 3656.1875683
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1499.0382080, 2825.9675293, -1597.8852539, 3036.6901855, -4535.7280273, 4423.8525391
1: -504.0126648, 1064.1436768, -540.7952881, 1141.2825928, -1645.2952881, 1604.9389648
2: -257.7232361, 1070.3291016, -276.7867432, 1149.4813232, -1407.2044678, 1347.1158447
3: -587.7527466, 1302.2279053, -629.7171631, 1398.7666016, -1986.5192871, 1931.9450684
4: -334.6189270, 1100.3907471, -359.0502014, 1180.8200684, -1515.4388428, 1459.4409180

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1840056, upper bound: 3656.1881260
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1780661, upper bound: 3656.1811315
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1835842, upper bound: 3656.1826790
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830793
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1558.0628662, 2938.7141113, -1620.5898438, 3078.2421875, -4636.3046875, 4559.3037109
1: -524.6494141, 1107.4379883, -548.5305176, 1157.6600342, -1682.3094482, 1655.9685059
2: -267.9169312, 1113.4915771, -280.7391663, 1165.6950684, -1433.6119385, 1394.2304688
3: -610.9157715, 1354.8485107, -638.8238525, 1418.6977539, -2029.6135254, 1993.6723633
4: -348.0439453, 1143.2000732, -364.2924500, 1197.4006348, -1545.4445801, 1507.4925537

Time for backsubstitution: 3.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1840620, upper bound: 3656.1888932
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1785345, upper bound: 3656.1817289
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1549548, upper bound: 3656.1688094
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1295.7105713, 2463.7824707, -1464.8946533, 2801.5659180, -4097.2763672, 3928.6772461
1: -438.3490601, 925.9326782, -498.1364136, 1051.5172119, -1489.8662109, 1424.0690918
2: -224.5829620, 932.1920166, -253.9821014, 1058.9133301, -1283.4962158, 1186.1739502
3: -511.9200439, 1135.5030518, -578.4245605, 1288.4840088, -1800.4040527, 1713.9274902
4: -291.4964600, 958.9721069, -329.5774841, 1085.5007324, -1376.9971924, 1288.5494385

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1803842, upper bound: 3656.1800413
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1803842, upper bound: 3656.1800413
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1342.1259766, 2553.6499023, -1486.0560303, 2839.0578613, -4181.1835938, 4039.7060547
1: -454.9230957, 961.2343750, -505.0692139, 1066.4263916, -1521.3494873, 1466.3034668
2: -232.7212372, 967.4543457, -257.5642700, 1073.8118896, -1306.5330811, 1225.0185547
3: -530.4423828, 1178.4503174, -586.7623901, 1306.6372070, -1837.0795898, 1765.2126465
4: -302.2681885, 993.1567383, -334.3492737, 1100.5614014, -1402.8295898, 1327.5058594

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1329.3225098, 2531.3205566, -1505.8327637, 2876.2983398, -4205.6210938, 4037.1533203
1: -450.8377991, 953.1029053, -511.4398193, 1080.1156006, -1530.9533691, 1464.5426025
2: -230.8460236, 959.5006714, -261.2738953, 1088.2811279, -1319.1271973, 1220.7745361
3: -526.2179565, 1168.6235352, -595.2888184, 1324.0393066, -1850.2573242, 1763.9123535
4: -299.7994385, 985.6746216, -338.9918823, 1117.2391357, -1417.0385742, 1324.6665039

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1802688, upper bound: 3656.1769291
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1802688, upper bound: 3656.1769291
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1363.8975830, 2594.8242188, -1518.8266602, 2898.5886230, -4262.4858398, 4113.6499023
1: -462.2783203, 976.5924072, -515.5987549, 1088.7395020, -1551.0178223, 1492.1910400
2: -236.5741730, 983.2650146, -263.4128723, 1096.8554688, -1333.4295654, 1246.6776123
3: -538.9760742, 1197.2913818, -600.0904541, 1334.5039062, -1873.4799805, 1797.3818359
4: -307.1475830, 1010.0280151, -341.7787476, 1126.0428467, -1433.1904297, 1351.8066406

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1812891
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1813023
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1579.2250977, 3002.9047852, -1592.6590576, 3029.2629395, -4608.4882812, 4595.5625000
1: -535.0154419, 1129.2259521, -539.5967407, 1139.0300293, -1674.0454102, 1668.8227539
2: -273.4765625, 1136.7916260, -275.6536865, 1146.5501709, -1420.0267334, 1412.4453125
3: -622.4585571, 1383.5877686, -627.7530518, 1395.4862061, -2017.9448242, 2011.3408203
4: -355.0197449, 1166.3265381, -357.9206543, 1176.2619629, -1531.2817383, 1524.2471924

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.0872667, upper bound: 3656.1409002
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1025527, upper bound: 3656.1806922
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1601.0137939, 3042.3544922, -1626.8863525, 3091.2395020, -4692.2529297, 4669.2407227
1: -542.0805664, 1143.7875977, -550.6613159, 1162.1087646, -1704.1893311, 1694.4489746
2: -277.3031921, 1151.8432617, -281.7601013, 1170.2886963, -1447.5917969, 1433.6033936
3: -630.7808838, 1401.5328369, -641.4290771, 1424.0783691, -2054.8593750, 2042.9617920
4: -359.7942505, 1182.6766357, -365.6141663, 1202.4516602, -1562.2458496, 1548.2907715

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1132265, upper bound: 3656.1554391
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1099539, upper bound: 3656.1814611
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.18 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1865481, upper bound: 3656.1872828
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1869181, upper bound: 3656.1881571
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1862324, upper bound: 3656.1829470
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1865945, upper bound: 3656.1830821
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1844113, upper bound: 3656.1843826
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1849457, upper bound: 3656.1864996
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1847650, upper bound: 3656.1828436
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1842312, upper bound: 3656.1826844
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1871700, upper bound: 3656.1886417
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1875230, upper bound: 3656.1890957
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1871398, upper bound: 3656.1872338
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1874936, upper bound: 3656.1873361
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1858009, upper bound: 3656.1880688
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1851758, upper bound: 3656.1879534
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1857615, upper bound: 3656.1871326
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1851440, upper bound: 3656.1869603
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1659748, upper bound: 3656.1558573
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1659748, upper bound: 3656.1558511
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1354497, upper bound: 3656.1325017
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1401591, upper bound: 3656.1340408
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1877392, upper bound: 3656.1834215
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1865966, upper bound: 3656.1833526
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1897228, upper bound: 3656.1876233
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1875683, upper bound: 3656.1875683
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1835842, upper bound: 3656.1826790
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830793
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1785345, upper bound: 3656.1817289
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1549548, upper bound: 3656.1688094
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1803842, upper bound: 3656.1800413
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1803842, upper bound: 3656.1800413
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1809374, upper bound: 3656.1805431
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1802688, upper bound: 3656.1769291
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1802688, upper bound: 3656.1769291
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1812891
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1815023, upper bound: 3656.1813023
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.0872667, upper bound: 3656.1409002
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1025527, upper bound: 3656.1806922
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1132265, upper bound: 3656.1554391
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.18
Output dim: 0, lower bound: -3656.1099539, upper bound: 3656.1814611

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1325.6230469, 2510.4050293, -1235.1685791, 2330.0490723, -3655.6721191, 3745.5732422
1: -446.7766724, 944.5640259, -415.5644836, 878.1232910, -1324.8999023, 1360.1285400
2: -228.3867798, 950.3493652, -212.3418274, 883.0781860, -1111.4647217, 1162.6910400
3: -521.5376587, 1156.4395752, -485.4468689, 1075.3361816, -1596.8737793, 1641.8862305
4: -296.5385437, 976.3591309, -275.9159241, 907.8615112, -1204.4000244, 1252.2750244

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865481, upper bound: 3656.1872828
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865481, upper bound: 3656.1872828
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1335.4476318, 2528.4350586, -1254.7464600, 2368.3203125, -3703.7680664, 3783.1811523
1: -449.9171448, 951.0352783, -422.1758118, 891.8804321, -1341.7976074, 1373.2108154
2: -230.1273041, 957.0216064, -215.7556152, 897.1577148, -1127.2850342, 1172.7769775
3: -525.4255981, 1164.5689697, -492.8439636, 1092.3033447, -1617.7290039, 1657.4125977
4: -298.7251587, 983.6530762, -280.2292175, 922.3198853, -1221.0450439, 1263.8823242

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1869181, upper bound: 3656.1881571
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1869181, upper bound: 3656.1881571
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1356.7153320, 2567.4589844, -1250.8713379, 2358.5048828, -3715.2202148, 3818.3303223
1: -457.1240234, 965.7851562, -420.7810059, 888.9437866, -1346.0677490, 1386.5661621
2: -233.4746552, 971.8585815, -214.9343414, 893.9216919, -1127.3963623, 1186.7929688
3: -532.9546509, 1182.1893311, -491.2863159, 1088.3247070, -1621.2792969, 1673.4754639
4: -303.1277161, 998.1339722, -279.2815857, 918.9038086, -1222.0314941, 1277.4155273

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1862324, upper bound: 3656.1829470
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1862324, upper bound: 3656.1829470
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1363.5289307, 2580.2490234, -1267.5883789, 2391.1596680, -3754.6884766, 3847.8374023
1: -459.2990417, 970.2014771, -426.3895874, 900.5949097, -1359.8939209, 1396.5910645
2: -234.7343903, 976.4674072, -217.8480682, 905.8552246, -1140.5895996, 1194.3150635
3: -535.7357178, 1187.8229980, -497.5170898, 1102.7302246, -1638.4659424, 1685.3400879
4: -304.6951904, 1003.3701782, -282.9464417, 931.1896362, -1235.8847656, 1286.3165283

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865945, upper bound: 3656.1830821
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865945, upper bound: 3656.1830821
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1306.5789795, 2473.2402344, -1302.5351562, 2475.8312988, -3782.4099121, 3775.7751465
1: -440.5320740, 932.0986328, -440.2254028, 930.1682739, -1370.7003174, 1372.3239746
2: -225.2024536, 937.1662598, -224.9269409, 936.6217651, -1161.8239746, 1162.0932617
3: -514.0364380, 1140.8858643, -513.3825073, 1140.0596924, -1654.0961914, 1654.2683105
4: -292.5843506, 962.0521851, -291.9719543, 962.1922607, -1254.7766113, 1254.0239258

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1844113, upper bound: 3656.1843826
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1844113, upper bound: 3656.1843826
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1341.0511475, 2539.8586426, -1331.9268799, 2536.2648926, -3877.3159180, 3871.7856445
1: -452.0831604, 955.6822510, -451.4072266, 953.8315430, -1405.9146729, 1407.0893555
2: -231.1727600, 961.6716309, -231.0740509, 960.5699463, -1191.7426758, 1192.7457275
3: -527.9740601, 1170.2752686, -526.5181885, 1169.5814209, -1697.5552979, 1696.7933350
4: -300.1654663, 988.2910767, -299.9168396, 986.9490356, -1287.1142578, 1288.2077637

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1849457, upper bound: 3656.1864996
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1849457, upper bound: 3656.1864996
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1356.7153320, 2567.4589844, -1317.8652344, 2508.1315918, -3864.8469238, 3885.3239746
1: -457.1240234, 965.7851562, -446.3092346, 942.4134521, -1399.5373535, 1412.0943604
2: -233.4746552, 971.8585815, -228.5021057, 949.1160278, -1182.5904541, 1200.3607178
3: -532.9546509, 1182.1893311, -520.6604004, 1155.6866455, -1688.6411133, 1702.8497314
4: -303.1277161, 998.1339722, -296.4781494, 975.9054565, -1279.0332031, 1294.6119385

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1847650, upper bound: 3656.1828436
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1847650, upper bound: 3656.1828436
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1363.5289307, 2580.2490234, -1327.5512695, 2526.8830566, -3890.4121094, 3907.8002930
1: -459.2990417, 970.2014771, -449.4305420, 948.8704224, -1408.1694336, 1419.6319580
2: -234.7343903, 976.4674072, -230.1774139, 956.1408081, -1190.8751221, 1206.6445312
3: -535.7357178, 1187.8229980, -524.2747803, 1163.9663086, -1699.7020264, 1712.0977783
4: -304.6951904, 1003.3701782, -298.5609131, 983.0027466, -1287.6977539, 1301.9311523

Time for backsubstitution: 3.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1842312, upper bound: 3656.1826844
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1842312, upper bound: 3656.1826844
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1386.2547607, 2626.3151855, -1267.2441406, 2388.7058105, -3774.9604492, 3893.5585938
1: -467.7131348, 989.1704712, -426.4720764, 901.1132202, -1368.8261719, 1415.6425781
2: -238.7669373, 994.9799194, -217.7447052, 905.8598022, -1144.6265869, 1212.7246094
3: -545.1036987, 1210.7886963, -497.8477783, 1103.0538330, -1648.1574707, 1708.6363525
4: -310.2019043, 1020.7278442, -283.0437317, 930.9039307, -1241.1058350, 1303.7716064

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1870880, upper bound: 3656.1886417
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1870880, upper bound: 3656.1886417
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1395.2940674, 2642.8708496, -1284.9790039, 2423.6340332, -3818.9282227, 3927.8498535
1: -470.6188049, 995.0403442, -432.4510498, 913.5108643, -1384.1296387, 1427.4914551
2: -240.3590240, 1001.0599976, -220.8551025, 918.5835571, -1158.9426270, 1221.9150391
3: -548.6604614, 1218.1768799, -504.5878601, 1118.4130859, -1667.0734863, 1722.7647705
4: -312.1944580, 1027.3912354, -286.9612732, 944.0551147, -1256.2495117, 1314.3525391

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1874454, upper bound: 3656.1890957
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1874454, upper bound: 3656.1890957
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1419.8703613, 2688.4226074, -1283.7423096, 2418.6523438, -3838.5227051, 3972.1650391
1: -478.9691772, 1012.1729126, -431.8941345, 912.3954468, -1391.3646240, 1444.0670166
2: -244.3202515, 1018.3446045, -220.4451752, 917.1290894, -1161.4490967, 1238.7897949
3: -557.5859985, 1238.7812500, -503.9074707, 1116.5544434, -1674.1401367, 1742.6887207
4: -317.3788757, 1044.4730225, -286.5440063, 942.3549805, -1259.7338867, 1331.0169678

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1870880, upper bound: 3656.1872338
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1870880, upper bound: 3656.1872338
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1427.1035156, 2702.1921387, -1299.9366455, 2450.3632812, -3877.4663086, 4002.1286621
1: -481.3479309, 1016.8422241, -437.3304443, 923.6184692, -1404.9660645, 1454.1726074
2: -245.6554260, 1023.2481689, -223.2924347, 928.6812744, -1174.3366699, 1246.5404053
3: -560.5891113, 1244.7539062, -510.0696411, 1130.5253906, -1691.1145020, 1754.8234863
4: -319.0415955, 1050.0837402, -290.1178894, 954.4204102, -1273.4620361, 1340.2016602

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1874454, upper bound: 3656.1873361
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1874454, upper bound: 3656.1873361
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1386.2547607, 2626.3151855, -1329.2075195, 2528.7766113, -3915.0312500, 3955.5222168
1: -467.7131348, 989.1704712, -450.1807556, 951.2149658, -1418.9279785, 1439.3510742
2: -238.7669373, 994.9799194, -230.5018463, 957.6260986, -1196.3929443, 1225.4815674
3: -545.1036987, 1210.7886963, -525.4827881, 1166.3903809, -1711.4936523, 1736.2714844
4: -310.2019043, 1020.7278442, -299.2281189, 984.5350342, -1294.7369385, 1319.9559326

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1856943, upper bound: 3656.1880688
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1856943, upper bound: 3656.1880688
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1395.2940674, 2642.8708496, -1339.4595947, 2549.4177246, -3944.7119141, 3982.3303223
1: -470.6188049, 995.0403442, -453.6669312, 958.4093628, -1429.0281982, 1448.7072754
2: -240.3590240, 1001.0599976, -232.3518524, 965.4949951, -1205.8540039, 1233.4116211
3: -548.6604614, 1218.1768799, -529.4299927, 1175.5473633, -1724.2077637, 1747.6069336
4: -312.1944580, 1027.3912354, -301.5397034, 992.3280029, -1304.5222168, 1328.9309082

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1851103, upper bound: 3656.1879534
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1851103, upper bound: 3656.1879534
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1419.8703613, 2688.4226074, -1346.0386963, 2558.2797852, -3978.1501465, 4034.4604492
1: -478.9691772, 1012.1729126, -455.6561890, 962.4335938, -1441.4028320, 1467.8288574
2: -244.3202515, 1018.3446045, -233.3058624, 968.8889771, -1213.2092285, 1251.6505127
3: -557.5859985, 1238.7812500, -531.6979980, 1180.0780029, -1737.6639404, 1770.4792480
4: -317.3788757, 1044.4730225, -302.8516846, 996.0722046, -1313.4509277, 1347.3247070

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1856840, upper bound: 3656.1871326
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1856840, upper bound: 3656.1871326
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1427.1035156, 2702.1921387, -1355.5004883, 2577.1799316, -4004.2829590, 4057.6926270
1: -481.3479309, 1016.8422241, -458.8113708, 969.0219727, -1450.3698730, 1475.6534424
2: -245.6554260, 1023.2481689, -234.9862366, 976.0828247, -1221.7382812, 1258.2342529
3: -560.5891113, 1244.7539062, -535.3678589, 1188.4088135, -1748.9978027, 1780.1218262
4: -319.0415955, 1050.0837402, -304.9505005, 1003.3840942, -1322.4256592, 1355.0341797

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1851103, upper bound: 3656.1869603
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1851103, upper bound: 3656.1869603
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1449.3104248, 2733.7272949, -1461.9450684, 2750.5642090, -4199.8740234, 4195.6718750
1: -487.9602661, 1030.2261963, -491.5949097, 1037.7528076, -1525.7131348, 1521.8209229
2: -249.1735077, 1035.3966064, -251.1341553, 1043.0234375, -1292.1967773, 1286.5307617
3: -567.5737915, 1259.7534180, -572.2384033, 1268.7882080, -1836.3620605, 1831.9916992
4: -323.7851868, 1062.2753906, -326.3227844, 1070.5456543, -1394.3308105, 1388.5981445

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1659121, upper bound: 3656.1558290
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1659121, upper bound: 3656.1558511
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1425.3947754, 2691.0598145, -1638.1129150, 3075.4104004, -4500.8051758, 4329.1728516
1: -480.3513184, 1014.1132202, -550.6784058, 1160.9317627, -1641.2829590, 1564.7916260
2: -245.0843353, 1019.1158447, -282.1512451, 1167.8552246, -1412.9390869, 1301.2668457
3: -558.0936890, 1239.9705811, -640.6752930, 1419.3041992, -1977.3977051, 1880.6458740
4: -318.5247192, 1044.7226562, -365.7937317, 1201.0544434, -1519.5791016, 1410.5163574

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1659116, upper bound: 3656.1558290
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1659121, upper bound: 3656.1558511
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1479.0329590, 2790.0859375, -1486.2559814, 2808.2575684, -4287.2900391, 4276.3413086
1: -497.4905396, 1050.5515137, -500.7272949, 1057.7175293, -1555.2080078, 1551.2786865
2: -254.4097290, 1056.6798096, -255.9284058, 1063.7174072, -1318.1267090, 1312.6081543
3: -580.2836304, 1285.7849121, -583.5802612, 1294.4429932, -1874.7265625, 1869.3652344
4: -330.3396301, 1086.2824707, -332.3997498, 1092.0745850, -1422.4141846, 1418.6821289

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1777592, upper bound: 3656.1757974
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1877392, upper bound: 3656.1834215
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1877392, upper bound: 3656.1833227
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1492.6999512, 2814.1906738, -1516.9123535, 2864.1667480, -4356.8657227, 4331.1025391
1: -501.9397888, 1059.8497314, -510.8432922, 1078.5119629, -1580.4517822, 1570.6929932
2: -256.6395874, 1065.9542236, -260.9335022, 1084.7233887, -1341.3627930, 1326.8875732
3: -585.2982178, 1296.9011230, -594.8731079, 1319.6123047, -1904.9104004, 1891.7741699
4: -333.2321167, 1095.8098145, -338.8830261, 1113.5592041, -1446.7912598, 1434.6928711

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865966, upper bound: 3656.1833526
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865966, upper bound: 3656.1833142
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1537.1573486, 2901.1911621, -1515.8111572, 2861.5446777, -4398.7021484, 4417.0024414
1: -517.8334351, 1093.2569580, -510.7176208, 1078.6705322, -1596.5039062, 1603.9742432
2: -264.4441528, 1099.2423096, -260.9103394, 1084.3917236, -1348.8356934, 1360.1523438
3: -603.0876465, 1337.6608887, -595.0702515, 1319.8636475, -1922.9512939, 1932.7312012
4: -343.5615540, 1128.4293213, -338.9955139, 1113.2369385, -1456.7984619, 1467.4248047

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1826064, upper bound: 3656.1800082
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1895378, upper bound: 3656.1860533
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1894072, upper bound: 3656.1876233
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1894072, upper bound: 3656.1875075
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1552.3067627, 2928.1433105, -1548.1455078, 2920.7243652, -4473.0307617, 4476.2890625
1: -522.7834473, 1103.5605469, -521.4179688, 1100.6967773, -1623.4802246, 1624.9785156
2: -266.9383240, 1109.5498047, -266.2148132, 1106.6510010, -1373.5893555, 1375.7646484
3: -608.7033081, 1350.0443115, -607.0136719, 1346.5019531, -1955.2053223, 1957.0579834
4: -346.7898865, 1139.0690918, -345.8569336, 1135.9899902, -1482.7799072, 1484.9260254

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1875009, upper bound: 3656.1875683
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1875009, upper bound: 3656.1875009
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1479.0329590, 2790.0859375, -1553.4162598, 2954.2155762, -4433.2480469, 4343.5019531
1: -497.4905396, 1050.5515137, -526.0905151, 1110.7988281, -1608.2893066, 1576.6420898
2: -254.4097290, 1056.6798096, -269.3084412, 1118.5686035, -1372.9780273, 1325.9880371
3: -580.2836304, 1285.7849121, -612.9303589, 1361.4614258, -1941.7451172, 1898.7152100
4: -330.3396301, 1086.2824707, -349.4328003, 1148.9564209, -1479.2960205, 1435.7152100

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834495, upper bound: 3656.1823281
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834717, upper bound: 3656.1824090
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1492.6999512, 2814.1906738, -1586.7617188, 3016.0756836, -4508.7749023, 4400.9521484
1: -501.9397888, 1059.8497314, -537.1934204, 1133.6855469, -1635.6252441, 1597.0432129
2: -256.6395874, 1065.9542236, -274.9118652, 1141.7734375, -1398.4125977, 1340.8659668
3: -585.2982178, 1296.9011230, -625.4462891, 1389.4237061, -1974.7219238, 1922.3474121
4: -333.2321167, 1095.8098145, -356.6344910, 1172.8056641, -1506.0378418, 1452.4443359

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830793
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1848879, upper bound: 3656.1830325
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1519.8796387, 2863.2336426, -1570.7852783, 2978.7653809, -4498.6450195, 4434.0190430
1: -511.4862671, 1079.8128662, -531.4170532, 1121.6503906, -1633.1363525, 1611.2297363
2: -261.2852478, 1085.5812988, -272.1271057, 1129.0236816, -1390.3089600, 1357.7081299
3: -595.6827393, 1320.6263428, -619.1192627, 1374.2619629, -1969.9447021, 1939.7456055
4: -339.4577942, 1114.6169434, -353.1913757, 1159.9399414, -1499.3977051, 1467.8082275

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1785345, upper bound: 3656.1817289
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1421053, upper bound: 3656.1538992
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1421970, upper bound: 3656.1577622
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1481.2811279, 2799.4345703, -1664.0535889, 3149.2724609, -4630.5537109, 4463.4877930
1: -499.5818176, 1053.4886475, -562.3238525, 1183.1287842, -1682.7105713, 1615.8125000
2: -255.1115417, 1060.0722656, -288.1364136, 1192.5026855, -1447.6142578, 1348.2083740
3: -581.3674316, 1289.5957031, -654.0569458, 1448.7651367, -2030.1325684, 1943.6525879
4: -331.3534851, 1088.3328857, -373.4925232, 1226.6534424, -1558.0065918, 1461.8253174

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1482628, upper bound: 3656.1603196
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1482796, upper bound: 3656.1636963
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1295.7105713, 2463.7824707, -1325.5461426, 2525.8291016, -3821.5395508, 3789.3283691
1: -438.3490601, 925.9326782, -449.3715820, 949.1889038, -1387.5378418, 1375.3041992
2: -224.5829620, 932.1920166, -229.6471710, 955.5048218, -1180.0872803, 1161.8388672
3: -511.9200439, 1135.5030518, -523.4179688, 1163.4716797, -1675.3916016, 1658.9210205
4: -291.4964600, 958.9721069, -298.1504822, 980.5848999, -1272.0812988, 1257.1223145

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1800857, upper bound: 3656.1800213
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1800857, upper bound: 3656.1800413
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1295.7105713, 2463.7824707, -1554.7923584, 2959.9326172, -4255.6430664, 4018.5744629
1: -438.3490601, 925.9326782, -527.0149536, 1112.1751709, -1550.5241699, 1452.9476318
2: -224.5829620, 932.1920166, -269.1737366, 1119.8740234, -1344.4569092, 1201.3653564
3: -511.9200439, 1135.5030518, -612.8299561, 1362.7397461, -1874.6597900, 1748.3330078
4: -291.4964600, 958.9721069, -349.3823547, 1148.8337402, -1440.3302002, 1308.3543701

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1800857, upper bound: 3656.1800220
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1800857, upper bound: 3656.1800413
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1342.1259766, 2553.6499023, -1349.9052734, 2569.6323242, -3911.7578125, 3903.5551758
1: -454.9230957, 961.2343750, -457.5352173, 966.5867310, -1421.5097656, 1418.7695312
2: -232.7212372, 967.4543457, -233.8355713, 972.8230591, -1205.5443115, 1201.2899170
3: -530.4423828, 1178.4503174, -533.0646362, 1184.7552490, -1715.1976318, 1711.5148926
4: -302.2681885, 993.1567383, -303.6951904, 998.2694092, -1300.5374756, 1296.8519287

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1806645, upper bound: 3656.1804104
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1806645, upper bound: 3656.1805431
time: 0.91 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.86 + 415.85 = 420.71 seconds
