## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 2861.10740463984


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-18001.9667969, 21840.2187500, -18001.9667969, 21840.2187500, -39842.1835938, 39842.1835938)
1: (-2072.5097656, 1876.1496582, -2072.5097656, 1876.1496582, -3948.6594238, 3948.6594238)
2: (-1211.9670410, 2068.2844238, -1211.9670410, 2068.2844238, -3280.2512207, 3280.2512207)
3: (-1004.3207397, 2161.7316895, -1004.3207397, 2161.7316895, -3166.0522461, 3166.0522461)
4: (-1455.2456055, 1834.0839844, -1455.2456055, 1834.0839844, -3289.3295898, 3289.3295898)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.46 + 2.30 = 5.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -2861.1360160, upper bound: 2861.1360160

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347527, upper bound: 2861.1355532
time: 1.06 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347527, upper bound: 2861.1347527
time: 0.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 3, lower bound: -2861.1347527, upper bound: 2861.1355532
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.22
Output dim: 3, lower bound: -2861.1347527, upper bound: 2861.1347527

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -17680.3300781, 21405.9316406, -17810.8925781, 21581.1406250, -39261.4648438, 39216.8242188
1: -2029.9102783, 1841.0512695, -2047.0758057, 1855.2624512, -3885.1723633, 3888.1269531
2: -1189.4149170, 2027.3892822, -1198.5552979, 2043.9154053, -3233.3303223, 3225.9445801
3: -986.4774170, 2118.4030762, -993.7183228, 2135.9045410, -3122.3818359, 3112.1210938
4: -1428.1351318, 1798.3940430, -1439.1336670, 1812.8228760, -3240.9577637, 3237.5266113

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1342278, upper bound: 2861.1346637
time: 0.87 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347526, upper bound: 2861.1355520
time: 0.80 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -19405.2324219, 23418.4121094, -17789.6015625, 21581.4609375, -40986.6914062, 41208.0117188
1: -2217.9970703, 2017.9523926, -2047.6002197, 1854.0843506, -4072.0815430, 4065.5524902
2: -1302.7626953, 2219.9653320, -1197.7589111, 2043.7132568, -3346.4760742, 3417.7241211
3: -1083.6083984, 2319.0744629, -992.6545410, 2135.9799805, -3219.5883789, 3311.7290039
4: -1565.3811035, 1970.3856201, -1438.3038330, 1812.3474121, -3377.7282715, 3408.6894531

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1342278, upper bound: 2861.1344155
time: 0.73 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347526, upper bound: 2861.1347527
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.22 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 3, lower bound: -2861.1342278, upper bound: 2861.1346637
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 3, lower bound: -2861.1347526, upper bound: 2861.1355520
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 3, lower bound: -2861.1342278, upper bound: 2861.1344155
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 3, lower bound: -2861.1347526, upper bound: 2861.1347527

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -17521.3515625, 21229.9296875, -17564.3945312, 21304.5937500, -38825.9453125, 38794.3125000
1: -2013.3557129, 1825.2303467, -2021.0284424, 1830.5646973, -3843.9204102, 3846.2587891
2: -1179.1400146, 2010.2635498, -1182.5404053, 2017.1007080, -3196.2395020, 3192.8039551
3: -977.7102051, 2100.7475586, -980.1140137, 2108.2270508, -3085.9372559, 3080.8613281
4: -1415.8984375, 1783.0288086, -1420.0424805, 1788.8121338, -3204.7104492, 3203.0705566

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1336171
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1335242
time: 0.87 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -17381.4609375, 21062.6171875, -18038.5351562, 21836.4628906, -39217.9179688, 39101.1523438
1: -1997.8211670, 1810.7445068, -2070.6062012, 1878.3728027, -3876.1938477, 3881.3505859
2: -1169.8511963, 1994.4482422, -1213.2022705, 2069.9799805, -3239.8310547, 3207.6503906
3: -969.7923584, 2084.6408691, -1006.1160278, 2161.9343262, -3131.7265625, 3090.7561035
4: -1404.5545654, 1768.8375244, -1457.3666992, 1835.6673584, -3240.2216797, 3226.2041016

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347457, upper bound: 2861.1355520
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347493, upper bound: 2861.1353894
time: 0.89 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -19236.8300781, 23215.6953125, -17497.4707031, 21239.7792969, -40476.6093750, 40713.1601562
1: -2198.6652832, 2000.4727783, -2015.3194580, 1824.1606445, -4022.8259277, 4015.7922363
2: -1291.4278564, 2200.7827148, -1178.4152832, 2010.9128418, -3302.3398438, 3379.1979980
3: -1074.2611084, 2298.9340820, -976.4462891, 2101.9753418, -3176.2363281, 3275.3803711
4: -1551.9188232, 1953.3192139, -1415.1929932, 1783.0941162, -3335.0129395, 3368.5117188

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1332354
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1333315
time: 0.87 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -19065.2949219, 23036.0703125, -17997.2929688, 21813.6152344, -40878.9101562, 41033.3632812
1: -2182.3229980, 1983.9106445, -2068.6499023, 1875.4382324, -4057.7607422, 4052.5605469
2: -1280.8034668, 2183.1645508, -1211.2077637, 2067.0090332, -3347.8125000, 3394.3720703
3: -1064.6721191, 2281.3984375, -1004.0169678, 2159.5998535, -3224.2719727, 3285.4152832
4: -1538.9299316, 1937.2597656, -1455.2263184, 1833.2110596, -3372.1403809, 3392.4860840

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1336480
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1339719
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.54 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1336171
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1335242
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1347457, upper bound: 2861.1355520
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1347493, upper bound: 2861.1353894
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1332354
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1333315
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1336480
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.54
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1339719

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -17092.8027344, 20726.6367188, -17297.1621094, 20991.5488281, -38084.3515625, 38023.7968750
1: -1966.5229492, 1780.9523926, -1991.9227295, 1802.9539795, -3769.4765625, 3772.8750000
2: -1150.2034912, 1962.7445068, -1164.4976807, 1987.5427246, -3137.7458496, 3127.2416992
3: -953.6291504, 2051.2062988, -965.0888672, 2077.3854980, -3031.0146484, 3016.2949219
4: -1381.3692627, 1740.2508545, -1398.4876709, 1762.1784668, -3143.5466309, 3138.7385254

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1336169
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1336170
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17718.4824219, 21447.6328125, -17216.5644531, 20903.3476562, -38621.8281250, 38664.1914062
1: -2034.5626221, 1844.2044678, -1983.4238281, 1795.0332031, -3829.5957031, 3827.6279297
2: -1190.9267578, 2032.0166016, -1159.5274658, 1978.7908936, -3169.7167969, 3191.5434570
3: -988.0057373, 2121.8452148, -960.6203613, 2068.0458984, -3056.0517578, 3082.4650879
4: -1430.1284180, 1801.6916504, -1392.4296875, 1754.5350342, -3184.6625977, 3194.1208496

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1335242
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1335242
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17042.3867188, 20640.6953125, -17845.7382812, 21602.4082031, -38644.7968750, 38486.4218750
1: -1957.4608154, 1774.9898682, -2048.0795898, 1858.3489990, -3815.8090820, 3823.0690918
2: -1146.7117920, 1954.5344238, -1200.1829834, 2047.6469727, -3194.3581543, 3154.7172852
3: -950.9591064, 2042.7849121, -995.4088135, 2138.6804199, -3089.6389160, 3038.1936035
4: -1376.8605957, 1733.4429932, -1441.7960205, 1815.8483887, -3192.7080078, 3175.2387695

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347457, upper bound: 2861.1353442
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347041, upper bound: 2861.1353442
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -17804.9472656, 21504.3496094, -17822.5839844, 21579.0351562, -39383.9765625, 39326.9335938
1: -2039.0089111, 1851.3647461, -2046.2360840, 1856.0455322, -3895.0541992, 3897.6005859
2: -1196.1595459, 2037.9896240, -1198.7518311, 2045.4339600, -3241.5930176, 3236.7414551
3: -993.1229858, 2129.8671875, -994.1845703, 2136.4228516, -3129.5458984, 3124.0517578
4: -1436.3977051, 1807.9503174, -1440.2067871, 1813.7320557, -3250.1293945, 3248.1572266

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347493, upper bound: 2861.1351847
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347041, upper bound: 2861.1351847
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18746.0898438, 22636.2714844, -17243.1718750, 20941.8164062, -39687.9023438, 39879.4375000
1: -2144.7304688, 1949.7628174, -1987.6026611, 1797.9355469, -3942.6660156, 3937.3654785
2: -1258.2824707, 2146.1345215, -1161.2476807, 1982.8034668, -3241.0852051, 3307.3820801
3: -1046.6341553, 2241.8371582, -962.1483154, 2072.6364746, -3119.2705078, 3203.9853516
4: -1512.2604980, 1904.1925049, -1394.6645508, 1757.7652588, -3270.0249023, 3298.8566895

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1332353
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1332354
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19302.2089844, 23268.1562500, -17135.1738281, 20820.7500000, -40122.9570312, 40403.3281250
1: -2204.3198242, 2005.3566895, -1976.0894775, 1787.1575928, -3991.4775391, 3981.4458008
2: -1294.6206055, 2206.6022949, -1154.4349365, 1970.9606934, -3265.5800781, 3361.0371094
3: -1077.1131592, 2304.1579590, -956.1425171, 2060.0939941, -3137.2067871, 3260.3002930
4: -1555.3452148, 1958.4344482, -1386.4246826, 1747.3488770, -3302.6938477, 3344.8591309

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1333315
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1333315
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18592.2519531, 22477.0527344, -17736.2050781, 21508.9589844, -40101.2109375, 40213.2578125
1: -2130.2846680, 1934.9731445, -2040.3843994, 1848.4732666, -3978.7575684, 3975.3574219
2: -1248.8127441, 2130.4631348, -1193.5694580, 2038.1922607, -3287.0048828, 3324.0327148
3: -1038.0424805, 2226.3286133, -989.3175659, 2129.5200195, -3167.5617676, 3215.6459961
4: -1500.6638184, 1889.9061279, -1434.2235107, 1807.2750244, -3307.9389648, 3324.1296387

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339618, upper bound: 2861.1335268
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339618, upper bound: 2861.1335270
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19095.9628906, 23043.4960938, -17627.9765625, 21384.9257812, -40480.8867188, 40671.4726562
1: -2183.6672363, 1985.0616455, -2028.3370361, 1837.6455078, -4021.3120117, 4013.3984375
2: -1281.5722656, 2184.8334961, -1186.7858887, 2026.0643311, -3307.6357422, 3371.6193848
3: -1065.5452881, 2282.2021484, -983.3888550, 2116.6760254, -3182.2211914, 3265.5910645
4: -1539.4152832, 1938.6275635, -1425.9218750, 1796.6054688, -3336.0205078, 3364.5488281

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339618, upper bound: 2861.1339719
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1339719
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.39 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1336169
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1336170
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1335242
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1335242
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1347457, upper bound: 2861.1353442
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1347041, upper bound: 2861.1353442
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1347493, upper bound: 2861.1351847
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1347041, upper bound: 2861.1351847
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1332353
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331667, upper bound: 2861.1332354
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1333315
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1331671, upper bound: 2861.1333315
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1339618, upper bound: 2861.1335268
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1339618, upper bound: 2861.1335270
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1339618, upper bound: 2861.1339719
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1339719

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17092.8027344, 20726.6367188, -17172.7285156, 20827.3105469, -37920.1132812, 37899.3671875
1: -1966.5229492, 1780.9523926, -1975.8282471, 1789.5590820, -3756.0820312, 3756.7807617
2: -1150.2034912, 1962.7445068, -1155.8630371, 1971.9530029, -3122.1560059, 3118.6071777
3: -953.6291504, 2051.2062988, -958.2017212, 2060.9311523, -3014.5603027, 3009.4079590
4: -1381.3692627, 1740.2508545, -1388.1203613, 1748.5433350, -3129.9118652, 3128.3706055

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1314160, upper bound: 2861.1302505
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1334357
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17092.8027344, 20726.6367188, -18847.9414062, 22755.4414062, -39848.2421875, 39574.5781250
1: -1966.5229492, 1780.9523926, -2155.5756836, 1960.2738037, -3926.7963867, 3936.5280762
2: -1150.2034912, 1962.7445068, -1265.1888428, 2157.2795410, -3307.4826660, 3227.9331055
3: -953.6291504, 2051.2062988, -1052.4172363, 2253.4450684, -3207.0742188, 3103.6232910
4: -1381.3692627, 1740.2508545, -1520.5999756, 1914.2844238, -3295.6533203, 3260.8505859

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1314160, upper bound: 2861.1302505
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1334357
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17718.4824219, 21447.6328125, -17095.0546875, 20742.3593750, -38460.8359375, 38542.6835938
1: -2034.5626221, 1844.2044678, -1967.6359863, 1781.9140625, -3816.4763184, 3811.8403320
2: -1190.9267578, 2032.0166016, -1151.0922852, 1963.5286865, -3154.4545898, 3183.1088867
3: -988.0057373, 2121.8452148, -953.8908691, 2051.8737793, -3039.8793945, 3075.7355957
4: -1430.1284180, 1801.6916504, -1382.2687988, 1741.1871338, -3171.3151855, 3183.9604492

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1307011
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1333301
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17718.4824219, 21447.6328125, -18879.0722656, 22800.6972656, -40519.1757812, 40326.6953125
1: -2034.5626221, 1844.2044678, -2159.5144043, 1963.7894287, -3998.3518066, 4003.7187500
2: -1190.9267578, 2032.0166016, -1267.7322998, 2161.0395508, -3351.9660645, 3299.7485352
3: -988.0057373, 2121.8452148, -1054.2962646, 2257.4013672, -3245.4072266, 3176.1413574
4: -1430.1284180, 1801.6916504, -1523.5457764, 1917.8576660, -3347.9858398, 3325.2373047

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1307011
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1333301
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16925.5195312, 20503.4140625, -17658.7949219, 21383.1445312, -38308.6601562, 38162.2031250
1: -1944.5240479, 1762.9802246, -2027.4993896, 1839.1317139, -3783.6557617, 3790.4794922
2: -1139.0084229, 1941.3842773, -1187.8869629, 2026.6848145, -3165.6926270, 3129.2709961
3: -944.4252930, 2029.0566406, -984.9554443, 2116.7622070, -3061.1875000, 3014.0122070
4: -1367.5780029, 1721.7456055, -1426.9167480, 1797.1901855, -3164.7675781, 3148.6618652

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339246, upper bound: 2861.1341298
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347956
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16722.3593750, 20291.6054688, -18207.4785156, 22270.8808594, -38993.2382812, 38499.0820312
1: -1924.7803955, 1743.2807617, -2119.5031738, 1903.1557617, -3827.9357910, 3862.7839355
2: -1126.2540283, 1920.3762207, -1227.7496338, 2108.1696777, -3234.4230957, 3148.1254883
3: -933.2231445, 2007.6468506, -1015.2808838, 2202.5983887, -3135.8215332, 3022.9277344
4: -1352.3344727, 1702.8569336, -1477.2716064, 1865.2261963, -3217.5605469, 3180.1284180

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339190, upper bound: 2861.1340958
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273486, upper bound: 2861.1250243
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1345948, upper bound: 2861.1351883
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17679.2949219, 21356.1328125, -17629.2910156, 21351.5957031, -39030.8906250, 38985.4218750
1: -2025.0476074, 1838.3670654, -2024.8613281, 1836.0909424, -3861.1379395, 3863.2280273
2: -1187.8319092, 2023.8217773, -1185.9833984, 2023.6582031, -3211.4899902, 3209.8051758
3: -986.0870361, 2115.0368652, -983.3464966, 2113.6408691, -3099.7277832, 3098.3830566
4: -1426.3485107, 1795.3281250, -1424.7805176, 1794.3489990, -3220.6975098, 3220.1086426

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17479.8339844, 21146.0117188, -18191.5449219, 22269.4316406, -39749.2656250, 39337.5546875
1: -2005.4177246, 1819.0396729, -2119.6215820, 1902.2384033, -3907.6562500, 3938.6611328
2: -1175.3183594, 2002.9486084, -1227.0261230, 2107.8840332, -3283.2023926, 3229.9741211
3: -975.0869751, 2093.9030762, -1014.5982666, 2202.2988281, -3177.3852539, 3108.5012207
4: -1411.3771973, 1776.5825195, -1476.5555420, 1864.6733398, -3276.0505371, 3253.1381836

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -18746.0898438, 22636.2714844, -17172.7285156, 20827.3105469, -39573.3984375, 39809.0000000
1: -2144.7304688, 1949.7628174, -1975.8282471, 1789.5590820, -3934.2895508, 3925.5910645
2: -1258.2824707, 2146.1345215, -1155.8630371, 1971.9530029, -3230.2346191, 3301.9975586
3: -1046.6341553, 2241.8371582, -958.2017212, 2060.9311523, -3107.5654297, 3200.0388184
4: -1512.2604980, 1904.1925049, -1388.1203613, 1748.5433350, -3260.8032227, 3292.3122559

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1313718, upper bound: 2861.1301146
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1330714
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -18746.0898438, 22636.2714844, -18847.9414062, 22755.4414062, -41501.5312500, 41484.2109375
1: -2144.7304688, 1949.7628174, -2155.5756836, 1960.2738037, -4105.0039062, 4105.3383789
2: -1258.2824707, 2146.1345215, -1265.1888428, 2157.2795410, -3415.5615234, 3411.3232422
3: -1046.6341553, 2241.8371582, -1052.4172363, 2253.4450684, -3300.0791016, 3294.2541504
4: -1512.2604980, 1904.1925049, -1520.5999756, 1914.2844238, -3426.5444336, 3424.7924805

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1313718, upper bound: 2861.1301146
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1330714
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -19302.2089844, 23268.1562500, -17095.0546875, 20742.3593750, -40044.5625000, 40363.2109375
1: -2204.3198242, 2005.3566895, -1967.6359863, 1781.9140625, -3986.2336426, 3972.9926758
2: -1294.6206055, 2206.6022949, -1151.0922852, 1963.5286865, -3258.1481934, 3357.6945801
3: -1077.1131592, 2304.1579590, -953.8908691, 2051.8737793, -3128.9868164, 3258.0483398
4: -1555.3452148, 1958.4344482, -1382.2687988, 1741.1871338, -3296.5322266, 3340.7031250

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1316393, upper bound: 2861.1304114
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1331399
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -19302.2089844, 23268.1562500, -18879.0722656, 22800.6972656, -42102.9023438, 42147.2265625
1: -2204.3198242, 2005.3566895, -2159.5144043, 1963.7894287, -4168.1093750, 4164.8710938
2: -1294.6206055, 2206.6022949, -1267.7322998, 2161.0395508, -3455.6596680, 3474.3344727
3: -1077.1131592, 2304.1579590, -1054.2962646, 2257.4013672, -3334.5144043, 3358.4538574
4: -1555.3452148, 1958.4344482, -1523.5457764, 1917.8576660, -3473.2023926, 3481.9802246

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1316393, upper bound: 2861.1304114
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1331399
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -18451.4316406, 22312.4277344, -17532.7343750, 21269.2109375, -39720.6406250, 39845.1640625
1: -2114.7932129, 1920.4606934, -2017.8380127, 1827.4573975, -3942.2504883, 3938.2988281
2: -1239.5350342, 2114.6933594, -1180.1235352, 2015.2130127, -3254.7478027, 3294.8164062
3: -1030.1674805, 2209.8632812, -977.9147949, 2105.5205078, -3135.6879883, 3187.7780762
4: -1489.4572754, 1875.8514404, -1417.9951172, 1786.8458252, -3276.3029785, 3293.8464355

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1274865, upper bound: 2861.1246142
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333991
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -18318.5644531, 22170.8007812, -18138.5332031, 22238.0488281, -40556.6132812, 40309.3359375
1: -2101.4831543, 1907.6303711, -2117.2724609, 1898.1862793, -3999.6684570, 4024.9028320
2: -1231.1562500, 2100.6716309, -1224.1158447, 2104.2292480, -3335.3854980, 3324.7875977
3: -1022.9238281, 2195.6818848, -1011.6508179, 2198.7573242, -3221.6811523, 3207.3325195
4: -1479.4401855, 1863.3079834, -1473.3924561, 1861.0383301, -3340.4782715, 3336.7004395

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1274879, upper bound: 2861.1246141
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333994
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -18962.5019531, 22885.1152344, -17432.9531250, 21156.7343750, -40119.2343750, 40318.0585938
1: -2168.7761230, 1971.1947021, -2006.8901367, 1817.5737305, -3986.3498535, 3978.0849609
2: -1272.7147217, 2169.7595215, -1173.9428711, 2004.1602783, -3276.8745117, 3343.7023926
3: -1058.0484619, 2266.3928223, -972.4698486, 2093.8029785, -3151.8510742, 3238.8627930
4: -1528.6761475, 1925.2094727, -1410.3997803, 1777.1044922, -3305.7807617, 3335.6088867

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273965, upper bound: 2861.1248943
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -18789.7480469, 22700.6621094, -18005.2363281, 22085.6015625, -40875.3515625, 40705.8984375
1: -2151.4023438, 1954.4454346, -2102.6130371, 1884.7720947, -4036.1743164, 4057.0585938
2: -1261.8499756, 2151.4919434, -1215.5981445, 2089.2399902, -3351.0895996, 3367.0895996
3: -1048.6375732, 2247.8583984, -1004.3142700, 2183.0854492, -3231.7229004, 3252.1726074
4: -1515.7044678, 1908.8980713, -1463.0240479, 1847.8447266, -3363.5493164, 3371.9221191

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273980, upper bound: 2861.1248943
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
time: 0.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.91 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1314160, upper bound: 2861.1302505
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1334357
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1314160, upper bound: 2861.1302505
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1334357
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1307011
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1333301
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1307011
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1317380, upper bound: 2861.1333301
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1339246, upper bound: 2861.1341298
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347956
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1273486, upper bound: 2861.1250243
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1345948, upper bound: 2861.1351883
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1313718, upper bound: 2861.1301146
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1330714
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1313718, upper bound: 2861.1301146
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1330714
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1316393, upper bound: 2861.1304114
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1331399
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1316393, upper bound: 2861.1304114
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1329681, upper bound: 2861.1331399
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1274865, upper bound: 2861.1246142
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333991
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1274879, upper bound: 2861.1246141
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333994
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1273965, upper bound: 2861.1248943
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1273980, upper bound: 2861.1248943
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16601.8554688, 20165.7382812, -16940.5761719, 20543.3535156, -37145.2070312, 37106.3085938
1: -1913.7990723, 1730.9779053, -1949.1561279, 1764.9962158, -3678.7951660, 3680.1335449
2: -1117.4772949, 1909.2380371, -1139.7570801, 1945.5852051, -3063.0625000, 3048.9951172
3: -926.3304443, 1995.3227539, -945.1918945, 2032.8999023, -2959.2302246, 2940.5141602
4: -1342.8221436, 1692.1500244, -1369.0727539, 1724.8807373, -3067.7028809, 3061.2226562

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1343098, upper bound: 2861.1341255
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1341225, upper bound: 2861.1340859
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16847.5585938, 20370.9042969, -17013.1757812, 20593.9257812, -37441.4804688, 37384.0781250
1: -1931.5612793, 1753.0330811, -1952.8886719, 1771.3205566, -3702.8815918, 3705.9213867
2: -1132.2485352, 1930.0029297, -1144.1346436, 1950.5212402, -3082.7695312, 3074.1372070
3: -939.8657837, 2016.1743164, -949.2381592, 2037.9036865, -2977.7692871, 2965.4125977
4: -1359.5970459, 1711.9594727, -1373.9245605, 1730.0111084, -3089.6081543, 3085.8840332

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1352056, upper bound: 2861.1341674
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1346267, upper bound: 2861.1341675
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16601.8554688, 20165.7382812, -18572.5078125, 22412.2265625, -39014.0781250, 38738.2421875
1: -1913.7990723, 1730.9779053, -2123.3332520, 1930.7375488, -3844.5361328, 3854.3105469
2: -1117.4772949, 1909.2380371, -1245.8834229, 2125.5124512, -3242.9897461, 3155.1213379
3: -926.3304443, 1995.3227539, -1036.8613281, 2219.6711426, -3146.0012207, 3032.1840820
4: -1342.8221436, 1692.1500244, -1497.6243896, 1885.9145508, -3228.7365723, 3189.7736816

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1295028, upper bound: 2861.1283387
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1302069, upper bound: 2861.1289912
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16847.5585938, 20370.9042969, -18715.5820312, 22565.9199219, -39413.4765625, 39086.4765625
1: -1931.5612793, 1753.0330811, -2136.8298340, 1945.3856201, -3876.9467773, 3889.8627930
2: -1132.2485352, 1930.0029297, -1255.6220703, 2139.9345703, -3272.1826172, 3185.6247559
3: -939.8657837, 2016.1743164, -1045.0606689, 2234.8037109, -3174.6689453, 3061.2346191
4: -1359.5970459, 1711.9594727, -1509.0844727, 1899.3288574, -3258.9257812, 3221.0439453

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332163
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329651, upper bound: 2861.1332313
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17204.4746094, 20864.2167969, -16848.1230469, 20442.5800781, -37647.0429688, 37712.3320312
1: -1979.6228027, 1792.1760254, -1939.5218506, 1755.8830566, -3735.5058594, 3731.6977539
2: -1156.7296143, 1976.3088379, -1133.9952393, 1935.6463623, -3092.3759766, 3110.3041992
3: -959.5562744, 2063.7277832, -940.1024780, 2022.3037109, -2981.8598633, 3003.8303223
4: -1389.7303467, 1751.6505127, -1361.9105225, 1716.1343994, -3105.8642578, 3113.5607910

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1342763, upper bound: 2861.1336209
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1341008, upper bound: 2861.1335421
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17486.5429688, 21106.5371094, -16939.5468750, 20514.3593750, -38000.9023438, 38046.0820312
1: -2001.0059814, 1817.3618164, -1945.1839600, 1764.0247803, -3765.0307617, 3762.5454102
2: -1173.8771973, 2000.6962891, -1139.6752930, 1942.5767822, -3116.4541016, 3140.3715820
3: -974.9412231, 2088.3237305, -945.1312256, 2029.4769287, -3004.4182129, 3033.4550781
4: -1409.4226074, 1774.6932373, -1368.4190674, 1723.1068115, -3132.5292969, 3143.1123047

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1351539, upper bound: 2861.1346886
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1350315, upper bound: 2861.1346886
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17204.4746094, 20864.2167969, -18596.3203125, 22446.2343750, -39650.6953125, 39460.5351562
1: -1979.6228027, 1792.1760254, -2126.2451172, 1933.5069580, -3913.1291504, 3918.4211426
2: -1156.7296143, 1976.3088379, -1247.9545898, 2128.2502441, -3284.9799805, 3224.2634277
3: -959.5562744, 2063.7277832, -1038.3200684, 2222.4311523, -3181.9870605, 3102.0478516
4: -1389.7303467, 1751.6505127, -1499.9364014, 1888.5826416, -3278.3129883, 3251.5864258

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1297585, upper bound: 2861.1287208
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1304693, upper bound: 2861.1293784
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17486.5429688, 21106.5371094, -18750.8437500, 22616.8007812, -40103.3398438, 39857.3750000
1: -2001.0059814, 1817.3618164, -2141.4094238, 1949.4206543, -3950.4260254, 3958.7709961
2: -1173.8771973, 2000.6962891, -1258.4921875, 2144.3403320, -3318.2172852, 3259.1884766
3: -974.9412231, 2088.3237305, -1047.1657715, 2239.3754883, -3214.3166504, 3135.4895020
4: -1409.4226074, 1774.6932373, -1512.3793945, 1903.4216309, -3312.8442383, 3287.0727539

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332102
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329651, upper bound: 2861.1333244
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16528.9765625, 20034.9609375, -17397.1660156, 21077.5410156, -37606.5156250, 37432.1250000
1: -1900.9145508, 1721.9459229, -1999.1737061, 1812.1140137, -3713.0285645, 3721.1193848
2: -1112.1787109, 1897.2084961, -1170.2197266, 1997.8155518, -3109.9941406, 3067.4282227
3: -922.1586304, 1983.0534668, -970.2304077, 2086.6340332, -3008.7919922, 2953.2834473
4: -1335.5355225, 1682.0168457, -1405.8730469, 1771.1959229, -3106.7302246, 3087.8894043

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339246, upper bound: 2861.1341296
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339246, upper bound: 2861.1341298
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17055.6445312, 20638.1171875, -17305.1679688, 20974.0878906, -38029.7343750, 37943.2851562
1: -1957.8979492, 1774.8840332, -1989.1284180, 1802.9860840, -3760.8840332, 3764.0122070
2: -1146.2518311, 1955.3287354, -1164.5196533, 1987.5871582, -3133.8388672, 3119.8483887
3: -950.8883667, 2041.9484863, -965.2031250, 2075.8046875, -3026.6931152, 3007.1516113
4: -1376.2906494, 1733.4202881, -1398.8673096, 1762.1715088, -3138.4621582, 3132.2875977

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347956
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347956
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16270.3818359, 19770.6347656, -17995.6406250, 22017.2226562, -38287.6015625, 37766.2734375
1: -1875.8548584, 1697.1229248, -2095.7954102, 1880.8601074, -3756.7145996, 3792.9184570
2: -1095.9161377, 1870.9835205, -1213.1995850, 2084.5549316, -3180.4709473, 3084.1826172
3: -908.1705322, 1955.9829102, -1003.4671021, 2177.5571289, -3085.7275391, 2959.4499512
4: -1316.6918945, 1658.4157715, -1460.0806885, 1843.8790283, -3160.5708008, 3118.4965820

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273486, upper bound: 2861.1250243
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273486, upper bound: 2861.1250243
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16488.0371094, 19953.6445312, -18028.6855469, 22016.2207031, -38504.2578125, 37982.3281250
1: -1891.4016113, 1716.6773682, -2094.5793457, 1883.0527344, -3774.4541016, 3811.2568359
2: -1109.2030029, 1889.2183838, -1214.9400635, 2084.5261230, -3193.7290039, 3104.1579590
3: -920.0831299, 1974.5939941, -1005.1878662, 2177.2958984, -3097.3789062, 2979.7814941
4: -1331.6730957, 1676.0180664, -1461.5966797, 1844.7403564, -3176.4135742, 3137.6145020

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1345948, upper bound: 2861.1351884
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1345948, upper bound: 2861.1351884
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -17250.7949219, 20857.9394531, -17368.7070312, 21046.9492188, -38297.7343750, 38226.6406250
1: -1978.7297363, 1794.3344727, -1996.6153564, 1809.1567383, -3787.8859863, 3790.9497070
2: -1159.1275635, 1976.6450195, -1168.3779297, 1994.8752441, -3154.0026855, 3145.0224609
3: -962.0610352, 2065.9028320, -968.6786499, 2083.5734863, -3045.6342773, 3034.5815430
4: -1391.8831787, 1752.8394775, -1403.8110352, 1768.4433594, -3160.3259277, 3156.6503906

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339487, upper bound: 2861.1338513
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339487, upper bound: 2861.1338513
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -17755.3222656, 21417.0429688, -17272.3847656, 20939.4023438, -38694.7226562, 38689.4257812
1: -2031.4151611, 1844.1987305, -1986.1535645, 1799.6370850, -3831.0522461, 3830.3520508
2: -1191.1988525, 2030.9112549, -1162.4288330, 1984.2387695, -3175.4375000, 3193.3400879
3: -989.4409180, 2120.8518066, -963.4173584, 2072.3283691, -3061.7690430, 3084.2690430
4: -1430.3131104, 1801.0914307, -1396.5026855, 1759.0498047, -3189.3623047, 3197.5937500

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -17042.9609375, 20638.9980469, -17938.7265625, 21977.5996094, -39020.5546875, 38577.7187500
1: -1958.2810059, 1774.1766357, -2092.5593262, 1876.3021240, -3834.5830078, 3866.7358398
2: -1146.0838623, 1954.9222412, -1210.0357666, 2080.2756348, -3226.3593750, 3164.9572754
3: -950.6252441, 2043.8833008, -1000.4218750, 2173.5014648, -3124.1267090, 3044.3051758
4: -1376.3142090, 1733.3721924, -1456.4156494, 1839.7613525, -3216.0756836, 3189.7878418

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -17537.1699219, 21183.9570312, -17818.2070312, 21838.2304688, -39375.3984375, 39002.1601562
1: -2009.5245361, 1822.9456787, -2079.1220703, 1864.1253662, -3873.6499023, 3902.0671387
2: -1177.4378662, 2007.8905029, -1202.3074951, 2066.5979004, -3244.0354004, 3210.1975098
3: -977.4773560, 2097.3696289, -993.7792969, 2159.1645508, -3136.6418457, 3091.1489258
4: -1413.8685303, 1780.4841309, -1447.0375977, 1827.7407227, -3241.6093750, 3227.5217285

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -18306.3281250, 22101.0820312, -16940.5761719, 20543.3535156, -38849.6796875, 39041.6484375
1: -2094.0139160, 1903.6019287, -1949.1561279, 1764.9962158, -3859.0097656, 3852.7578125
2: -1227.8375244, 2096.0371094, -1139.7570801, 1945.5852051, -3173.4223633, 3235.7941895
3: -1022.1480103, 2188.8205566, -945.1918945, 2032.8999023, -3055.0478516, 3134.0122070
4: -1476.4263916, 1859.4250488, -1369.0727539, 1724.8807373, -3201.3063965, 3228.4978027

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1334701, upper bound: 2861.1322452
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1331845, upper bound: 2861.1321375
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18545.9707031, 22349.6953125, -17013.1757812, 20593.9257812, -39139.8945312, 39362.8710938
1: -2116.4611816, 1927.1990967, -1952.8886719, 1771.3205566, -3887.7812500, 3880.0878906
2: -1243.8175049, 2119.8813477, -1144.1346436, 1950.5212402, -3194.3388672, 3264.0153809
3: -1035.4832764, 2213.6872559, -949.2381592, 2037.9036865, -3073.3869629, 3162.9250488
4: -1494.8229980, 1881.5501709, -1373.9245605, 1730.0111084, -3224.8339844, 3255.4746094

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1348968, upper bound: 2861.1335094
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1346227, upper bound: 2861.1335088
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -18306.3281250, 22101.0820312, -18572.5078125, 22412.2265625, -40718.5546875, 40673.5820312
1: -2094.0139160, 1903.6019287, -2123.3332520, 1930.7375488, -4024.7507324, 4026.9348145
2: -1227.8375244, 2096.0371094, -1245.8834229, 2125.5124512, -3353.3498535, 3341.9204102
3: -1022.1480103, 2188.8205566, -1036.8613281, 2219.6711426, -3241.8190918, 3225.6818848
4: -1476.4263916, 1859.4250488, -1497.6243896, 1885.9145508, -3362.3400879, 3357.0488281

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1306723, upper bound: 2861.1287715
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1303595, upper bound: 2861.1289600
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18545.9707031, 22349.6953125, -18715.5820312, 22565.9199219, -41111.8906250, 41065.2734375
1: -2116.4611816, 1927.1990967, -2136.8298340, 1945.3856201, -4061.8466797, 4064.0288086
2: -1243.8175049, 2119.8813477, -1255.6220703, 2139.9345703, -3383.7519531, 3375.5029297
3: -1035.4832764, 2213.6872559, -1045.0606689, 2234.8037109, -3270.2866211, 3258.7473145
4: -1494.8229980, 1881.5501709, -1509.0844727, 1899.3288574, -3394.1518555, 3390.6347656

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1328879
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329651, upper bound: 2861.1328346
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18830.6562500, 22695.4648438, -16848.1230469, 20442.5800781, -39273.2226562, 39543.5859375
1: -2150.1950684, 1956.0468750, -1939.5218506, 1755.8830566, -3906.0781250, 3895.5683594
2: -1262.2192383, 2153.0791016, -1133.9952393, 1935.6463623, -3197.8652344, 3287.0739746
3: -1050.9432373, 2247.5097656, -940.1024780, 2022.3037109, -3073.2468262, 3187.6123047
4: -1517.0970459, 1910.6298828, -1361.9105225, 1716.1343994, -3233.2302246, 3272.5402832

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1333553, upper bound: 2861.1315866
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329386, upper bound: 2861.1315157
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -19095.6992188, 22973.5390625, -16939.5468750, 20514.3593750, -39610.0507812, 39913.0859375
1: -2175.3686523, 1982.2341309, -1945.1839600, 1764.0247803, -3939.3935547, 3927.4179688
2: -1279.7415771, 2179.6811523, -1139.6752930, 1942.5767822, -3222.3181152, 3319.3564453
3: -1065.6160889, 2275.1904297, -945.1312256, 2029.4769287, -3095.0927734, 3220.3217773
4: -1537.3459473, 1935.1496582, -1368.4190674, 1723.1068115, -3260.4526367, 3303.5688477

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347689, upper bound: 2861.1337343
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1347138, upper bound: 2861.1337343
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18830.6562500, 22695.4648438, -18596.3203125, 22446.2343750, -41276.8750000, 41291.7851562
1: -2150.1950684, 1956.0468750, -2126.2451172, 1933.5069580, -4083.7014160, 4082.2915039
2: -1262.2192383, 2153.0791016, -1247.9545898, 2128.2502441, -3390.4687500, 3401.0336914
3: -1050.9432373, 2247.5097656, -1038.3200684, 2222.4311523, -3273.3740234, 3285.8298340
4: -1517.0970459, 1910.6298828, -1499.9364014, 1888.5826416, -3405.6784668, 3410.5659180

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1309174, upper bound: 2861.1290796
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1306137, upper bound: 2861.1292924
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -19095.6992188, 22973.5390625, -18750.8437500, 22616.8007812, -41712.4843750, 41724.3789062
1: -2175.3686523, 1982.2341309, -2141.4094238, 1949.4206543, -4124.7890625, 4123.6435547
2: -1279.7415771, 2179.6811523, -1258.4921875, 2144.3403320, -3424.0812988, 3438.1733398
3: -1065.6160889, 2275.1904297, -1047.1657715, 2239.3754883, -3304.9914551, 3322.3562012
4: -1537.3459473, 1935.1496582, -1512.3793945, 1903.4216309, -3440.7675781, 3447.5290527

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1330584
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329652, upper bound: 2861.1331283
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -18030.2343750, 21801.5332031, -17259.7558594, 20904.6132812, -38934.8476562, 39061.2890625
1: -2066.5046387, 1876.2352295, -1983.4603271, 1796.8680420, -3863.3723145, 3859.6950684
2: -1210.4001465, 2066.8698730, -1160.2460938, 1981.5869141, -3191.9870605, 3227.1159668
3: -1006.6723022, 2159.2316895, -962.3787842, 2069.2966309, -3075.9689941, 3121.6101074
4: -1455.1055908, 1833.0673828, -1394.3182373, 1756.9866943, -3212.0917969, 3227.3854980

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1274865, upper bound: 2861.1246141
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1274865, upper bound: 2861.1246141
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18240.2617188, 22011.2792969, -17392.1718750, 21086.7128906, -39326.9765625, 39403.4414062
1: -2085.1101074, 1896.7039795, -1999.7518311, 1812.4437256, -3897.5537109, 3896.4558105
2: -1224.3098145, 2087.0642090, -1170.4412842, 1997.9602051, -3222.2700195, 3257.5053711
3: -1018.4133911, 2180.1760254, -970.1054688, 2087.4265137, -3105.8398438, 3150.2814941
4: -1471.1013184, 1851.9951172, -1406.2268066, 1771.8151855, -3242.9160156, 3258.2216797

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333991
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333991
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17898.3281250, 21655.4433594, -17924.3066406, 21978.4628906, -39876.7812500, 39579.7500000
1: -2052.7023926, 1863.2554932, -2092.8632812, 1875.4924316, -3928.1948242, 3956.1186523
2: -1201.9807129, 2052.5935059, -1209.4165039, 2079.8771973, -3281.8576660, 3262.0097656
3: -999.5059204, 2144.7438965, -999.6120605, 2173.1706543, -3172.6765137, 3144.3557129
4: -1445.0169678, 1820.3588867, -1455.9755859, 1839.1625977, -3284.1796875, 3276.3342285

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1274879, upper bound: 2861.1246141
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1274879, upper bound: 2861.1246142
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18116.0937500, 21880.5566406, -17962.5234375, 21981.3671875, -40097.4609375, 39843.0781250
1: -2072.8422852, 1884.7915039, -2092.1784668, 1878.0457764, -3950.8881836, 3976.9699707
2: -1216.5214844, 2074.0690918, -1211.1932373, 2080.6245117, -3297.1459961, 3285.2622070
3: -1011.6586304, 2167.1149902, -1001.6697388, 2173.5263672, -3185.1845703, 3168.7844238
4: -1461.8142090, 1840.3515625, -1457.6997070, 1840.5921631, -3302.4062500, 3298.0510254

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333994
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333994
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18485.8261719, 22313.0820312, -17157.0664062, 20788.9824219, -39274.8085938, 39470.1445312
1: -2114.6364746, 1921.5147705, -1972.2078857, 1786.6760254, -3901.3120117, 3893.7221680
2: -1240.0660400, 2116.1079102, -1153.8911133, 1970.1228027, -3210.1889648, 3269.9985352
3: -1031.5877686, 2209.6655273, -956.7669067, 2057.2687988, -3088.8562012, 3166.4323730
4: -1490.1853027, 1877.2476807, -1386.5085449, 1746.9008789, -3237.0861816, 3263.7558594

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273965, upper bound: 2861.1248943
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273965, upper bound: 2861.1248943
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -18748.9140625, 22580.0312500, -17294.9570312, 20977.3027344, -39726.2187500, 39874.9804688
1: -2138.7194824, 1947.2329102, -1989.1099854, 1802.8358154, -3941.5549316, 3936.3427734
2: -1257.2718506, 2141.7602539, -1164.4113770, 1987.1962891, -3244.4682617, 3306.1716309
3: -1046.1522217, 2236.3615723, -964.8133545, 2076.0017090, -3122.1538086, 3201.1748047
4: -1510.0928955, 1900.9884033, -1398.8315430, 1762.3448486, -3272.4377441, 3299.8198242

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18339.6445312, 22148.4960938, -17787.1289062, 21822.1875000, -40161.8320312, 39935.6250000
1: -2099.1223145, 1907.0317383, -2077.8796387, 1861.7307129, -3960.8530273, 3984.9111328
2: -1230.7171631, 2100.0395508, -1200.5892334, 2064.7160645, -3295.4328613, 3300.6286621
3: -1023.6626587, 2193.3139648, -992.1205444, 2157.0952148, -3180.7578125, 3185.4343262
4: -1478.9761963, 1863.0092773, -1445.2126465, 1825.7307129, -3304.7070312, 3308.2214355

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273980, upper bound: 2861.1248943
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1273980, upper bound: 2861.1248943
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -18582.5351562, 22403.6269531, -17832.5312500, 21833.5332031, -40416.0703125, 40236.1523438
1: -2122.1296387, 1931.1644287, -2077.9396973, 1865.1433105, -3987.2729492, 4009.1037598
2: -1246.8759766, 2124.2990723, -1203.1140137, 2065.9907227, -3312.8664551, 3327.4128418
3: -1037.1059570, 2218.6518555, -994.5401611, 2158.3173828, -3195.4233398, 3213.1916504
4: -1497.6589355, 1885.3834229, -1447.7957764, 1827.7377930, -3325.3964844, 3333.1789551

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.81 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1343098, upper bound: 2861.1341255
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1341225, upper bound: 2861.1340859
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1352056, upper bound: 2861.1341674
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1346267, upper bound: 2861.1341675
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1295028, upper bound: 2861.1283387
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1302069, upper bound: 2861.1289912
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332163
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329651, upper bound: 2861.1332313
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1342763, upper bound: 2861.1336209
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1341008, upper bound: 2861.1335421
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1351539, upper bound: 2861.1346886
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1350315, upper bound: 2861.1346886
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1297585, upper bound: 2861.1287208
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1304693, upper bound: 2861.1293784
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332102
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329651, upper bound: 2861.1333244
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339246, upper bound: 2861.1341296
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339246, upper bound: 2861.1341298
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347956
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347956
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1273486, upper bound: 2861.1250243
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1273486, upper bound: 2861.1250243
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1345948, upper bound: 2861.1351884
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1345948, upper bound: 2861.1351884
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339487, upper bound: 2861.1338513
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339487, upper bound: 2861.1338513
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339485, upper bound: 2861.1338513
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1345911
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1334701, upper bound: 2861.1322452
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1331845, upper bound: 2861.1321375
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1348968, upper bound: 2861.1335094
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1346227, upper bound: 2861.1335088
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1306723, upper bound: 2861.1287715
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1303595, upper bound: 2861.1289600
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1328879
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329651, upper bound: 2861.1328346
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1333553, upper bound: 2861.1315866
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329386, upper bound: 2861.1315157
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1347689, upper bound: 2861.1337343
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1347138, upper bound: 2861.1337343
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1309174, upper bound: 2861.1290796
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1306137, upper bound: 2861.1292924
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1330584
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1329652, upper bound: 2861.1331283
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1274865, upper bound: 2861.1246141
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1274865, upper bound: 2861.1246141
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333991
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333991
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1274879, upper bound: 2861.1246141
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1274879, upper bound: 2861.1246142
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333994
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338164, upper bound: 2861.1333994
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1273965, upper bound: 2861.1248943
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1273965, upper bound: 2861.1248943
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1273980, upper bound: 2861.1248943
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1273980, upper bound: 2861.1248943
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 3, lower bound: -2861.1338295, upper bound: 2861.1338295

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16205.0029297, 19652.5410156, -16330.6503906, 19749.1269531, -35954.1289062, 35983.1914062
1: -1864.0275879, 1688.5349121, -1872.0098877, 1699.5167236, -3563.5444336, 3560.5449219
2: -1090.1806641, 1861.0555420, -1097.7641602, 1870.9815674, -2961.1618652, 2958.8198242
3: -904.3222656, 1944.5216064, -911.3416748, 1954.3902588, -2858.7116699, 2855.8632812
4: -1309.7662354, 1650.0106201, -1318.1815186, 1659.7523193, -2969.5185547, 2968.1921387

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1337411, upper bound: 2861.1338112
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1337424, upper bound: 2861.1335243
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16244.6035156, 19703.5136719, -16692.6601562, 20167.9199219, -36412.5156250, 36396.1718750
1: -1869.8806152, 1692.3983154, -1912.6693115, 1736.0791016, -3605.9594727, 3605.0671387
2: -1092.5599365, 1866.4916992, -1120.9992676, 1912.1960449, -3004.7558594, 2987.4909668
3: -906.1380005, 1949.7995605, -931.1983032, 1996.6137695, -2902.7512207, 2880.9975586
4: -1312.8769531, 1654.3024902, -1346.4981689, 1696.0007324, -3008.8774414, 3000.8005371

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1332092, upper bound: 2861.1336971
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1332092, upper bound: 2861.1335057
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16639.9589844, 20116.7656250, -16691.7070312, 20199.4765625, -36839.4375000, 36808.4726562
1: -1907.4534912, 1731.2683105, -1915.4537354, 1737.5822754, -3645.0351562, 3646.7216797
2: -1118.2468262, 1905.7683105, -1122.4200439, 1912.9400635, -3031.1870117, 3028.1877441
3: -928.2911987, 1991.0877686, -931.3056030, 1999.0107422, -2927.3020020, 2922.3933105
4: -1342.7824707, 1690.4968262, -1347.8597412, 1696.7421875, -3039.5246582, 3038.3564453

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1351886, upper bound: 2861.1341657
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1351886, upper bound: 2861.1341675
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16656.9531250, 20156.1875000, -17578.0234375, 21286.3417969, -37943.2968750, 37734.2109375
1: -1911.3762207, 1733.8867188, -2014.8925781, 1832.7094727, -3744.0856934, 3748.7790527
2: -1119.8640137, 1909.2593994, -1183.2574463, 2017.3553467, -3137.2192383, 3092.5168457
3: -929.3290405, 1994.7446289, -982.6134644, 2105.8703613, -3035.1994629, 2977.3581543
4: -1344.7510986, 1693.3874512, -1421.4865723, 1790.4431152, -3135.1933594, 3114.8732910

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1332092, upper bound: 2861.1337389
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1332092, upper bound: 2861.1334749
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -16481.2187500, 19992.1210938, -18427.6835938, 22203.6074219, -38684.8203125, 38419.8046875
1: -1897.0371094, 1717.1835938, -2102.8559570, 1914.1917725, -3811.2290039, 3820.0393066
2: -1108.5830078, 1893.4219971, -1235.1705322, 2106.4111328, -3214.9941406, 3128.5920410
3: -919.4442749, 1978.3149414, -1028.6312256, 2198.5864258, -3118.0307617, 3006.9462891
4: -1332.1094971, 1678.3381348, -1484.8052979, 1869.3720703, -3201.4814453, 3163.1430664

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1284104, upper bound: 2861.1275945
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1284104, upper bound: 2861.1283387
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -16486.2441406, 20015.4980469, -18515.4882812, 22309.4882812, -38795.7304688, 38530.9843750
1: -1898.8480225, 1718.7125244, -2112.4631348, 1923.8459473, -3822.6936035, 3831.1755371
2: -1109.5686035, 1894.9239502, -1241.4340820, 2116.3710938, -3225.9394531, 3136.3579102
3: -920.0430298, 1980.4753418, -1033.9226074, 2209.9545898, -3129.9970703, 3014.3979492
4: -1333.2917480, 1679.6993408, -1492.2800293, 1878.3465576, -3211.6381836, 3171.9790039

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1287779, upper bound: 2861.1279561
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1287779, upper bound: 2861.1289912
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -16631.6132812, 20119.9511719, -18450.4023438, 22254.4824219, -38886.0937500, 38570.3515625
1: -1907.9444580, 1731.0864258, -2107.3925781, 1918.2866211, -3826.2302246, 3838.4790039
2: -1118.1921387, 1905.7871094, -1238.2393799, 2110.0468750, -3228.2390137, 3144.0263672
3: -927.8847046, 1991.3999023, -1030.3681641, 2204.0302734, -3131.9150391, 3021.7680664
4: -1342.5555420, 1690.4996338, -1488.1218262, 1872.8868408, -3215.4421387, 3178.6215820

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332163
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332163
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -16777.1699219, 20289.5664062, -18948.0039062, 22780.0937500, -39557.2656250, 39237.5625000
1: -1923.9807129, 1745.7215576, -2155.0527344, 1966.7504883, -3890.7309570, 3900.7744141
2: -1127.5959473, 1922.2008057, -1269.4958496, 2162.3640137, -3289.9592285, 3191.6960449
3: -935.9446411, 2008.0386963, -1058.2596436, 2256.2509766, -3192.1955566, 3066.2980957
4: -1354.0052490, 1704.9647217, -1526.2663574, 1920.1744385, -3274.1796875, 3231.2309570

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329631, upper bound: 2861.1332313
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329631, upper bound: 2861.1332313
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16824.7070312, 20367.8691406, -16264.2695312, 19676.7968750, -36501.4921875, 36632.1328125
1: -1931.4367676, 1751.3021240, -1864.9953613, 1692.9873047, -3624.4233398, 3616.2973633
2: -1130.5252686, 1929.6812744, -1093.7141113, 1863.7658691, -2994.2910156, 3023.3952637
3: -938.4187012, 2014.6800537, -907.6317139, 1946.5491943, -2884.9677734, 2922.3110352
4: -1358.0863037, 1710.9230957, -1313.2031250, 1653.4857178, -3011.5720215, 3024.1262207

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1342763, upper bound: 2861.1336209
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1342763, upper bound: 2861.1336209
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16802.7304688, 20356.6152344, -16605.4140625, 20074.1992188, -36876.9296875, 36962.0312500
1: -1931.5540771, 1749.3770752, -1903.6281738, 1727.7481689, -3659.3022461, 3653.0053711
2: -1129.0566406, 1929.1091309, -1115.6700439, 1902.9223633, -3031.9790039, 3044.7792969
3: -936.9698486, 2013.5985107, -926.5781860, 1986.7263184, -2923.6962891, 2940.1767578
4: -1356.5031738, 1709.7906494, -1339.8972168, 1687.8907471, -3044.3940430, 3049.6875000

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1341008, upper bound: 2861.1335421
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1341008, upper bound: 2861.1335421
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17305.4023438, 20895.2109375, -16617.1230469, 20138.8945312, -37444.2929688, 37512.3242188
1: -1981.0633545, 1798.8771973, -1909.8120117, 1731.1430664, -3712.2058105, 3708.6889648
2: -1162.1013184, 1980.2657471, -1118.6619873, 1906.3240967, -3068.4250488, 3098.9277344
3: -964.8978882, 2067.4655762, -927.2001953, 1992.3952637, -2957.2929688, 2994.6657715
4: -1395.1317139, 1756.6398926, -1342.9570312, 1691.0129395, -3086.1442871, 3099.5969238

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1351539, upper bound: 2861.1346886
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1351539, upper bound: 2861.1346886
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17374.4238281, 20976.8027344, -17151.9980469, 20726.6777344, -38101.1015625, 38128.7929688
1: -1988.8806152, 1805.8304443, -1963.3872070, 1784.2285156, -3773.1088867, 3769.2177734
2: -1166.5239258, 1988.2480469, -1152.8041992, 1964.4458008, -3130.9694824, 3141.0522461
3: -968.7199707, 2075.4006348, -957.3027344, 2050.6738281, -3019.3937988, 3032.7033691
4: -1400.5454102, 1763.5837402, -1384.7172852, 1743.4816895, -3144.0270996, 3148.3010254

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1351020, upper bound: 2861.1346886
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1351020, upper bound: 2861.1346886
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -17081.3808594, 20687.6171875, -18450.1406250, 22235.4492188, -39316.8281250, 39137.7460938
1: -1962.5769043, 1778.1188965, -2105.5588379, 1916.7878418, -3879.3647461, 3883.6777344
2: -1147.6685791, 1960.2181396, -1237.1218262, 2108.9570312, -3256.6254883, 3197.3391113
3: -952.5468750, 2046.4401855, -1030.0144043, 2201.1718750, -3153.7187500, 3076.4545898
4: -1378.8146973, 1737.5966797, -1486.9940186, 1871.8695068, -3250.6838379, 3224.5908203

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1289334, upper bound: 2861.1282798
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1289334, upper bound: 2861.1287208
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -17090.7402344, 20716.2148438, -18549.7753906, 22358.1503906, -39448.8906250, 39265.9921875
1: -1964.8631592, 1780.0718994, -2116.7307129, 1927.7675781, -3892.6308594, 3896.8022461
2: -1148.9606934, 1962.1812744, -1244.2846680, 2120.4291992, -3269.3898926, 3206.4655762
3: -953.3639526, 2049.0891113, -1036.0048828, 2213.9951172, -3167.3583984, 3085.0939941
4: -1380.3511963, 1739.3828125, -1495.5003662, 1882.1735840, -3262.5249023, 3234.8825684

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1295330, upper bound: 2861.1288759
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1295330, upper bound: 2861.1293784
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -17305.4023438, 20895.2109375, -18490.0527344, 22310.6621094, -39616.0625000, 39385.2656250
1: -1981.0633545, 1798.8771973, -2112.4340820, 1922.7768555, -3903.8398438, 3911.3110352
2: -1162.1013184, 1980.2657471, -1241.4145508, 2114.9631348, -3277.0642090, 3221.6801758
3: -964.8978882, 2067.4655762, -1032.7435303, 2209.1076660, -3174.0051270, 3100.2089844
4: -1395.1317139, 1756.6398926, -1491.7869873, 1877.4345703, -3272.5664062, 3248.4267578

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332102
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329444, upper bound: 2861.1332102
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -17374.4238281, 20976.8027344, -18902.6093750, 22742.3828125, -40116.8046875, 39879.4101562
1: -1988.8806152, 1805.8304443, -2151.1977539, 1962.8096924, -3951.6901855, 3957.0283203
2: -1166.5239258, 1988.2480469, -1267.1042480, 2158.1298828, -3324.6530762, 3255.3522949
3: -968.7199707, 2075.4006348, -1055.8692627, 2251.9111328, -3220.6308594, 3131.2700195
4: -1400.5454102, 1763.5837402, -1523.2739258, 1916.5223389, -3317.0678711, 3286.8571777

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329652, upper bound: 2861.1333245
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1329652, upper bound: 2861.1333245
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16528.9765625, 20034.9609375, -17272.4296875, 20915.5898438, -37444.5664062, 37307.3867188
1: -1900.9145508, 1721.9459229, -1983.3110352, 1798.6651611, -3699.5795898, 3705.2568359
2: -1112.1787109, 1897.2084961, -1161.5441895, 1982.3908691, -3094.5688477, 3058.7526855
3: -922.1586304, 1983.0534668, -963.2850952, 2070.4108887, -2992.5695801, 2946.3378906
4: -1335.5355225, 1682.0168457, -1395.4481201, 1757.6120605, -3093.1462402, 3077.4648438

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260586, upper bound: 2861.1232494
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1337850, upper bound: 2861.1340152
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16528.9765625, 20034.9609375, -19616.1562500, 23739.9921875, -40268.9687500, 39651.1132812
1: -1900.9145508, 1721.9459229, -2247.8925781, 2043.8319092, -3944.7460938, 3969.8383789
2: -1112.1787109, 1897.2084961, -1317.9003906, 2248.7800293, -3360.9582520, 3215.1088867
3: -922.1586304, 1983.0534668, -1095.8286133, 2348.5466309, -3270.7048340, 3078.8815918
4: -1335.5355225, 1682.0168457, -1585.9879150, 1994.5838623, -3330.1188965, 3268.0043945

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1260586, upper bound: 2861.1232494
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1337850, upper bound: 2861.1340152
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17055.6445312, 20638.1171875, -17183.0859375, 20815.5332031, -37871.1796875, 37821.1992188
1: -1957.8979492, 1774.8840332, -1973.5899658, 1789.8253174, -3747.7231445, 3748.4738770
2: -1146.2518311, 1955.3287354, -1156.0297852, 1972.4700928, -3118.7214355, 3111.3583984
3: -950.8883667, 2041.9484863, -958.4048462, 2059.9279785, -3010.8164062, 3000.3530273
4: -1376.2906494, 1733.4202881, -1388.6685791, 1748.8931885, -3125.1838379, 3122.0886230

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347956
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2861.1339719, upper bound: 2861.1347882
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17055.6445312, 20638.1171875, -19657.6425781, 23800.9531250, -40856.5976562, 40295.7578125
1: -1957.8979492, 1774.8840332, -2253.4536133, 2048.6567383, -4006.5544434, 4028.3371582
2: -1146.2518311, 1955.3287354, -1321.2237549, 2253.9033203, -3400.1550293, 3276.5524902
3: -950.8883667, 2041.9484863, -1098.2745361, 2353.9377441, -3304.8261719, 3140.2229004
4: -1376.2906494, 1733.4202881, -1589.8488770, 1999.2293701, -3375.5192871, 3323.2690430

Time for backsubstitution: 2.72 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.76 + 416.78 = 422.54 seconds
