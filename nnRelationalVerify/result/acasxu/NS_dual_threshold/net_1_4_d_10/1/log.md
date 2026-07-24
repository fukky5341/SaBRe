## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 886.64361740241


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742)
1: (-437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521)
2: (-439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498)
3: (-536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455)
4: (-473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.82 + 2.14 = 2.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -886.6879518, upper bound: 886.6879518

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6879186, upper bound: 886.6879400
time: 1.09 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 0, lower bound: -886.6879186, upper bound: 886.6879400
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -354.1024475, 568.2473145, -367.7500916, 592.1647339, -946.2670898, 935.9973755
1: -396.4194031, 562.9596558, -411.8090210, 586.4174805, -982.8369141, 974.7686768
2: -397.9836426, 557.5975342, -413.4058533, 580.7301025, -978.7137451, 971.0032959
3: -486.2630920, 648.3890381, -505.2509766, 675.4591064, -1161.7221680, 1153.6400146
4: -428.3536377, 638.0024414, -445.2143250, 664.1424561, -1092.4960938, 1083.2166748

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
time: 1.04 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
time: 0.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -529.4804688, 859.2293091, -375.7784729, 610.1853027, -1139.6657715, 1235.0078125
1: -596.3991699, 857.3454590, -421.2735291, 602.7648315, -1199.1640625, 1278.6190186
2: -595.2865601, 847.3470459, -422.5351868, 596.2720337, -1191.5585938, 1269.8822021
3: -733.5942383, 983.1370239, -516.7918091, 694.2438354, -1427.8380127, 1499.9288330
4: -640.1510010, 969.8715820, -456.0000610, 681.5297241, -1321.6805420, 1425.2207031

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
time: 0.98 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
time: 1.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.36 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.36
Output dim: 0, lower bound: -886.6879141, upper bound: 886.6879141

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -354.1024475, 568.2473145, -354.1024475, 568.2473145, -922.3497314, 922.3497314
1: -396.4194031, 562.9596558, -396.4194031, 562.9596558, -959.3790283, 959.3790283
2: -397.9836426, 557.5975342, -397.9836426, 557.5975342, -955.5811768, 955.5811768
3: -486.2630920, 648.3890381, -486.2630920, 648.3890381, -1134.6520996, 1134.6519775
4: -428.3536377, 638.0024414, -428.3536377, 638.0024414, -1066.3560791, 1066.3560791

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816830, upper bound: 886.6801243
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803794, upper bound: 886.6803794
time: 0.78 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -354.1024475, 568.2473145, -529.4804688, 859.2293091, -1213.3317871, 1097.7277832
1: -396.4194031, 562.9596558, -596.3991699, 857.3454590, -1253.7648926, 1159.3586426
2: -397.9836426, 557.5975342, -595.2865601, 847.3470459, -1245.3306885, 1152.8840332
3: -486.2630920, 648.3890381, -733.5942383, 983.1370239, -1469.4001465, 1381.9829102
4: -428.3536377, 638.0024414, -640.1510010, 969.8715820, -1397.2796631, 1278.1534424

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6879186, upper bound: 886.6858604
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6870979, upper bound: 886.6873203
time: 0.82 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -529.4804688, 859.2293091, -354.1024475, 568.2473145, -1097.7277832, 1213.3317871
1: -596.3991699, 857.3454590, -396.4194031, 562.9596558, -1159.3586426, 1253.7648926
2: -595.2865601, 847.3470459, -397.9836426, 557.5975342, -1152.8840332, 1245.3306885
3: -733.5942383, 983.1370239, -486.2630920, 648.3890381, -1381.9830322, 1469.4001465
4: -640.1510010, 969.8715820, -428.3536377, 638.0024414, -1278.1533203, 1397.2796631

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6855493, upper bound: 886.6879141
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6870921, upper bound: 886.6870921
time: 0.71 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -529.4804688, 859.2293091, -529.4804688, 859.2293091, -1388.7097168, 1388.7097168
1: -596.3991699, 857.3454590, -596.3991699, 857.3454590, -1452.0972900, 1452.0972900
2: -595.2865601, 847.3470459, -595.2865601, 847.3470459, -1441.0915527, 1441.0915527
3: -733.5942383, 983.1370239, -733.5942383, 983.1370239, -1715.7396240, 1715.7396240
4: -640.1510010, 969.8715820, -640.1510010, 969.8715820, -1607.1942139, 1607.1942139

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6824688
time: 1.11 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6809495
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.69 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6816830, upper bound: 886.6801243
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6803794, upper bound: 886.6803794
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6879186, upper bound: 886.6858604
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6870979, upper bound: 886.6873203
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6855493, upper bound: 886.6879141
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6870921, upper bound: 886.6870921
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6824688
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6809495

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -342.6473083, 549.9136353, -346.3334961, 555.7369385, -898.3841553, 896.2471313
1: -383.2557678, 544.6537476, -387.5244446, 550.4748535, -933.7305908, 932.1782227
2: -385.1361084, 539.4267578, -389.2693176, 545.2052612, -930.3413696, 928.6959229
3: -470.0091858, 627.5152588, -475.1812134, 634.1633911, -1104.1721191, 1102.6965332
4: -415.0746765, 617.3360596, -419.3348694, 623.9254150, -1039.0001221, 1036.6708984

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816248, upper bound: 886.6791138
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811471, upper bound: 886.6798548
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -390.0690002, 640.7873535, -340.9186707, 547.4867554, -937.5557251, 981.7060547
1: -436.3262329, 629.7741699, -382.1921692, 542.4670410, -978.7932739, 1011.9663086
2: -438.3491821, 623.4146118, -383.2426453, 537.0474243, -975.3965454, 1006.6572266
3: -535.7444458, 726.0071411, -468.7109985, 624.5856934, -1160.3300781, 1194.7181396
4: -470.9229736, 710.2755737, -413.4240723, 614.7454834, -1085.6682129, 1123.6997070

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803794, upper bound: 886.6790137
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797777, upper bound: 886.6797777
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -341.1024170, 548.8489380, -519.8854370, 845.2652588, -1186.3676758, 1068.7343750
1: -381.6498413, 543.0418701, -585.8307495, 842.5616455, -1224.2113037, 1128.8725586
2: -383.4052734, 537.9945679, -584.3614502, 832.7551880, -1215.4228516, 1122.3559570
3: -468.1733093, 625.6145630, -720.1342773, 965.9945679, -1434.1678467, 1345.7486572
4: -412.7966003, 615.0540771, -629.2649536, 952.5632935, -1363.5020752, 1244.3189697

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814195, upper bound: 886.6798633
time: 0.84 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6819864, upper bound: 886.6802103
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -351.7857971, 564.4530029, -525.1318359, 852.1044922, -1203.8902588, 1089.5845947
1: -393.8641968, 559.1889648, -591.5165405, 850.2245483, -1244.0886230, 1150.7053223
2: -395.3839722, 553.8792725, -590.3893433, 840.2908325, -1235.4282227, 1144.2685547
3: -483.1048584, 644.0614014, -727.5983887, 974.9738159, -1458.0784912, 1371.6595459
4: -425.6170044, 633.7583618, -634.8620605, 961.8915405, -1385.6807861, 1268.6203613

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812458, upper bound: 886.6820465
time: 0.86 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6810529, upper bound: 886.6803474
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -519.8854370, 845.2652588, -341.1024170, 548.8489380, -1068.7343750, 1186.3676758
1: -585.8307495, 842.5616455, -381.6498413, 543.0418701, -1128.8725586, 1224.2113037
2: -584.3614502, 832.7551880, -383.4052734, 537.9945679, -1122.3559570, 1215.4228516
3: -720.1342773, 965.9945679, -468.1733093, 625.6145630, -1345.7486572, 1434.1678467
4: -629.2649536, 952.5632935, -412.7966003, 615.0540771, -1244.3189697, 1363.5020752

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6814195
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802103, upper bound: 886.6819864
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -525.1318359, 852.1044922, -351.7857971, 564.4530029, -1089.5845947, 1203.8902588
1: -591.5165405, 850.2245483, -393.8641968, 559.1889648, -1150.7053223, 1244.0887451
2: -590.3893433, 840.2908325, -395.3839722, 553.8792725, -1144.2685547, 1235.4281006
3: -727.5983887, 974.9738159, -483.1048584, 644.0614014, -1371.6594238, 1458.0783691
4: -634.8620605, 961.8915405, -425.6170044, 633.7583618, -1268.6203613, 1385.6807861

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6820465, upper bound: 886.6812458
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803474, upper bound: 886.6810529
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -514.4626465, 834.4896851, -508.1027527, 824.5404053, -1339.0030518, 1342.5924072
1: -579.6018066, 832.5403442, -572.4766235, 822.5600586, -1400.2988281, 1402.9942627
2: -578.5418091, 822.7717285, -571.5482178, 812.8250732, -1389.7086182, 1392.4414062
3: -712.6386108, 954.6068115, -703.7421875, 943.1345825, -1654.3636475, 1656.8415527
4: -622.3593750, 941.8617554, -615.2960815, 930.3028564, -1550.0196533, 1554.4301758

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6801760
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6809495
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -518.6517334, 841.3345947, -510.4350586, 827.7733154, -1346.4250488, 1351.7696533
1: -584.1901855, 839.2843628, -574.9146729, 825.5961304, -1408.4399414, 1412.8374023
2: -583.1602173, 829.4910278, -573.9582520, 815.9797974, -1397.9370117, 1402.1546631
3: -718.5491333, 962.4592896, -707.1240845, 946.7854004, -1664.6105957, 1668.9141846
4: -627.0779419, 949.6379395, -617.1613770, 934.3087158, -1558.9331055, 1564.2648926

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6809495, upper bound: 886.6801760
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6809495, upper bound: 886.6809495
time: 1.10 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.84 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6816248, upper bound: 886.6791138
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6811471, upper bound: 886.6798548
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6803794, upper bound: 886.6790137
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6797777, upper bound: 886.6797777
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6814195, upper bound: 886.6798633
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6819864, upper bound: 886.6802103
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6812458, upper bound: 886.6820465
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6810529, upper bound: 886.6803474
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6814195
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6802103, upper bound: 886.6819864
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6820465, upper bound: 886.6812458
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6803474, upper bound: 886.6810529
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6801760
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6801760, upper bound: 886.6809495
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6809495, upper bound: 886.6801760
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 0, lower bound: -886.6809495, upper bound: 886.6809495

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -327.7272949, 527.1968994, -304.2355957, 492.0149536, -819.7422485, 831.4323730
1: -366.3355713, 521.3939819, -339.2859192, 485.1991882, -851.5347290, 860.6799316
2: -368.4119568, 516.4996948, -342.0159302, 480.5394287, -848.9514160, 858.5156250
3: -449.3239441, 600.8957520, -416.5084229, 559.7647095, -1009.0886230, 1017.4041748
4: -397.1821899, 590.6504517, -367.7330322, 549.0767822, -946.2589111, 958.3834839

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812722, upper bound: 886.6787130
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812722, upper bound: 886.6791138
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -341.2363892, 547.6173706, -344.1986389, 552.2764282, -893.5128174, 891.8159790
1: -381.7033691, 542.3707886, -385.1690979, 547.0507812, -928.7541504, 927.5397949
2: -383.5520020, 537.1614380, -386.8773804, 541.8023682, -925.3543701, 924.0388184
3: -468.0925293, 624.8709106, -472.2694397, 630.2049561, -1098.2974854, 1097.1403809
4: -413.3769226, 614.7592163, -416.7953796, 620.0413818, -1033.4180908, 1031.5545654

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6764133, upper bound: 886.6739102
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6770575, upper bound: 886.6741832
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -375.4254456, 618.4013672, -301.4024353, 487.9639893, -863.3894043, 919.8037109
1: -419.7203979, 606.9449463, -337.0548706, 481.5052490, -901.2256470, 943.9998169
2: -421.8510132, 600.7727661, -338.8489380, 476.6365662, -898.4875488, 939.6216431
3: -515.4085693, 699.9737549, -413.7070312, 555.1126709, -1070.5212402, 1113.6807861
4: -453.2984924, 684.0179443, -365.1127014, 544.7515259, -998.0500488, 1049.1306152

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6751453, upper bound: 886.6739203
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6748309, upper bound: 886.6739284
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -386.1738892, 634.6172485, -335.2683716, 538.4177856, -924.5916748, 969.8856201
1: -431.9323120, 623.4801636, -375.9209595, 533.3421631, -965.2744751, 999.4011230
2: -433.9816895, 617.2265015, -376.9728699, 528.0310059, -962.0126953, 994.1992798
3: -530.3025513, 718.8379517, -460.9445801, 614.1746826, -1144.4771729, 1179.7823486
4: -466.3063354, 703.1750488, -406.8312073, 604.4712524, -1070.7772217, 1110.0061035

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797777, upper bound: 886.6794150
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797777, upper bound: 886.6796883
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -319.3588257, 513.3963013, -499.5787354, 811.9069214, -1131.2657471, 1012.9750366
1: -357.4892883, 507.5682373, -562.9760132, 808.9344482, -1166.3675537, 1070.5440674
2: -359.2570801, 502.6076355, -561.7161865, 799.4581909, -1157.9121094, 1064.3238525
3: -438.1957397, 584.5921631, -691.6407471, 927.4205322, -1365.6162109, 1276.2329102
4: -387.4924011, 575.1215210, -605.3459473, 914.5051880, -1300.4481201, 1180.4674072

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6791662, upper bound: 886.6780525
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814195, upper bound: 886.6798278
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6810945, upper bound: 886.6798633
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -336.1912231, 540.7302856, -499.1142883, 811.0087280, -1147.1997070, 1039.8446045
1: -376.0331116, 534.8093262, -562.5621338, 808.1931763, -1184.2261963, 1097.3714600
2: -377.9038391, 529.9337769, -561.1648560, 798.7039185, -1176.1324463, 1091.0986328
3: -461.3590698, 616.1744995, -691.3553467, 926.5415649, -1387.9005127, 1307.5297852
4: -406.8352356, 605.8259277, -604.4805908, 913.9580688, -1319.2049561, 1210.3063965

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814195, upper bound: 886.6796819
time: 1.73 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812995, upper bound: 886.6802103
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -332.0552979, 532.6130981, -503.6023560, 817.2449951, -1149.3002930, 1036.2154541
1: -371.8782043, 527.3163452, -567.4346924, 815.3160400, -1187.1942139, 1094.7509766
2: -373.4598389, 522.0651245, -566.4943848, 805.6324463, -1178.9047852, 1088.5595703
3: -455.8274231, 607.2841187, -697.5782471, 934.8330078, -1390.6602783, 1304.8623047
4: -402.6110535, 597.8049927, -609.9027710, 922.1452026, -1323.3199463, 1207.7076416

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6810732, upper bound: 886.6814176
time: 0.92 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6810732, upper bound: 886.6818324
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -345.9162598, 554.7746582, -505.9755249, 820.4547729, -1166.3707275, 1060.7502441
1: -387.2107544, 549.4042969, -569.8987427, 818.2639771, -1205.4747314, 1119.3029785
2: -388.8085022, 544.2700806, -568.9277344, 808.7182617, -1197.5190430, 1113.1977539
3: -474.9815063, 632.8267212, -700.9715576, 938.3798218, -1413.3610840, 1333.7983398
4: -418.5682068, 622.7348022, -611.6838989, 926.0924683, -1343.1304932, 1234.4185791

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808867, upper bound: 886.6796979
time: 1.18 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808867, upper bound: 886.6801176
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -499.5787354, 811.9069214, -319.3588257, 513.3963013, -1012.9750366, 1131.2657471
1: -562.9760132, 808.9344482, -357.4892883, 507.5682373, -1070.5441895, 1166.3676758
2: -561.7161865, 799.4581909, -359.2570801, 502.6076355, -1064.3238525, 1157.9121094
3: -691.6407471, 927.4205322, -438.1957397, 584.5921631, -1276.2329102, 1365.6162109
4: -605.3459473, 914.5051880, -387.4924011, 575.1215210, -1180.4674072, 1300.4481201

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6780525, upper bound: 886.6791662
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798278, upper bound: 886.6814195
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6810945
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -499.1142883, 811.0087280, -336.1912231, 540.7302856, -1039.8446045, 1147.1998291
1: -562.5621338, 808.1931763, -376.0331116, 534.8093262, -1097.3714600, 1184.2261963
2: -561.1648560, 798.7039185, -377.9038391, 529.9337769, -1091.0986328, 1176.1324463
3: -691.3553467, 926.5415649, -461.3590698, 616.1744995, -1307.5297852, 1387.9005127
4: -604.4805908, 913.9580688, -406.8352356, 605.8259277, -1210.3063965, 1319.2049561

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796819, upper bound: 886.6819864
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6812995
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -503.6023560, 817.2449951, -332.0552979, 532.6130981, -1036.2154541, 1149.3002930
1: -567.4346924, 815.3160400, -371.8782043, 527.3163452, -1094.7509766, 1187.1942139
2: -566.4943848, 805.6324463, -373.4598389, 522.0651245, -1088.5595703, 1178.9047852
3: -697.5782471, 934.8330078, -455.8274231, 607.2841187, -1304.8623047, 1390.6602783
4: -609.9027710, 922.1452026, -402.6110535, 597.8049927, -1207.7076416, 1323.3199463

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6810732
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6810732
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -505.9755249, 820.4547729, -345.9162598, 554.7746582, -1060.7502441, 1166.3707275
1: -569.8987427, 818.2639771, -387.2107544, 549.4042969, -1119.3029785, 1205.4747314
2: -568.9277344, 808.7182617, -388.8085022, 544.2700806, -1113.1977539, 1197.5189209
3: -700.9715576, 938.3798218, -474.9815063, 632.8267212, -1333.7983398, 1413.3610840
4: -611.6838989, 926.0924683, -418.5682068, 622.7348022, -1234.4185791, 1343.1304932

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796979, upper bound: 886.6808867
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6808867
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -508.1027527, 824.5404053, -508.1027527, 824.5404053, -1332.6431885, 1332.6431885
1: -572.4766235, 822.5600586, -572.4766235, 822.5600586, -1393.2694092, 1393.2692871
2: -571.5482178, 812.8250732, -571.5482178, 812.8250732, -1382.7717285, 1382.7717285
3: -703.7421875, 943.1345825, -703.7421875, 943.1345825, -1645.5853271, 1645.5853271
4: -615.2960815, 930.3028564, -615.2960815, 930.3028564, -1543.1523438, 1543.1523438

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6783315, upper bound: 886.6800963
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801758, upper bound: 886.6819570
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -510.4350586, 827.7733154, -508.1027527, 824.5404053, -1334.9753418, 1335.8760986
1: -574.9146729, 825.5961304, -572.4766235, 822.5600586, -1395.7082520, 1396.1629639
2: -573.9582520, 815.9797974, -571.5482178, 812.8250732, -1385.2165527, 1385.8137207
3: -707.1240845, 946.7854004, -703.7421875, 943.1345825, -1648.9956055, 1649.1170654
4: -617.1613770, 934.3087158, -615.2960815, 930.3028564, -1544.8955078, 1547.0570068

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6783315, upper bound: 886.6801117
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6783315, upper bound: 886.6824467
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -508.1027527, 824.5404053, -510.4350586, 827.7733154, -1335.8760986, 1334.9753418
1: -572.4766235, 822.5600586, -574.9146729, 825.5961304, -1396.1630859, 1395.7082520
2: -571.5482178, 812.8250732, -573.9582520, 815.9797974, -1385.8135986, 1385.2165527
3: -703.7421875, 943.1345825, -707.1240845, 946.7854004, -1649.1170654, 1648.9956055
4: -615.2960815, 930.3028564, -617.1613770, 934.3087158, -1547.0570068, 1544.8955078

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5669172, upper bound: 886.6379611
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801758, upper bound: 886.6801758
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -510.4350586, 827.7733154, -510.4350586, 827.7733154, -1338.2083740, 1338.2083740
1: -574.9146729, 825.5961304, -574.9146729, 825.5961304, -1399.2292480, 1399.2292480
2: -573.9582520, 815.9797974, -573.9582520, 815.9797974, -1388.7603760, 1388.7604980
3: -707.1240845, 946.7854004, -707.1240845, 946.7854004, -1653.2811279, 1653.2811279
4: -617.1613770, 934.3087158, -617.1613770, 934.3087158, -1549.0668945, 1549.0668945

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5758117, upper bound: 886.5649660
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801758, upper bound: 886.6809272
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.49 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6812722, upper bound: 886.6787130
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6812722, upper bound: 886.6791138
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6764133, upper bound: 886.6739102
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6770575, upper bound: 886.6741832
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6751453, upper bound: 886.6739203
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6748309, upper bound: 886.6739284
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6797777, upper bound: 886.6794150
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6797777, upper bound: 886.6796883
NS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6814195, upper bound: 886.6798278
NS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6810945, upper bound: 886.6798633
NS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6814195, upper bound: 886.6796819
NS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6812995, upper bound: 886.6802103
NS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6810732, upper bound: 886.6814176
NS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6810732, upper bound: 886.6818324
NS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6808867, upper bound: 886.6796979
NS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6808867, upper bound: 886.6801176
NS_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6798278, upper bound: 886.6814195
NS_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6810945
NS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6796819, upper bound: 886.6819864
NS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6812995
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6810732
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6810732
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6796979, upper bound: 886.6808867
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6808867
NS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6783315, upper bound: 886.6800963
NS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6801758, upper bound: 886.6819570
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6783315, upper bound: 886.6801117
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6783315, upper bound: 886.6824467
NS_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.5669172, upper bound: 886.6379611
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6801758, upper bound: 886.6801758
NS_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.5758117, upper bound: 886.5649660
NS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.49
Output dim: 0, lower bound: -886.6801758, upper bound: 886.6809272

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -327.7272949, 527.1968994, -299.0959778, 483.5195618, -811.2468262, 826.2927856
1: -366.3355713, 521.3939819, -333.3561707, 476.6803589, -843.0159302, 854.7501221
2: -368.4119568, 516.4996948, -336.2389526, 472.0960999, -840.5080566, 852.7386475
3: -449.3239441, 600.8957520, -409.2476196, 550.0382690, -999.3621826, 1010.1433716
4: -397.1821899, 590.6504517, -361.5089417, 539.4716797, -936.6538086, 952.1594238

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765683, upper bound: 886.6739102
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6775239, upper bound: 886.6741832
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -327.7272949, 527.1968994, -383.4909973, 632.9713745, -960.2495117, 910.6878662
1: -366.3355713, 521.3939819, -428.3980408, 619.8620605, -986.1976318, 949.7919922
2: -368.4119568, 516.4996948, -430.7797852, 614.1267700, -982.5386963, 947.2794800
3: -449.3239441, 600.8957520, -526.5805054, 715.5024414, -1164.8264160, 1127.4760742
4: -397.1821899, 590.6504517, -462.8924561, 699.1849365, -1096.3670654, 1053.5428467

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811458, upper bound: 886.6785108
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811458, upper bound: 886.6788595
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -318.6701965, 512.8953857, -325.0014954, 521.2191772, -839.8894043, 837.8968506
1: -356.8322449, 506.6532898, -363.7823181, 516.0242920, -872.8564453, 870.4356079
2: -358.8418579, 501.6371765, -365.5518188, 510.8558960, -869.6976929, 867.1889648
3: -436.7966919, 583.9028320, -445.6395874, 594.3685913, -1031.1652832, 1029.5424805
4: -387.7786255, 574.5294189, -394.3592224, 585.1213379, -972.8998413, 968.8886108

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760389, upper bound: 886.6732517
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6752193, upper bound: 886.6736001
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -331.1015625, 531.0423584, -337.9718933, 542.0130005, -873.1145020, 869.0142822
1: -370.2244263, 525.5634155, -378.1127014, 536.6397095, -906.8641357, 903.6760864
2: -372.2037048, 520.7018433, -379.9012451, 531.6011963, -903.8048706, 900.6030884
3: -454.1714172, 605.5418091, -463.6909790, 618.2362061, -1072.4074707, 1069.2325439
4: -401.1650391, 595.8226318, -409.2712402, 608.3235474, -1009.4885864, 1005.0938721

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -350.6663513, 578.0685425, -283.3705444, 459.2656250, -809.9320068, 861.4389038
1: -392.3681335, 566.0692139, -317.0884705, 452.3088684, -844.6770020, 883.1577148
2: -394.5059814, 559.9786377, -318.9523010, 447.6842651, -842.1901245, 878.9309082
3: -481.1281128, 652.9342041, -388.8270264, 521.3934326, -1002.5215454, 1041.7611084
4: -424.4251709, 637.6273193, -344.3997192, 511.8744507, -936.2996216, 982.0270386

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6749602, upper bound: 886.6739203
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6749602, upper bound: 886.6739203
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -362.4913025, 596.5627441, -294.2640076, 476.2749939, -838.7662964, 890.8267822
1: -405.2043457, 585.2329712, -328.9698486, 469.6170959, -874.8214111, 914.2028198
2: -407.3449402, 579.3163452, -330.8330994, 464.9403992, -872.2852783, 910.1494141
3: -497.5676575, 674.8923340, -403.8551331, 541.5825195, -1039.1501465, 1078.7474365
4: -437.8743591, 659.8839722, -356.3956909, 531.5100098, -969.3841553, 1016.2796631

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6745715, upper bound: 886.6739284
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6745715, upper bound: 886.6739284
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -386.1738892, 634.6172485, -340.7220154, 546.7853394, -932.9592285, 975.3392334
1: -431.9323120, 623.4801636, -381.1342773, 541.5595703, -973.4917603, 1004.6144409
2: -433.9816895, 617.2265015, -382.9755554, 536.3522339, -970.3339233, 1000.2020264
3: -530.3025513, 718.8379517, -467.3895874, 623.9288940, -1154.2314453, 1186.2274170
4: -466.3063354, 703.1750488, -412.7691956, 613.8334351, -1080.1396484, 1115.9442139

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6744445, upper bound: 886.6739203
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6739284, upper bound: 886.6739284
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -386.1738892, 634.6172485, -384.3522949, 631.7689819, -1017.9428711, 1018.9695435
1: -431.9323120, 623.4801636, -429.8685303, 620.5733032, -1052.5056152, 1053.3486328
2: -433.9816895, 617.2265015, -431.9433289, 614.3768311, -1048.3583984, 1049.1697998
3: -530.3025513, 718.8379517, -527.7456665, 715.5344238, -1245.8369141, 1246.5833740
4: -466.3063354, 703.1750488, -464.1484070, 699.8935547, -1166.1998291, 1167.3234863

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6739102, upper bound: 886.6741685
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6739284, upper bound: 886.6739284
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -312.9841003, 502.7385864, -491.0488586, 793.8690186, -1106.8531494, 993.7873535
1: -350.5139771, 497.4879761, -553.8104858, 792.7794189, -1143.2934570, 1051.2982178
2: -352.0639648, 492.5448608, -551.9913940, 783.2709961, -1134.7015381, 1044.5360107
3: -429.6339722, 572.4817505, -680.3311768, 908.4708862, -1338.1048584, 1252.8127441
4: -380.0469666, 563.6500854, -594.7395020, 897.0834961, -1275.5793457, 1158.3896484

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800799, upper bound: 886.6798278
time: 1.07 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800799, upper bound: 886.6798278
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -316.8091125, 508.8184509, -497.8132019, 809.0231934, -1125.8322754, 1006.6315308
1: -354.6345520, 503.1039734, -560.9993286, 806.0504150, -1160.5130615, 1064.1032715
2: -356.3851318, 498.1908569, -559.7190552, 796.5993042, -1152.1442871, 1057.9099121
3: -434.6600037, 579.4716797, -689.2288818, 924.1143188, -1358.7738037, 1268.7005615
4: -384.4094849, 570.2536621, -603.2398071, 911.2624512, -1294.1315918, 1173.4934082

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797549, upper bound: 886.6798633
time: 0.98 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797549, upper bound: 886.6798633
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -330.7326965, 531.5455933, -490.3223877, 792.4829102, -1123.2153320, 1021.8679810
1: -370.0465393, 526.0419312, -553.1027222, 791.5791016, -1161.6256104, 1079.1446533
2: -371.7483826, 521.0663452, -551.1196899, 782.0870972, -1153.4227295, 1072.1860352
3: -454.0150757, 605.8285522, -679.6862183, 907.0408936, -1361.0555420, 1285.5147705
4: -400.5035400, 596.0501099, -593.6221924, 896.0225220, -1294.7965088, 1189.6723633

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737174, upper bound: 886.6662863
time: 0.83 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6721404, upper bound: 886.6663748
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -333.2594910, 535.4112549, -497.4494019, 808.3284302, -1141.5876465, 1032.8605957
1: -372.7714539, 529.6436768, -560.6849976, 805.4944458, -1178.2655029, 1090.3286133
2: -374.5974121, 524.7986450, -559.2790527, 796.0242310, -1170.1218262, 1084.0776367
3: -457.3004150, 610.2125244, -689.0695190, 923.4489136, -1380.7489014, 1299.2819824
4: -403.3079834, 600.1681519, -602.4730225, 910.9048462, -1312.6632080, 1202.6411133

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6810895, upper bound: 886.6799823
time: 0.89 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6810895, upper bound: 886.6802103
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -315.7460632, 506.8948364, -499.6952209, 810.7128906, -1126.4589844, 1006.5900879
1: -353.2734680, 501.4751892, -563.0335693, 808.7923584, -1162.0656738, 1064.5085449
2: -354.9538269, 496.4324341, -562.0794067, 799.1892700, -1153.7669678, 1058.5115967
3: -433.2565918, 577.4556885, -692.1737061, 927.3511353, -1360.6075439, 1269.6293945
4: -381.9422913, 568.4033813, -605.0819092, 914.8605957, -1295.2375488, 1173.4851074

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6788471, upper bound: 886.6792192
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6814176
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6814176
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -341.4857788, 549.2919922, -490.4579468, 796.0449219, -1137.5306396, 1039.7500000
1: -381.8196716, 543.5779419, -552.6120605, 793.8352661, -1175.6547852, 1096.1899414
2: -383.8876038, 537.8378296, -551.6123657, 784.1390991, -1168.0267334, 1089.4500732
3: -468.2194824, 627.1686401, -679.4325562, 910.2442017, -1378.4636230, 1306.6011963
4: -412.6742249, 615.6817017, -593.5498047, 897.7910767, -1309.0794678, 1209.2313232

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6818324
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6811649
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -327.7250061, 525.7000122, -501.6998901, 813.3760376, -1141.1010742, 1027.3999023
1: -366.5014648, 520.1251831, -565.0808105, 811.0997314, -1177.6010742, 1085.2060547
2: -368.1830750, 515.1625977, -564.0930786, 801.6339722, -1169.6511230, 1079.2556152
3: -449.8151855, 599.1945801, -695.0361328, 930.1635742, -1379.9787598, 1294.2307129
4: -395.6330261, 589.6274414, -606.3902588, 918.0571289, -1312.0827637, 1196.0174561

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6795964
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6796979
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -352.3258972, 565.6660767, -493.7385254, 800.9507446, -1153.2766113, 1059.4045410
1: -393.8340454, 559.9351196, -556.1115723, 798.7612305, -1192.5952148, 1116.0466309
2: -395.8488770, 554.6286011, -555.0532227, 789.2916260, -1185.1405029, 1109.6815186
3: -483.3472595, 645.9515381, -684.2304688, 915.9234009, -1399.2703857, 1330.1818848
4: -425.1168213, 634.7530518, -596.5521240, 903.8897705, -1327.5268555, 1231.3051758

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6794889, upper bound: 886.6801176
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6794889, upper bound: 886.6800307
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -491.0488586, 793.8690186, -312.9841003, 502.7385864, -993.7873535, 1106.8531494
1: -553.8104858, 792.7794189, -350.5139771, 497.4879761, -1051.2983398, 1143.2934570
2: -551.9913940, 783.2709961, -352.0639648, 492.5448608, -1044.5360107, 1134.7015381
3: -680.3311768, 908.4708862, -429.6339722, 572.4817505, -1252.8127441, 1338.1048584
4: -594.7395020, 897.0834961, -380.0469666, 563.6500854, -1158.3896484, 1275.5792236

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798278, upper bound: 886.6800799
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798278, upper bound: 886.6814195
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -497.8132019, 809.0231934, -316.8091125, 508.8184509, -1006.6315308, 1125.8322754
1: -560.9993286, 806.0504150, -354.6345520, 503.1039734, -1064.1032715, 1160.5130615
2: -559.7190552, 796.5993042, -356.3851318, 498.1908569, -1057.9099121, 1152.1442871
3: -689.2288818, 924.1143188, -434.6600037, 579.4716797, -1268.7005615, 1358.7738037
4: -603.2398071, 911.2624512, -384.4094849, 570.2536621, -1173.4934082, 1294.1315918

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6797549
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6810945
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -490.3223877, 792.4829102, -330.7326965, 531.5455933, -1021.8679810, 1123.2153320
1: -553.1027222, 791.5791016, -370.0465393, 526.0419312, -1079.1446533, 1161.6256104
2: -551.1196899, 782.0870972, -371.7483826, 521.0663452, -1072.1860352, 1153.4227295
3: -679.6862183, 907.0408936, -454.0150757, 605.8285522, -1285.5147705, 1361.0556641
4: -593.6221924, 896.0225220, -400.5035400, 596.0501099, -1189.6723633, 1294.7965088

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6662863, upper bound: 886.6737174
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6663748, upper bound: 886.6721404
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -497.4494019, 808.3284302, -333.2594910, 535.4112549, -1032.8605957, 1141.5877686
1: -560.6849976, 805.4944458, -372.7714539, 529.6436768, -1090.3286133, 1178.2656250
2: -559.2790527, 796.0242310, -374.5974121, 524.7986450, -1084.0776367, 1170.1219482
3: -689.0695190, 923.4489136, -457.3004150, 610.2125244, -1299.2819824, 1380.7489014
4: -602.4730225, 910.9048462, -403.3079834, 600.1681519, -1202.6409912, 1312.6632080

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6812976
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6812995
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -499.6952209, 810.7128906, -315.7460632, 506.8948364, -1006.5900879, 1126.4589844
1: -563.0335693, 808.7923584, -353.2734680, 501.4751892, -1064.5085449, 1162.0656738
2: -562.0794067, 799.1892700, -354.9538269, 496.4324341, -1058.5115967, 1153.7670898
3: -692.1737061, 927.3511353, -433.2565918, 577.4556885, -1269.6293945, 1360.6075439
4: -605.0819092, 914.8605957, -381.9422913, 568.4033813, -1173.4851074, 1295.2374268

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6792192, upper bound: 886.6788471
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6805319
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6810732
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -490.4579468, 796.0449219, -341.4857788, 549.2919922, -1039.7500000, 1137.5306396
1: -552.6120605, 793.8352661, -381.8196716, 543.5779419, -1096.1899414, 1175.6547852
2: -551.6123657, 784.1390991, -383.8876038, 537.8378296, -1089.4500732, 1168.0267334
3: -679.4325562, 910.2442017, -468.2194824, 627.1686401, -1306.6011963, 1378.4636230
4: -593.5498047, 897.7910767, -412.6742249, 615.6817017, -1209.2313232, 1309.0794678

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6796754
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6803040
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -501.6998901, 813.3760376, -327.7250061, 525.7000122, -1027.3999023, 1141.1010742
1: -565.0808105, 811.0997314, -366.5014648, 520.1251831, -1085.2060547, 1177.6010742
2: -564.0930786, 801.6339722, -368.1830750, 515.1625977, -1079.2556152, 1169.6511230
3: -695.0361328, 930.1635742, -449.8151855, 599.1945801, -1294.2307129, 1379.9787598
4: -606.3902588, 918.0571289, -395.6330261, 589.6274414, -1196.0173340, 1312.0827637

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -493.7385254, 800.9507446, -352.3258972, 565.6660767, -1059.4045410, 1153.2766113
1: -556.1115723, 798.7612305, -393.8340454, 559.9351196, -1116.0466309, 1192.5952148
2: -555.0532227, 789.2916260, -395.8488770, 554.6286011, -1109.6815186, 1185.1405029
3: -684.2304688, 915.9234009, -483.3472595, 645.9515381, -1330.1817627, 1399.2703857
4: -596.5521240, 903.8897705, -425.1168213, 634.7530518, -1231.3051758, 1327.5268555

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6794889
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6795550
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -503.6767883, 817.1486816, -503.4063721, 816.5255737, -1320.2020264, 1320.5548096
1: -567.5106201, 815.2822266, -567.2109375, 814.8658447, -1380.5051270, 1380.4227295
2: -566.5572510, 805.6655273, -566.1912842, 805.2928467, -1370.1816406, 1370.0704346
3: -697.6441040, 934.7715454, -697.4225464, 934.2309570, -1630.4804688, 1630.5623779
4: -609.9658203, 922.1118164, -609.4550781, 921.6151733, -1529.0914307, 1528.9965820

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6784086
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6800963
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -507.2889404, 823.2482300, -505.8939209, 821.0324707, -1328.3214111, 1329.1420898
1: -571.5657349, 821.2772217, -570.0043335, 819.0781250, -1388.8524170, 1389.4302979
2: -570.6337280, 811.5574341, -569.0660400, 809.3848877, -1378.3822021, 1378.9559326
3: -702.6406860, 941.6658936, -700.7531738, 939.1489258, -1640.4755859, 1641.0350342
4: -614.3399658, 928.8489380, -612.7015991, 926.3571167, -1538.2208252, 1539.0634766

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800963, upper bound: 886.6801339
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800963, upper bound: 886.6819570
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -505.8763428, 820.1217041, -503.4063721, 816.5255737, -1322.4014893, 1323.3963623
1: -569.8047485, 818.0728760, -567.2109375, 814.8658447, -1382.8018799, 1383.0648193
2: -568.8123779, 808.5827637, -566.1912842, 805.2928467, -1372.4711914, 1372.8728027
3: -700.8424683, 938.1273804, -697.4225464, 934.2309570, -1633.7093506, 1633.7873535
4: -611.6430054, 925.8558350, -609.4550781, 921.6151733, -1530.6434326, 1532.6422119

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5658204, upper bound: 886.6247058
time: 1.02 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5658204, upper bound: 886.6801117
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -509.5733948, 826.4197998, -505.8939209, 821.0324707, -1330.6057129, 1332.3135986
1: -573.9484253, 824.2539062, -570.0043335, 819.0781250, -1391.2335205, 1392.2646484
2: -572.9907227, 814.6523438, -569.0660400, 809.3848877, -1380.7727051, 1381.9376221
3: -705.9576416, 945.2511597, -700.7531738, 939.1489258, -1643.8173828, 1644.5012207
4: -616.1548462, 932.7806396, -612.7015991, 926.3571167, -1539.9135742, 1542.8936768

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5670915, upper bound: 886.6385017
time: 1.14 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5670915, upper bound: 886.6824467
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -505.8939209, 821.0324707, -509.5733948, 826.4197998, -1332.3135986, 1330.6058350
1: -570.0043335, 819.0781250, -573.9484253, 824.2539062, -1392.2645264, 1391.2335205
2: -569.0660400, 809.3848877, -572.9907227, 814.6523438, -1381.9377441, 1380.7727051
3: -700.7531738, 939.1489258, -705.9576416, 945.2511597, -1644.5013428, 1643.8173828
4: -612.7015991, 926.3571167, -616.1548462, 932.7806396, -1542.8936768, 1539.9134521

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5758116, upper bound: 886.5638213
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5758116, upper bound: 886.6801758
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -509.5733948, 826.4197998, -508.0707092, 824.0618896, -1333.6352539, 1334.4904785
1: -573.9484253, 824.2539062, -572.2630005, 821.9155273, -1394.5632324, 1395.1519775
2: -572.9907227, 814.6523438, -571.3037109, 812.3393555, -1384.1250000, 1384.7149658
3: -705.9576416, 945.2511597, -703.9230347, 942.5779419, -1647.8890381, 1648.4537354
4: -616.1548462, 932.7806396, -614.3991089, 930.1170044, -1543.8455811, 1544.7396240

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5669172, upper bound: 886.6379556
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5669172, upper bound: 886.6809272
time: 0.95 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.49 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6765683, upper bound: 886.6739102
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6775239, upper bound: 886.6741832
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6811458, upper bound: 886.6785108
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6811458, upper bound: 886.6788595
NS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6760389, upper bound: 886.6732517
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6752193, upper bound: 886.6736001
NS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
NS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6749602, upper bound: 886.6739203
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6749602, upper bound: 886.6739203
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6745715, upper bound: 886.6739284
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6745715, upper bound: 886.6739284
NS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6744445, upper bound: 886.6739203
NS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6739284, upper bound: 886.6739284
NS_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6739102, upper bound: 886.6741685
NS_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6739284, upper bound: 886.6739284
NS_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6800799, upper bound: 886.6798278
NS_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6800799, upper bound: 886.6798278
NS_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6797549, upper bound: 886.6798633
NS_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6797549, upper bound: 886.6798633
NS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6737174, upper bound: 886.6662863
NS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6721404, upper bound: 886.6663748
NS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6810895, upper bound: 886.6799823
NS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6810895, upper bound: 886.6802103
NS_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6814176
NS_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6814176
NS_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6818324
NS_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6811649
NS_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6795964
NS_A1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6796979
NS_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6794889, upper bound: 886.6801176
NS_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6794889, upper bound: 886.6800307
NS_A2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6798278, upper bound: 886.6800799
NS_A2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6798278, upper bound: 886.6814195
NS_A2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6797549
NS_A2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6798633, upper bound: 886.6810945
NS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6662863, upper bound: 886.6737174
NS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6663748, upper bound: 886.6721404
NS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6812976
NS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6812995
NS_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6805319
NS_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6810732
NS_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6796754
NS_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6803040
NS_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
NS_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
NS_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6794889
NS_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6795550
NS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6784086
NS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6800963
NS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6800963, upper bound: 886.6801339
NS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.6800963, upper bound: 886.6819570
NS_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5658204, upper bound: 886.6247058
NS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5658204, upper bound: 886.6801117
NS_A2_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5670915, upper bound: 886.6385017
NS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5670915, upper bound: 886.6824467
NS_A2_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5758116, upper bound: 886.5638213
NS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5758116, upper bound: 886.6801758
NS_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5669172, upper bound: 886.6379556
NS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.49
Output dim: 0, lower bound: -886.5669172, upper bound: 886.6809272

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -305.6571960, 492.8957520, -281.6208496, 455.4470520, -761.1042480, 774.5166016
1: -342.0991516, 485.9771423, -313.9395752, 448.1112976, -790.2104492, 799.9167480
2: -344.2719421, 481.3521729, -316.8759155, 443.6770935, -787.9490356, 798.2280884
3: -418.7088623, 560.0913086, -384.8109131, 517.1845703, -935.8933716, 944.9020386
4: -372.1111145, 550.9262085, -341.3342590, 507.3933716, -879.5042725, 892.2603149

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6763972, upper bound: 886.6749722
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6759325, upper bound: 886.6753164
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -319.3155823, 513.4044800, -291.3801270, 470.9902344, -790.3057861, 804.7846069
1: -356.7502136, 507.3784485, -324.6842041, 464.0549622, -820.8051758, 832.0626221
2: -358.9671936, 502.7969360, -327.6102600, 459.6798706, -818.6470947, 830.4072266
3: -437.7598572, 584.7794189, -398.6688843, 535.5150757, -973.2749023, 983.4483032
4: -386.9197693, 574.9425659, -352.1370239, 525.2665405, -912.1862793, 927.0795898

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6774499, upper bound: 886.6752000
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6774499, upper bound: 886.6755502
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -310.0726013, 499.0182495, -377.4266968, 623.3227539, -932.9012451, 876.4448853
1: -346.2801819, 493.0499573, -421.3950806, 609.9562378, -956.2363281, 914.4450684
2: -348.3768005, 488.3540344, -423.9265747, 604.2958984, -952.6727295, 912.2806396
3: -425.0463562, 568.2976685, -517.9862061, 704.3002319, -1129.3465576, 1086.2839355
4: -374.8595581, 558.6082153, -455.3646851, 687.9597778, -1062.8190918, 1013.9727783

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6805371, upper bound: 886.6784033
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6804023, upper bound: 886.6768803
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -336.4377747, 541.7335815, -375.5356140, 620.3563232, -956.5253906, 917.2691650
1: -375.7515259, 535.8486938, -419.3980103, 607.2867432, -983.0382690, 955.2467041
2: -378.0438232, 530.6969604, -421.7554932, 601.6444702, -979.6882935, 952.4524536
3: -461.1686401, 618.4282227, -515.7632446, 701.1072998, -1162.2758789, 1134.1914062
4: -406.7354736, 607.0690308, -452.8093567, 684.9269409, -1091.6623535, 1059.8782959

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6713990, upper bound: 886.6650162
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6714652, upper bound: 886.6652050
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -302.1397705, 486.7652283, -314.3872070, 504.3781433, -806.5178833, 801.1523438
1: -338.0361328, 480.5949402, -351.6335754, 499.0347595, -837.0708618, 832.2285156
2: -340.0501099, 475.7366943, -353.4835205, 494.0427856, -834.0928955, 829.2200928
3: -414.0634766, 553.7343750, -430.8548889, 574.7666626, -988.8301392, 984.5892334
4: -366.7779541, 544.7932129, -380.8126221, 565.8002930, -932.5782471, 925.6057739

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760303, upper bound: 886.6731193
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760303, upper bound: 886.6732517
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -323.0477600, 521.3051147, -319.3325806, 511.8192444, -834.8670044, 840.6375732
1: -361.2111511, 514.6909180, -357.2866516, 506.6132507, -867.8244019, 871.9775391
2: -363.5712891, 509.2450562, -359.1137390, 501.5347290, -865.1060181, 868.3587646
3: -442.4188538, 594.4000854, -437.6814270, 583.5252075, -1025.9440918, 1032.0814209
4: -391.9054871, 583.1520996, -387.3074036, 574.3518066, -966.2573242, 970.4594727

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6752111, upper bound: 886.6734683
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6752111, upper bound: 886.6736001
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -286.4380798, 463.0567017, -337.9718933, 542.0130005, -828.4510498, 801.0285645
1: -319.1374207, 456.0961914, -378.1127014, 536.6397095, -855.7770996, 834.2088623
2: -322.0881653, 451.8367615, -379.9012451, 531.6011963, -853.6893311, 831.7380371
3: -391.9034729, 526.3308716, -463.6909790, 618.2362061, -1010.1396484, 990.0217896
4: -346.1516724, 516.2714844, -409.2712402, 608.3235474, -954.4751587, 925.5427246

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -330.2913513, 529.6994629, -337.9718933, 542.0130005, -872.3043213, 867.6713867
1: -369.3267822, 524.2376709, -378.1127014, 536.6397095, -905.9663696, 902.3503418
2: -371.2991333, 519.3915405, -379.9012451, 531.6011963, -902.9002686, 899.2927856
3: -453.0569458, 604.0022583, -463.6909790, 618.2362061, -1071.2929688, 1067.6931152
4: -400.2043152, 594.3149414, -409.2712402, 608.3235474, -1008.5278320, 1003.5861816

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -350.6663513, 578.0685425, -281.6208496, 455.4470520, -806.1133423, 859.6893921
1: -392.3681335, 566.0692139, -313.9395752, 448.1112976, -840.4794312, 880.0087891
2: -394.5059814, 559.9786377, -316.8759155, 443.6770935, -838.1830444, 876.8545532
3: -481.1281128, 652.9342041, -384.8109131, 517.1845703, -998.3126831, 1037.7451172
4: -424.4251709, 637.6273193, -341.3342590, 507.3933716, -931.8185425, 978.9615479

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6748714, upper bound: 886.6713139
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6747658, upper bound: 886.6738555
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -350.6663513, 578.0685425, -367.0920105, 605.9075928, -956.1402588, 945.1602783
1: -392.3681335, 566.0692139, -410.1111450, 592.4345703, -984.8027344, 976.1802979
2: -394.5059814, 559.9786377, -412.4617615, 586.7765503, -981.2824707, 972.4404297
3: -481.1281128, 652.9342041, -503.8140564, 683.9164429, -1165.0443115, 1156.7481689
4: -424.4251709, 637.6273193, -443.3889160, 668.1072388, -1092.5324707, 1081.0162354

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6746850, upper bound: 886.6737973
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6747658, upper bound: 886.6738555
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -362.4913025, 596.5627441, -291.3801270, 470.9902344, -833.4815674, 887.9428711
1: -405.2043457, 585.2329712, -324.6842041, 464.0549622, -869.2592163, 909.9171753
2: -407.3449402, 579.3163452, -327.6102600, 459.6798706, -867.0247803, 906.9266357
3: -497.5676575, 674.8923340, -398.6688843, 535.5150757, -1033.0827637, 1073.5609131
4: -437.8743591, 659.8839722, -352.1370239, 525.2665405, -963.1408081, 1012.0209961

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6685741, upper bound: 886.6640324
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6744306, upper bound: 886.6726632
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731741, upper bound: 886.6725228
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -362.4913025, 596.5627441, -373.9876709, 617.3334961, -979.8248291, 970.5502930
1: -405.2043457, 585.2329712, -417.7424927, 604.1917725, -1009.3961182, 1002.9754639
2: -407.3449402, 579.3163452, -420.1463928, 598.7049561, -1006.0499268, 999.4627686
3: -497.5676575, 674.8923340, -513.4816895, 697.4212646, -1194.9888916, 1188.3737793
4: -437.8743591, 659.8839722, -451.5110474, 681.7132568, -1119.5876465, 1111.3946533

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6685741, upper bound: 886.6640324
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6728193, upper bound: 886.6697162
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6745715, upper bound: 886.6739284
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6741534, upper bound: 886.6739284
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -360.9342957, 593.8043213, -321.4933167, 515.6528931, -876.5870972, 915.0379028
1: -403.9343567, 581.9702759, -359.7061157, 510.4137878, -914.3480835, 941.6763916
2: -406.0740967, 575.7664185, -361.6218872, 505.2769165, -911.3510132, 937.3883057
3: -495.2354736, 671.1785889, -440.5453186, 588.0465088, -1083.2819824, 1111.7235107
4: -436.7272034, 655.9875488, -390.3047485, 578.8289795, -1015.5561523, 1046.2921143

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6738791, upper bound: 886.6768346
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6743351, upper bound: 886.6768346
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -372.7318420, 611.9431152, -334.3929138, 536.3737793, -909.1055908, 946.3360596
1: -416.7958069, 600.8573608, -373.9689636, 530.9948120, -947.7906494, 974.8262939
2: -418.9109497, 594.9185791, -375.8876953, 526.0015869, -944.9124146, 970.8062744
3: -511.7256165, 692.7358398, -458.6827393, 611.7823486, -1123.5079346, 1151.4184570
4: -450.2460938, 678.1066895, -405.1336365, 601.9384155, -1052.1845703, 1083.2401123

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6734408, upper bound: 886.6768426
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6738853, upper bound: 886.6768426
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -368.7207947, 606.1630249, -359.1934509, 591.0895996, -959.8104248, 965.3564453
1: -412.4241638, 594.4722900, -401.9608459, 579.1860352, -991.6102295, 996.4329224
2: -414.5043640, 588.3934326, -404.1271973, 573.0301514, -987.5345459, 992.5206299
3: -506.0841064, 685.7263794, -492.7873535, 668.0111694, -1174.0952148, 1178.5135498
4: -445.4825134, 670.3585815, -434.6558533, 652.8358154, -1098.3183594, 1105.0141602

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737463, upper bound: 886.6737513
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737463, upper bound: 886.6737594
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -378.4546509, 621.6447144, -370.6227112, 608.6351318, -987.0897827, 992.2674561
1: -423.2255859, 610.5181274, -414.4149780, 597.4848633, -1020.7104492, 1024.9331055
2: -425.3326111, 604.4548950, -416.5511169, 591.6019897, -1016.9345703, 1021.0059814
3: -519.6141968, 703.9040527, -508.7724915, 688.8926392, -1208.5068359, 1212.6765137
4: -457.0979919, 688.8024902, -447.7651672, 674.2894897, -1131.3874512, 1136.5676270

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737594, upper bound: 886.6739203
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737594, upper bound: 886.6739284
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -287.0137024, 463.8797913, -491.0488586, 793.8690186, -1080.8826904, 954.9286499
1: -320.7918701, 457.7766418, -553.8104858, 792.7794189, -1113.5609131, 1011.5871582
2: -322.8685303, 453.1473083, -551.9913940, 783.2709961, -1105.4542236, 1005.1386719
3: -393.4365845, 527.2431030, -680.3311768, 908.4708862, -1301.9074707, 1207.5742188
4: -348.1585999, 517.9266357, -594.7395020, 897.0834961, -1243.5664062, 1112.6661377

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6796103
time: 1.06 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6798278
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -324.5570068, 520.0396729, -491.0488586, 793.8690186, -1118.4260254, 1011.0884399
1: -363.6609497, 515.3393555, -553.8104858, 792.7794189, -1156.2750244, 1069.1497803
2: -364.9896851, 510.1193542, -551.9913940, 783.2709961, -1147.3979492, 1062.1105957
3: -445.7495728, 592.9848633, -680.3311768, 908.4708862, -1354.2204590, 1273.3160400
4: -393.8153381, 584.2042236, -594.7395020, 897.0834961, -1288.6578369, 1178.9437256

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6796103
time: 0.94 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6798278
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -293.6987915, 475.4500732, -497.8132019, 809.0231934, -1102.7219238, 973.2631226
1: -328.0613708, 467.9395752, -560.9993286, 806.0504150, -1133.8408203, 1028.9387207
2: -330.4018860, 463.3641052, -559.7190552, 796.5993042, -1126.1365967, 1023.0831299
3: -402.3458862, 539.9793701, -689.2288818, 924.1143188, -1326.4597168, 1229.2082520
4: -355.9644470, 529.7532349, -603.2398071, 911.2624512, -1265.4920654, 1132.9930420

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6796430
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6798633
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -329.2723389, 527.7182007, -497.8132019, 809.0231934, -1138.2955322, 1025.5313721
1: -368.7728882, 522.5643921, -560.9993286, 806.0504150, -1174.3613281, 1083.5635986
2: -370.3352356, 517.3683472, -559.7190552, 796.5993042, -1165.8465576, 1077.0874023
3: -451.9827576, 601.8367920, -689.2288818, 924.1143188, -1376.0966797, 1291.0656738
4: -399.2688904, 592.6180420, -603.2398071, 911.2624512, -1308.2539062, 1195.8577881

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6796430
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6798633
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -320.1048279, 514.5168457, -483.4086914, 781.3688354, -1101.4736328, 997.9253540
1: -358.0197754, 508.9039001, -545.2546387, 780.3468018, -1138.3665771, 1054.1583252
2: -359.8383484, 504.1305542, -543.3615112, 771.0276489, -1130.3192139, 1047.4919434
3: -439.1643677, 586.2185059, -670.0274048, 894.2095337, -1333.3737793, 1256.2458496
4: -387.8655701, 576.6268921, -585.3394775, 883.2456665, -1269.1374512, 1161.9663086

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6707133, upper bound: 886.6646671
time: 0.93 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6705179, upper bound: 886.6634094
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -323.7322998, 520.4692993, -485.3543701, 784.5611572, -1108.2934570, 1005.8236694
1: -362.0963440, 514.7459717, -547.5022583, 783.6175537, -1145.7137451, 1062.2482910
2: -363.9067078, 510.0528259, -545.5332031, 774.2023926, -1137.7338867, 1055.5860596
3: -444.2946472, 593.0725708, -672.8260498, 897.9596558, -1342.2542725, 1265.8986816
4: -392.2029419, 583.0673828, -587.7595825, 886.9486084, -1277.5610352, 1170.8269043

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706083, upper bound: 886.6645599
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6709315, upper bound: 886.6662500
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706892, upper bound: 886.6645679
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -316.9536438, 511.2012024, -497.4494019, 808.3284302, -1125.2818604, 1008.6505737
1: -355.0775146, 504.2756348, -560.6849976, 805.4944458, -1160.3339844, 1064.9606934
2: -356.9686279, 499.4333191, -559.2790527, 796.0242310, -1151.8353271, 1058.7124023
3: -434.9897461, 580.9922485, -689.0695190, 923.4489136, -1358.4379883, 1270.0616455
4: -385.4715576, 571.4299316, -602.4730225, 910.9048462, -1294.8006592, 1173.9029541

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6779175, upper bound: 886.6775377
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6799823
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6799823
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -330.3445435, 530.6880493, -497.4494019, 808.3284302, -1138.6727295, 1028.1374512
1: -369.4373474, 524.8557739, -560.6849976, 805.4944458, -1174.9316406, 1085.5407715
2: -371.3348694, 520.1254272, -559.2790527, 796.0242310, -1166.8544922, 1079.4045410
3: -453.2719116, 604.7229004, -689.0695190, 923.4489136, -1376.7205811, 1293.7923584
4: -399.7754211, 594.7953491, -602.4730225, 910.9048462, -1309.1535645, 1197.2683105

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6779175, upper bound: 886.6781667
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6802103
time: 2.04 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6802103
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -315.7460632, 506.8948364, -497.2741089, 806.7100220, -1122.4560547, 1004.1689453
1: -353.2734680, 501.4751892, -560.3112183, 804.7674561, -1158.0407715, 1061.7862549
2: -354.9538269, 496.4324341, -559.3439331, 795.2078247, -1149.6677246, 1055.7763672
3: -433.2565918, 577.4556885, -688.8338623, 922.7393188, -1355.9958496, 1266.2895508
4: -381.9422913, 568.4033813, -602.0997925, 910.3527832, -1290.6510010, 1170.5031738

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6808126
time: 1.06 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6804912, upper bound: 886.6814176
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -315.7460632, 506.8948364, -493.9764404, 802.9646606, -1118.7106934, 1000.8712769
1: -353.2734680, 501.4751892, -556.4440918, 800.7713013, -1153.6347656, 1057.9191895
2: -354.9538269, 496.4324341, -555.3469238, 790.6830444, -1144.7628174, 1051.7792969
3: -433.2565918, 577.4556885, -684.2299194, 918.3331299, -1351.5895996, 1261.6855469
4: -381.9422913, 568.4033813, -597.0147705, 904.6518555, -1284.8223877, 1165.4182129

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6808126
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6804912, upper bound: 886.6814176
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -311.0900574, 505.7402039, -490.4579468, 796.0449219, -1107.1347656, 996.1981201
1: -347.0505066, 498.4164124, -552.6120605, 793.8352661, -1140.8856201, 1051.0283203
2: -349.7124329, 493.5307007, -551.6123657, 784.1390991, -1133.8515625, 1045.1430664
3: -426.1267700, 575.2863770, -679.4325562, 910.2442017, -1336.3709717, 1254.7189941
4: -375.4928284, 563.0198975, -593.5498047, 897.7910767, -1272.2751465, 1156.5694580

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6812275
time: 0.81 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6795172, upper bound: 886.6818324
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -340.7877808, 548.1430054, -490.4579468, 796.0449219, -1136.8325195, 1038.6009521
1: -381.0401001, 542.4392090, -552.6120605, 793.8352661, -1174.8751221, 1095.0512695
2: -383.1042786, 536.7133179, -551.6123657, 784.1390991, -1167.2434082, 1088.3256836
3: -467.2512512, 625.8507690, -679.4325562, 910.2442017, -1377.4954834, 1305.2832031
4: -411.8325806, 614.3975220, -593.5498047, 897.7910767, -1308.0997314, 1207.9468994

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6808197
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6795172, upper bound: 886.6811649
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -311.6872559, 502.6691284, -501.6998901, 813.3760376, -1125.0632324, 1004.3690186
1: -349.1390686, 496.2806702, -565.0808105, 811.0997314, -1160.1511230, 1061.3614502
2: -350.8332520, 491.2427063, -564.0930786, 801.6339722, -1151.7058105, 1055.3358154
3: -428.0234680, 571.6222534, -695.0361328, 930.1635742, -1358.1870117, 1266.6583252
4: -378.1982727, 562.2469482, -606.3902588, 918.0571289, -1294.5363770, 1168.6372070

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6776601, upper bound: 886.6777595
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6807379, upper bound: 886.6787649
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6795964
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -323.6772766, 519.1259766, -501.6998901, 813.3760376, -1137.0533447, 1020.8258057
1: -361.9174500, 513.4822998, -565.0808105, 811.0997314, -1173.0172119, 1078.5631104
2: -363.6483459, 508.6350098, -564.0930786, 801.6339722, -1165.1007080, 1072.7280273
3: -444.2401123, 591.5661011, -695.0361328, 930.1635742, -1374.4035645, 1286.6022949
4: -390.8212585, 582.1261597, -606.3902588, 918.0571289, -1307.2832031, 1188.5162354

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6776601, upper bound: 886.6780159
time: 0.81 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6807379, upper bound: 886.6788609
time: 0.83 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6796979
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -318.3107300, 516.1080933, -493.7385254, 800.9507446, -1119.2612305, 1009.8466187
1: -355.0302124, 508.9057617, -556.1115723, 798.7612305, -1153.7915039, 1065.0172119
2: -357.7380066, 504.3653870, -555.0532227, 789.2916260, -1147.0296631, 1059.4185791
3: -436.2491150, 587.3682861, -684.2304688, 915.9234009, -1352.1722412, 1271.5987549
4: -383.7341309, 575.3532715, -596.5521240, 903.8897705, -1286.6475830, 1171.9053955

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793451, upper bound: 886.6792792
time: 0.85 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6794889, upper bound: 886.6801176
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -351.2597656, 563.8720703, -493.7385254, 800.9507446, -1152.2104492, 1057.6105957
1: -392.6514282, 558.1578369, -556.1115723, 798.7612305, -1191.4125977, 1114.2694092
2: -394.6559753, 552.8757935, -555.0532227, 789.2916260, -1183.9476318, 1107.9285889
3: -481.8740540, 643.9133301, -684.2304688, 915.9234009, -1397.7972412, 1328.1437988
4: -423.8395081, 632.7755127, -596.5521240, 903.8897705, -1326.1168213, 1229.3276367

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6790991, upper bound: 886.6798494
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6790991, upper bound: 886.6800307
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -491.0488586, 793.8690186, -287.0137024, 463.8797913, -954.9286499, 1080.8826904
1: -553.8104858, 792.7794189, -320.7918701, 457.7766418, -1011.5871582, 1113.5609131
2: -551.9913940, 783.2709961, -322.8685303, 453.1473083, -1005.1386719, 1105.4541016
3: -680.3311768, 908.4708862, -393.4365845, 527.2431030, -1207.5742188, 1301.9074707
4: -594.7395020, 897.0834961, -348.1585999, 517.9266357, -1112.6661377, 1243.5666504

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6796846
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6800799
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -491.0488586, 793.8690186, -324.5570068, 520.0396729, -1011.0884399, 1118.4260254
1: -553.8104858, 792.7794189, -363.6609497, 515.3393555, -1069.1497803, 1156.2750244
2: -551.9913940, 783.2709961, -364.9896851, 510.1193542, -1062.1105957, 1147.3979492
3: -680.3311768, 908.4708862, -445.7495728, 592.9848633, -1273.3160400, 1354.2204590
4: -594.7395020, 897.0834961, -393.8153381, 584.2042236, -1178.9437256, 1288.6579590

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6814145
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6814195
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -497.8132019, 809.0231934, -293.6987915, 475.4500732, -973.2631226, 1102.7219238
1: -560.9993286, 806.0504150, -328.0613708, 467.9395752, -1028.9388428, 1133.8406982
2: -559.7190552, 796.5993042, -330.4018860, 463.3641052, -1023.0831299, 1126.1365967
3: -689.2288818, 924.1143188, -402.3458862, 539.9793701, -1229.2082520, 1326.4598389
4: -603.2398071, 911.2624512, -355.9644470, 529.7532349, -1132.9930420, 1265.4920654

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6793621
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6797549
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -497.8132019, 809.0231934, -329.2723389, 527.7182007, -1025.5313721, 1138.2955322
1: -560.9993286, 806.0504150, -368.7728882, 522.5643921, -1083.5634766, 1174.3613281
2: -559.7190552, 796.5993042, -370.3352356, 517.3683472, -1077.0874023, 1165.8465576
3: -689.2288818, 924.1143188, -451.9827576, 601.8367920, -1291.0656738, 1376.0966797
4: -603.2398071, 911.2624512, -399.2688904, 592.6180420, -1195.8579102, 1308.2539062

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6810895
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6810945
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -483.4086914, 781.3688354, -320.1048279, 514.5168457, -997.9254150, 1101.4736328
1: -545.2546387, 780.3468018, -358.0197754, 508.9039001, -1054.1583252, 1138.3665771
2: -543.3615112, 771.0276489, -359.8383484, 504.1305542, -1047.4918213, 1130.3193359
3: -670.0274048, 894.2095337, -439.1643677, 586.2185059, -1256.2458496, 1333.3736572
4: -585.3394775, 883.2456665, -387.8655701, 576.6268921, -1161.9663086, 1269.1374512

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6662628, upper bound: 886.6737174
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6662628, upper bound: 886.6737174
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -485.3543701, 784.5611572, -323.7322998, 520.4692993, -1005.8236694, 1108.2934570
1: -547.5022583, 783.6175537, -362.0963440, 514.7459717, -1062.2481689, 1145.7137451
2: -545.5332031, 774.2023926, -363.9067078, 510.0528259, -1055.5860596, 1137.7338867
3: -672.8260498, 897.9596558, -444.2946472, 593.0725708, -1265.8986816, 1342.2542725
4: -587.7595825, 886.9486084, -392.2029419, 583.0673828, -1170.8269043, 1277.5609131

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6645599, upper bound: 886.6706083
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_A2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6633574, upper bound: 886.6704153
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -497.4494019, 808.3284302, -316.9536438, 511.2012024, -1008.6505737, 1125.2818604
1: -560.6849976, 805.4944458, -355.0775146, 504.2756348, -1064.9606934, 1160.3339844
2: -559.2790527, 796.0242310, -356.9686279, 499.4333191, -1058.7124023, 1151.8354492
3: -689.0695190, 923.4489136, -434.9897461, 580.9922485, -1270.0616455, 1358.4379883
4: -602.4730225, 910.9048462, -385.4715576, 571.4299316, -1173.9029541, 1294.8006592

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6775377, upper bound: 886.6780606
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6795426
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6812976
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -497.4494019, 808.3284302, -330.3445435, 530.6880493, -1028.1374512, 1138.6727295
1: -560.6849976, 805.4944458, -369.4373474, 524.8557739, -1085.5407715, 1174.9317627
2: -559.2790527, 796.0242310, -371.3348694, 520.1254272, -1079.4044189, 1166.8544922
3: -689.0695190, 923.4489136, -453.2719116, 604.7229004, -1293.7923584, 1376.7205811
4: -602.4730225, 910.9048462, -399.7754211, 594.7953491, -1197.2683105, 1309.1535645

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6771402, upper bound: 886.6790723
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6799485
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6812995
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -497.2741089, 806.7100220, -315.7460632, 506.8948364, -1004.1689453, 1122.4560547
1: -560.3112183, 804.7674561, -353.2734680, 501.4751892, -1061.7862549, 1158.0407715
2: -559.3439331, 795.2078247, -354.9538269, 496.4324341, -1055.7763672, 1149.6677246
3: -688.8338623, 922.7393188, -433.2565918, 577.4556885, -1266.2895508, 1355.9957275
4: -602.0997925, 910.3527832, -381.9422913, 568.4033813, -1170.5031738, 1290.6510010

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808126, upper bound: 886.6805319
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6804912
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -493.9764404, 802.9646606, -315.7460632, 506.8948364, -1000.8712769, 1118.7106934
1: -556.4440918, 800.7713013, -353.2734680, 501.4751892, -1057.9193115, 1153.6347656
2: -555.3469238, 790.6830444, -354.9538269, 496.4324341, -1051.7792969, 1144.7629395
3: -684.2299194, 918.3331299, -433.2565918, 577.4556885, -1261.6855469, 1351.5895996
4: -597.0147705, 904.6518555, -381.9422913, 568.4033813, -1165.4180908, 1284.8223877

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808126, upper bound: 886.6810732
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6809150
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -490.4579468, 796.0449219, -311.0900574, 505.7402039, -996.1981201, 1107.1348877
1: -552.6120605, 793.8352661, -347.0505066, 498.4164124, -1051.0283203, 1140.8856201
2: -551.6123657, 784.1390991, -349.7124329, 493.5307007, -1045.1430664, 1133.8514404
3: -679.4325562, 910.2442017, -426.1267700, 575.2863770, -1254.7188721, 1336.3709717
4: -593.5498047, 897.7910767, -375.4928284, 563.0198975, -1156.5694580, 1272.2751465

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812275, upper bound: 886.6796754
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6795172
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -490.4579468, 796.0449219, -340.7877808, 548.1430054, -1038.6009521, 1136.8326416
1: -552.6120605, 793.8352661, -381.0401001, 542.4392090, -1095.0512695, 1174.8751221
2: -551.6123657, 784.1390991, -383.1042786, 536.7133179, -1088.3256836, 1167.2434082
3: -679.4325562, 910.2442017, -467.2512512, 625.8507690, -1305.2833252, 1377.4954834
4: -593.5498047, 897.7910767, -411.8325806, 614.3975220, -1207.9470215, 1308.0997314

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812275, upper bound: 886.6803040
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6802869
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -501.6998901, 813.3760376, -311.6872559, 502.6691284, -1004.3690186, 1125.0632324
1: -565.0808105, 811.0997314, -349.1390686, 496.2806702, -1061.3614502, 1160.1511230
2: -564.0930786, 801.6339722, -350.8332520, 491.2427063, -1055.3358154, 1151.7056885
3: -695.0361328, 930.1635742, -428.0234680, 571.6222534, -1266.6582031, 1358.1870117
4: -606.3902588, 918.0571289, -378.1982727, 562.2469482, -1168.6372070, 1294.5363770

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6777595, upper bound: 886.6776601
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6787649, upper bound: 886.6807379
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -501.6998901, 813.3760376, -323.6772766, 519.1259766, -1020.8258057, 1137.0533447
1: -565.0808105, 811.0997314, -361.9174500, 513.4822998, -1078.5631104, 1173.0172119
2: -564.0930786, 801.6339722, -363.6483459, 508.6350098, -1072.7280273, 1165.1008301
3: -695.0361328, 930.1635742, -444.2401123, 591.5661011, -1286.6022949, 1374.4035645
4: -606.3902588, 918.0571289, -390.8212585, 582.1261597, -1188.5162354, 1307.2832031

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6777595, upper bound: 886.6778546
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6787649, upper bound: 886.6807379
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -493.7385254, 800.9507446, -318.3107300, 516.1080933, -1009.8466187, 1119.2613525
1: -556.1115723, 798.7612305, -355.0302124, 508.9057617, -1065.0172119, 1153.7915039
2: -555.0532227, 789.2916260, -357.7380066, 504.3653870, -1059.4185791, 1147.0295410
3: -684.2304688, 915.9234009, -436.2491150, 587.3682861, -1271.5987549, 1352.1723633
4: -596.5521240, 903.8897705, -383.7341309, 575.3532715, -1171.9053955, 1286.6475830

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6792792, upper bound: 886.6793451
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6794889
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -493.7385254, 800.9507446, -351.2597656, 563.8720703, -1057.6105957, 1152.2103271
1: -556.1115723, 798.7612305, -392.6514282, 558.1578369, -1114.2694092, 1191.4125977
2: -555.0532227, 789.2916260, -394.6559753, 552.8757935, -1107.9285889, 1183.9476318
3: -684.2304688, 915.9234009, -481.8740540, 643.9133301, -1328.1437988, 1397.7972412
4: -596.5521240, 903.8897705, -423.8395081, 632.7755127, -1229.3276367, 1326.1168213

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799143, upper bound: 886.6793943
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799143, upper bound: 886.6793943
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -503.4063721, 816.5255737, -503.4063721, 816.5255737, -1319.9315186, 1319.9315186
1: -567.2109375, 814.8658447, -567.2109375, 814.8658447, -1380.0102539, 1380.0102539
2: -566.1912842, 805.2928467, -566.1912842, 805.2928467, -1369.6757812, 1369.6757812
3: -697.4225464, 934.2309570, -697.4225464, 934.2309570, -1630.0316162, 1630.0316162
4: -609.4550781, 921.6151733, -609.4550781, 921.6151733, -1528.4807129, 1528.4807129

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6640688, upper bound: 886.6630664
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6784086
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -505.8939209, 821.0324707, -503.4063721, 816.5255737, -1322.4190674, 1324.4287109
1: -570.0043335, 819.0781250, -567.2109375, 814.8658447, -1382.9053955, 1384.1199951
2: -569.0660400, 809.3848877, -566.1912842, 805.2928467, -1372.6131592, 1373.7221680
3: -700.7531738, 939.1489258, -697.4225464, 934.2309570, -1633.4744873, 1634.8137207
4: -612.7015991, 926.3571167, -609.4550781, 921.6151733, -1531.7593994, 1533.1909180

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6640688, upper bound: 886.6630664
time: 1.16 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6799612
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -503.4063721, 816.5255737, -505.8939209, 821.0324707, -1324.4285889, 1322.4191895
1: -567.2109375, 814.8658447, -570.0043335, 819.0781250, -1384.1199951, 1382.9052734
2: -566.1912842, 805.2928467, -569.0660400, 809.3848877, -1373.7220459, 1372.6131592
3: -697.4225464, 934.2309570, -700.7531738, 939.1489258, -1634.8137207, 1633.4746094
4: -609.4550781, 921.6151733, -612.7015991, 926.3571167, -1533.1909180, 1531.7593994

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6636618, upper bound: 886.6794288
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6636618, upper bound: 886.6801339
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -505.8939209, 821.0324707, -505.8939209, 821.0324707, -1326.9262695, 1326.9262695
1: -570.0043335, 819.0781250, -570.0043335, 819.0781250, -1387.2397461, 1387.2397461
2: -569.0660400, 809.3848877, -569.0660400, 809.3848877, -1376.7785645, 1376.7785645
3: -700.7531738, 939.1489258, -700.7531738, 939.1489258, -1638.5305176, 1638.5306396
4: -612.7015991, 926.3571167, -612.7015991, 926.3571167, -1536.5627441, 1536.5627441

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6636618, upper bound: 886.6819570
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6817819
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -507.9030457, 823.7993774, -503.4063721, 816.5255737, -1324.4283447, 1326.9486084
1: -572.0789185, 821.6602173, -567.2109375, 814.8658447, -1384.9732666, 1386.5596924
2: -571.1145020, 812.0858154, -566.1912842, 805.2928467, -1374.6942139, 1376.3168945
3: -703.6995850, 942.2774048, -697.4225464, 934.2309570, -1636.4385986, 1637.8204346
4: -614.1950073, 929.8233032, -609.4550781, 921.6151733, -1533.1296387, 1536.5596924

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6793286
time: 0.89 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6791433
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -508.0707092, 824.0618896, -505.8939209, 821.0324707, -1329.1031494, 1329.9558105
1: -572.2630005, 821.9155273, -570.0043335, 819.0781250, -1389.4931641, 1389.9346924
2: -571.3037109, 812.3393555, -569.0660400, 809.3848877, -1379.0482178, 1379.6203613
3: -703.9230347, 942.5779419, -700.7531738, 939.1489258, -1641.7207031, 1641.8411865
4: -614.3991089, 930.1170044, -612.7015991, 926.3571167, -1538.1374512, 1540.2204590

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6811216
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6809192
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -505.8939209, 821.0324707, -508.0707092, 824.0618896, -1329.9558105, 1329.1030273
1: -570.0043335, 819.0781250, -572.2630005, 821.9155273, -1389.9345703, 1389.4930420
2: -569.0660400, 809.3848877, -571.3037109, 812.3393555, -1379.6203613, 1379.0483398
3: -700.7531738, 939.1489258, -703.9230347, 942.5779419, -1641.8411865, 1641.7207031
4: -612.7015991, 926.3571167, -614.3991089, 930.1170044, -1540.2204590, 1538.1374512

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5682596, upper bound: 886.6791182
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5719115, upper bound: 886.6779638
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -508.0707092, 824.0618896, -508.0707092, 824.0618896, -1332.1325684, 1332.1325684
1: -572.2630005, 821.9155273, -572.2630005, 821.9155273, -1392.8258057, 1392.8258057
2: -571.3037109, 812.3393555, -571.3037109, 812.3393555, -1382.4024658, 1382.4024658
3: -703.9230347, 942.5779419, -703.9230347, 942.5779419, -1645.7967529, 1645.7967529
4: -614.3991089, 930.1170044, -614.3991089, 930.1170044, -1542.0708008, 1542.0708008

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.5602000, upper bound: 886.6502927
time: 0.98 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5599986, upper bound: 886.6259169
time: 1.10 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.51 seconds
NS_A1_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6763972, upper bound: 886.6749722
NS_A1_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6759325, upper bound: 886.6753164
NS_A1_B1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6774499, upper bound: 886.6752000
NS_A1_B1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6774499, upper bound: 886.6755502
NS_A1_B1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6805371, upper bound: 886.6784033
NS_A1_B1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6804023, upper bound: 886.6768803
NS_A1_B1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6713990, upper bound: 886.6650162
NS_A1_B1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6714652, upper bound: 886.6652050
NS_A1_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6760303, upper bound: 886.6731193
NS_A1_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6760303, upper bound: 886.6732517
NS_A1_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6752111, upper bound: 886.6734683
NS_A1_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6752111, upper bound: 886.6736001
NS_A1_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
NS_A1_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
NS_A1_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
NS_A1_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6756275, upper bound: 886.6741832
NS_A1_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6748714, upper bound: 886.6713139
NS_A1_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6747658, upper bound: 886.6738555
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6746850, upper bound: 886.6737973
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6747658, upper bound: 886.6738555
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6744306, upper bound: 886.6726632
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6731741, upper bound: 886.6725228
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6745715, upper bound: 886.6739284
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6741534, upper bound: 886.6739284
NS_A1_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6738791, upper bound: 886.6768346
NS_A1_B1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6743351, upper bound: 886.6768346
NS_A1_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6734408, upper bound: 886.6768426
NS_A1_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6738853, upper bound: 886.6768426
NS_A1_B1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6737463, upper bound: 886.6737513
NS_A1_B1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6737463, upper bound: 886.6737594
NS_A1_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6737594, upper bound: 886.6739203
NS_A1_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6737594, upper bound: 886.6739284
NS_A1_B2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6796103
NS_A1_B2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6798278
NS_A1_B2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6796103
NS_A1_B2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796846, upper bound: 886.6798278
NS_A1_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6796430
NS_A1_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6798633
NS_A1_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6796430
NS_A1_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6798633
NS_A1_B2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6707133, upper bound: 886.6646671
NS_A1_B2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6705179, upper bound: 886.6634094
NS_A1_B2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6709315, upper bound: 886.6662500
NS_A1_B2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6706892, upper bound: 886.6645679
NS_A1_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6799823
NS_A1_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6799823
NS_A1_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6802103
NS_A1_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793621, upper bound: 886.6802103
NS_A1_B2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6808126
NS_A1_B2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6804912, upper bound: 886.6814176
NS_A1_B2_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6805319, upper bound: 886.6808126
NS_A1_B2_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6804912, upper bound: 886.6814176
NS_A1_B2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6812275
NS_A1_B2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6795172, upper bound: 886.6818324
NS_A1_B2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796754, upper bound: 886.6808197
NS_A1_B2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6795172, upper bound: 886.6811649
NS_A1_B2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6807379, upper bound: 886.6787649
NS_A1_B2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6795964
NS_A1_B2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6807379, upper bound: 886.6788609
NS_A1_B2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6808817, upper bound: 886.6796979
NS_A1_B2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6793451, upper bound: 886.6792792
NS_A1_B2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6794889, upper bound: 886.6801176
NS_A1_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6790991, upper bound: 886.6798494
NS_A1_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6790991, upper bound: 886.6800307
NS_A2_B1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6796846
NS_A2_B1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6800799
NS_A2_B1_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6814145
NS_A2_B1_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796103, upper bound: 886.6814195
NS_A2_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6793621
NS_A2_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6797549
NS_A2_B1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6810895
NS_A2_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6810945
NS_A2_B1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6662628, upper bound: 886.6737174
NS_A2_B1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6662628, upper bound: 886.6737174
NS_A2_B1_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6645599, upper bound: 886.6706083
NS_A2_B1_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6633574, upper bound: 886.6704153
NS_A2_B1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6795426
NS_A2_B1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6812976
NS_A2_B1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6796430, upper bound: 886.6799485
NS_A2_B1_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6799823, upper bound: 886.6812995
NS_A2_B1_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6808126, upper bound: 886.6805319
NS_A2_B1_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6804912
NS_A2_B1_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6808126, upper bound: 886.6810732
NS_A2_B1_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6814176, upper bound: 886.6809150
NS_A2_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6812275, upper bound: 886.6796754
NS_A2_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6795172
NS_A2_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6812275, upper bound: 886.6803040
NS_A2_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6818324, upper bound: 886.6802869
NS_A2_B1_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6787649, upper bound: 886.6807379
NS_A2_B1_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
NS_A2_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6787649, upper bound: 886.6807379
NS_A2_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6795964, upper bound: 886.6808817
NS_A2_B1_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6792792, upper bound: 886.6793451
NS_A2_B1_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6801176, upper bound: 886.6794889
NS_A2_B1_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6799143, upper bound: 886.6793943
NS_A2_B1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6799143, upper bound: 886.6793943
NS_A2_B2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6640688, upper bound: 886.6630664
NS_A2_B2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6784086
NS_A2_B2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6640688, upper bound: 886.6630664
NS_A2_B2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6799612
NS_A2_B2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6636618, upper bound: 886.6794288
NS_A2_B2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6636618, upper bound: 886.6801339
NS_A2_B2_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6636618, upper bound: 886.6819570
NS_A2_B2_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.6784086, upper bound: 886.6817819
NS_A2_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6793286
NS_A2_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6791433
NS_A2_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6811216
NS_A2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5655461, upper bound: 886.6809192
NS_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5682596, upper bound: 886.6791182
NS_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5719115, upper bound: 886.6779638
NS_A2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5602000, upper bound: 886.6502927
NS_A2_B2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.51
Output dim: 0, lower bound: -886.5599986, upper bound: 886.6259169

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -288.8058472, 466.2946167, -274.8720398, 444.6781921, -733.4840088, 741.1665649
1: -323.0443115, 459.4102173, -306.2747192, 437.0561829, -760.1004028, 765.6849365
2: -325.1664429, 454.9240112, -309.2665100, 432.7247925, -757.8912354, 764.1905518
3: -395.6399231, 529.4348145, -375.4543152, 504.4940186, -900.1339111, 904.8890991
4: -350.8740540, 520.7164917, -333.0665894, 494.9523010, -845.8263550, 853.7830811

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6744754, upper bound: 886.6742817
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6747010, upper bound: 886.6741151
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -311.3512878, 503.3862305, -276.4292297, 447.0335693, -758.3848877, 779.8153076
1: -348.1929016, 496.1621399, -308.0655823, 439.9624329, -788.1553345, 804.2275391
2: -350.5265808, 491.1569824, -310.9670715, 435.5761108, -786.1026611, 802.1238403
3: -426.4289551, 573.2835083, -377.7672729, 507.6893005, -934.1182861, 951.0507812
4: -378.0692139, 562.2777100, -334.6965637, 498.0320435, -876.1011353, 896.9742432

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6695591, upper bound: 886.6675870
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6757941, upper bound: 886.6741865
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6751168, upper bound: 886.6740390
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -300.2607422, 482.8410645, -282.7000122, 456.9491882, -757.2099609, 765.5410156
1: -335.1759033, 476.6418457, -314.8608398, 449.7230530, -784.8989258, 791.5026855
2: -337.3558655, 472.2504578, -317.8348999, 445.4586487, -782.8145142, 790.0853271
3: -411.5100708, 549.4415894, -386.5639038, 519.0419922, -930.5520020, 936.0054321
4: -362.8706970, 540.2652588, -341.5359802, 509.1776123, -872.0483398, 881.8011475

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6622099, upper bound: 886.6542198
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6774499, upper bound: 886.6752000
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -323.5754700, 519.9789429, -287.6216431, 464.8513489, -788.4267578, 807.6005249
1: -361.3382568, 514.0121460, -320.4265747, 458.0711365, -819.4094238, 834.4385986
2: -363.6484985, 509.2173462, -323.3335876, 453.7365723, -817.3850708, 832.5509033
3: -443.4880371, 592.7652588, -393.5621643, 528.6509399, -972.1389771, 986.3273926
4: -391.3461304, 582.7015381, -347.4093018, 518.4214478, -909.7673950, 930.1108398

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6622099, upper bound: 886.6533471
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6774499, upper bound: 886.6755502
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -310.0726013, 499.0182495, -376.7810364, 622.3963013, -931.7603760, 875.7991333
1: -346.2801819, 493.0499573, -420.6375427, 608.9332886, -955.2131958, 913.6875000
2: -348.3768005, 488.3540344, -423.1929321, 603.2849731, -951.6617432, 911.5468750
3: -425.0463562, 568.2976685, -517.0490723, 703.1131592, -1128.1594238, 1085.3466797
4: -374.8595581, 558.6082153, -454.5419922, 686.7877197, -1061.6472168, 1013.1501465

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6767300, upper bound: 886.6713311
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6779074, upper bound: 886.6713615
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -310.0726013, 499.0182495, -376.9888306, 622.6063232, -932.0416870, 876.0069580
1: -346.2801819, 493.0499573, -420.8986206, 609.2452393, -955.5253296, 913.9486084
2: -348.3768005, 488.3540344, -423.4340210, 603.5914917, -951.9682617, 911.7880859
3: -425.0463562, 568.2976685, -517.3796997, 703.4808960, -1128.5270996, 1085.6773682
4: -374.8595581, 558.6082153, -454.8391113, 687.1603394, -1062.0198975, 1013.4472656

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765784, upper bound: 886.6697862
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765784, upper bound: 886.6698009
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -327.7629395, 527.6514282, -368.2615662, 608.7158203, -936.0908813, 895.9129639
1: -365.8799438, 521.6754150, -411.2327271, 595.5740967, -961.4540405, 932.9080811
2: -368.2852173, 516.6662598, -413.5881042, 590.0273438, -958.3124390, 930.2543945
3: -448.9193115, 602.2103882, -505.7252197, 687.7002563, -1136.6196289, 1107.9355469
4: -396.3608398, 591.0858765, -444.1581726, 671.4703979, -1067.8310547, 1035.2440186

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6677110, upper bound: 886.6606267
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6680720, upper bound: 886.6606267
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -326.9273071, 526.4905396, -370.3886108, 612.0898438, -938.7749023, 896.8791504
1: -365.0631714, 520.5346069, -413.5783997, 599.0254517, -964.0885620, 934.1130371
2: -367.3824158, 515.5528564, -415.9863892, 593.4879761, -960.8703003, 931.5392456
3: -448.0971680, 600.6775513, -508.5949707, 691.6522827, -1139.7492676, 1109.2724609
4: -395.4451599, 589.2955322, -446.7259521, 675.5093384, -1070.9544678, 1036.0213623

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6703644, upper bound: 886.6650972
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6702221, upper bound: 886.6634394
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -302.1397705, 486.7652283, -310.8713379, 500.8233948, -802.9630737, 797.6365356
1: -338.0361328, 480.5949402, -348.0654907, 494.4885254, -832.5246582, 828.6604004
2: -340.0501099, 475.7366943, -349.9492493, 489.5337830, -829.5838623, 825.6857910
3: -414.0634766, 553.7343750, -426.3220520, 569.7549438, -983.8184204, 980.0563965
4: -366.7779541, 544.7932129, -377.6205444, 560.4786987, -927.2566528, 922.4137573

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760187, upper bound: 886.6731193
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760187, upper bound: 886.6731193
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -302.1397705, 486.7652283, -321.8392334, 515.9621582, -818.1019287, 808.6043701
1: -338.0361328, 480.5949402, -359.7606201, 510.3937988, -848.4299316, 840.3555908
2: -340.0501099, 475.7366943, -361.6361389, 505.6689758, -845.7191162, 837.3727417
3: -414.0634766, 553.7343750, -441.4616394, 588.0678711, -1002.1313477, 995.1960449
4: -366.7779541, 544.7932129, -389.1470337, 578.6617432, -945.4396973, 933.9402466

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6745358, upper bound: 886.6714702
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760187, upper bound: 886.6732517
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760187, upper bound: 886.6732517
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -323.0477600, 521.3051147, -316.5196838, 509.5200806, -832.5678711, 837.8248291
1: -361.2111511, 514.6909180, -354.4276428, 503.0150146, -864.2261963, 869.1185303
2: -363.5712891, 509.2450562, -356.3714600, 498.0783997, -861.6496582, 865.6164551
3: -442.4188538, 594.4000854, -434.0364380, 579.6705322, -1022.0893555, 1028.4362793
4: -391.9054871, 583.1520996, -384.8628845, 570.2559204, -962.1613770, 968.0148315

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6751985, upper bound: 886.6734683
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6751985, upper bound: 886.6734683
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -323.0477600, 521.3051147, -328.3813782, 526.2692871, -849.3170166, 849.6865234
1: -361.2111511, 514.6909180, -367.1471558, 520.7193604, -881.9305420, 881.8380737
2: -363.5712891, 509.2450562, -369.0917358, 515.9728394, -879.5441284, 878.3366089
3: -442.4188538, 594.4000854, -450.3800049, 599.9805298, -1042.3994141, 1044.7797852
4: -391.9054871, 583.1520996, -397.4562683, 590.2601929, -982.1656494, 980.6083984

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6751985, upper bound: 886.6736001
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6751985, upper bound: 886.6736001
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -286.4380798, 463.0567017, -334.3932495, 536.3743286, -822.8123779, 797.4499512
1: -319.1374207, 456.0961914, -373.9693909, 530.9953613, -850.1328125, 830.0655518
2: -322.0881653, 451.8367615, -375.8881226, 526.0021973, -848.0903320, 827.7247314
3: -391.9034729, 526.3308716, -458.6832581, 611.7830200, -1003.6865234, 985.0141602
4: -346.1516724, 516.2714844, -405.1340942, 601.9392700, -948.0909424, 921.4054565

Time for backsubstitution: 0.99 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.96 + 417.07 = 420.03 seconds
