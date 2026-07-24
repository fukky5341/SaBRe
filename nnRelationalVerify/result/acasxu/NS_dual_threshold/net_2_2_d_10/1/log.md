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
execution time: IAR + RelationalAnalysis = 1.46 + 2.12 = 3.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2561.2089106, upper bound: 2561.2089106

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0913199, upper bound: 2561.1570215
time: 0.58 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2561.0892788, upper bound: 2561.0892788
time: 1.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.72 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -2561.0913199, upper bound: 2561.1570215
NS_A2, status: Status.VERIFIED, split count: 1, time: 1.72
Output dim: 0, lower bound: -2561.0892788, upper bound: 2561.0892788

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1091.7904053, 1704.6247559, -1091.8393555, 1704.6987305, -2796.4890137, 2796.4641113
1: -844.7322388, 1571.0827637, -844.7684937, 1571.1506348, -2415.8828125, 2415.8513184
2: -735.8150635, 1621.2486572, -735.8467407, 1621.3187256, -2357.1337891, 2357.0954590
3: -1145.6348877, 1614.0903320, -1145.6840820, 1614.1608887, -2759.7956543, 2759.7744141
4: -904.0523071, 1719.3662109, -904.0911865, 1719.4404297, -2623.4921875, 2623.4575195

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0913199, upper bound: 2561.1570215
time: 0.72 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0913027, upper bound: 2561.1556887
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.02 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -2561.0913199, upper bound: 2561.1570215
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 0, lower bound: -2561.0913027, upper bound: 2561.1556887

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -1089.4348145, 1700.8808594, -1091.8393555, 1704.6987305, -2794.1335449, 2792.7202148
1: -842.8968506, 1567.6606445, -844.7684937, 1571.1506348, -2414.0473633, 2412.4289551
2: -734.2087402, 1617.7165527, -735.8467407, 1621.3187256, -2355.5273438, 2353.5632324
3: -1143.1522217, 1610.5892334, -1145.6840820, 1614.1608887, -2757.3129883, 2756.2734375
4: -902.0716553, 1715.6229248, -904.0911865, 1719.4404297, -2621.5117188, 2619.7141113

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0903576, upper bound: 2561.1237638
time: 0.65 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0913199, upper bound: 2561.1570215
time: 0.83 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -1092.2701416, 1705.8005371, -1091.8393555, 1704.6987305, -2796.9687500, 2797.6398926
1: -845.1296997, 1572.1713867, -844.7684937, 1571.1506348, -2416.2802734, 2416.9399414
2: -736.1979370, 1622.3436279, -735.8467407, 1621.3187256, -2357.5166016, 2358.1904297
3: -1146.1154785, 1615.0903320, -1145.6840820, 1614.1608887, -2760.2763672, 2760.7744141
4: -904.5543823, 1720.4045410, -904.0911865, 1719.4404297, -2623.9938965, 2624.4956055

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1556887
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.90 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 0, lower bound: -2561.0903576, upper bound: 2561.1237638
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 0, lower bound: -2561.0913199, upper bound: 2561.1570215
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.90
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1556887

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -1077.0369873, 1681.8752441, -1090.1752930, 1702.1401367, -2779.1772461, 2772.0505371
1: -833.3454590, 1550.1154785, -843.4830322, 1568.7893066, -2402.1347656, 2393.5983887
2: -725.8635864, 1599.6499023, -734.7266235, 1618.8845215, -2344.7480469, 2334.3764648
3: -1130.2255859, 1592.4160156, -1143.9447021, 1611.7167969, -2741.9423828, 2736.3608398
4: -891.7825928, 1696.3831787, -902.7114258, 1716.8485107, -2608.6311035, 2599.0947266

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1208405
time: 0.58 seconds

## Relational analysis of NS_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1208405
time: 0.89 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -1084.1901855, 1692.8068848, -1091.8393555, 1704.6987305, -2788.8886719, 2784.6462402
1: -838.8945923, 1560.2205811, -844.7684937, 1571.1506348, -2410.0451660, 2404.9890137
2: -730.7289429, 1610.0281982, -735.8467407, 1621.3187256, -2352.0471191, 2345.8750000
3: -1137.7342529, 1602.9099121, -1145.6840820, 1614.1608887, -2751.8950195, 2748.5939941
4: -897.8019409, 1707.4757080, -904.0911865, 1719.4404297, -2617.2421875, 2611.5668945

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1559552
time: 0.71 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1570215
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -1092.2701416, 1705.8005371, -1089.4838867, 1700.9550781, -2793.2250977, 2795.2841797
1: -845.1296997, 1572.1713867, -842.9332886, 1567.7290039, -2412.8586426, 2415.1047363
2: -736.1979370, 1622.3436279, -734.2404785, 1617.7868652, -2353.9848633, 2356.5839844
3: -1146.1154785, 1615.0903320, -1143.2015381, 1610.6595459, -2756.7749023, 2758.2917480
4: -904.5543823, 1720.4045410, -902.1107788, 1715.6973877, -2620.2509766, 2622.5151367

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
time: 0.76 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1092.2701416, 1705.8005371, -1092.3220215, 1705.8784180, -2798.1481934, 2798.1225586
1: -845.1296997, 1572.1713867, -845.1680908, 1572.2432861, -2417.3730469, 2417.3393555
2: -736.1979370, 1622.3436279, -736.2315063, 1622.4174805, -2358.6154785, 2358.5751953
3: -1146.1154785, 1615.0903320, -1146.1674805, 1615.1645508, -2761.2800293, 2761.2578125
4: -904.5543823, 1720.4045410, -904.5956421, 1720.4825439, -2625.0368652, 2625.0002441

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
time: 0.81 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 7.34 seconds
NS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1208405
NS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1208405
NS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1559552
NS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1570215
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.34
Output dim: 0, lower bound: -2561.0906928, upper bound: 2561.1546117

## BFS NS instance: NS_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -1077.0369873, 1681.8752441, -1087.8256836, 1698.4038086, -2775.4409180, 2769.7006836
1: -833.3454590, 1550.1154785, -841.6524658, 1565.3751221, -2398.7207031, 2391.7680664
2: -725.8635864, 1599.6499023, -733.1236572, 1615.3603516, -2341.2238770, 2332.7731934
3: -1130.2255859, 1592.4160156, -1141.4680176, 1608.2235107, -2738.4492188, 2733.8840332
4: -891.7825928, 1696.3831787, -900.7350464, 1713.1140137, -2604.8964844, 2597.1176758

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1208405
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1208405
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1077.0369873, 1681.8752441, -1090.6854248, 1703.3570557, -2780.3940430, 2772.5603027
1: -833.3454590, 1550.1154785, -843.9013062, 1569.9161377, -2403.2617188, 2394.0166016
2: -725.8635864, 1599.6499023, -735.1281128, 1620.0186768, -2345.8823242, 2334.7775879
3: -1130.2255859, 1592.4160156, -1144.4530029, 1612.7572021, -2742.9829102, 2736.8691406
4: -891.7825928, 1696.3831787, -903.2366943, 1717.9287109, -2609.7114258, 2599.6198730

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1237638
time: 0.69 seconds

## Relational analysis of NS_A1_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0896862, upper bound: 2561.1237638
time: 0.82 seconds

## BFS NS instance: NS_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -1084.1901855, 1692.8068848, -1089.4838867, 1700.9550781, -2785.1452637, 2782.2907715
1: -838.8945923, 1560.2205811, -842.9332886, 1567.7290039, -2406.6232910, 2403.1535645
2: -730.7289429, 1610.0281982, -734.2404785, 1617.7868652, -2348.5153809, 2344.2685547
3: -1137.7342529, 1602.9099121, -1143.2015381, 1610.6595459, -2748.3937988, 2746.1113281
4: -897.8019409, 1707.4757080, -902.1107788, 1715.6973877, -2613.4990234, 2609.5864258

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1559552
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1559552
time: 0.89 seconds

## BFS NS instance: NS_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1084.1901855, 1692.8068848, -1092.3220215, 1705.8784180, -2790.0683594, 2785.1289062
1: -838.8945923, 1560.2205811, -845.1680908, 1572.2432861, -2411.1379395, 2405.3886719
2: -730.7289429, 1610.0281982, -736.2315063, 1622.4174805, -2353.1459961, 2346.2597656
3: -1137.7342529, 1602.9099121, -1146.1674805, 1615.1645508, -2752.8989258, 2749.0773926
4: -897.8019409, 1707.4757080, -904.5956421, 1720.4825439, -2618.2844238, 2612.0712891

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1570215
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2561.0907103, upper bound: 2561.1570215
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1092.2701416, 1705.8005371, -1089.4348145, 1700.8808594, -2793.1508789, 2795.2353516
1: -845.1296997, 1572.1713867, -842.8968506, 1567.6606445, -2412.7902832, 2415.0683594
2: -736.1979370, 1622.3436279, -734.2087402, 1617.7165527, -2353.9143066, 2356.5522461
3: -1146.1154785, 1615.0903320, -1143.1522217, 1610.5892334, -2756.7045898, 2758.2426758
4: -904.5543823, 1720.4045410, -902.0716553, 1715.6229248, -2620.1772461, 2622.4760742

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1092.2701416, 1705.8005371, -1114.9060059, 1741.7856445, -2834.0556641, 2820.7065430
1: -845.1296997, 1572.1713867, -862.9465332, 1605.1343994, -2450.2641602, 2435.1179199
2: -736.1979370, 1622.3436279, -751.6840820, 1656.5772705, -2392.7751465, 2374.0278320
3: -1146.1154785, 1615.0903320, -1170.4094238, 1648.9020996, -2795.0175781, 2785.4997559
4: -904.5543823, 1720.4045410, -923.4351196, 1756.8753662, -2661.4296875, 2643.8393555

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1092.2701416, 1705.8005371, -1092.2701416, 1705.8005371, -2798.0705566, 2798.0705566
1: -845.1296997, 1572.1713867, -845.1296997, 1572.1713867, -2417.3010254, 2417.3010254
2: -736.1979370, 1622.3436279, -736.1979370, 1622.3436279, -2358.5415039, 2358.5415039
3: -1146.1154785, 1615.0903320, -1146.1154785, 1615.0903320, -2761.2058105, 2761.2058105
4: -904.5543823, 1720.4045410, -904.5543823, 1720.4045410, -2624.9587402, 2624.9587402

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1092.2701416, 1705.8005371, -1116.4600830, 1744.4079590, -2836.6779785, 2822.2602539
1: -845.1296997, 1572.1713867, -864.1538696, 1607.5960693, -2452.7258301, 2436.3251953
2: -736.1979370, 1622.3436279, -752.7665405, 1659.0961914, -2395.2941895, 2375.1101074
3: -1146.1154785, 1615.0903320, -1171.9877930, 1651.3466797, -2797.4621582, 2787.0781250
4: -904.5543823, 1720.4045410, -924.7749023, 1759.4901123, -2664.0444336, 2645.1794434

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 5

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.58 + 59.72 = 63.30 seconds
