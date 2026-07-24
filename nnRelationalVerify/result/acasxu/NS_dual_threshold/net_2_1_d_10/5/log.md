## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 1442.1242243135432


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-710.2344360, 905.8535156, -710.2344360, 905.8535156, -1616.0878906, 1616.0878906)
1: (-562.7462158, 727.9412842, -562.7462158, 727.9412842, -1290.6872559, 1290.6872559)
2: (-488.6404724, 716.1848145, -488.6404724, 716.1848145, -1204.8249512, 1204.8249512)
3: (-664.5035400, 884.3118286, -664.5035400, 884.3118286, -1548.8153076, 1548.8153076)
4: (-654.3821411, 970.1540527, -654.3821411, 970.1540527, -1624.5361328, 1624.5361328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 2.27 = 3.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1442.1386457, upper bound: 1442.1386457

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1322537, upper bound: 1442.1356329
time: 0.76 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1319109, upper bound: 1442.1319109
time: 1.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.00 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 0, lower bound: -1442.1322537, upper bound: 1442.1356329
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 0, lower bound: -1442.1319109, upper bound: 1442.1319109

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -709.5701294, 904.9933472, -710.2344360, 905.8535156, -1615.4235840, 1615.2277832
1: -562.2149048, 727.2516479, -562.7462158, 727.9412842, -1290.1558838, 1289.9978027
2: -488.1798401, 715.5029297, -488.6404724, 716.1848145, -1204.3645020, 1204.1430664
3: -663.8747559, 883.4719849, -664.5035400, 884.3118286, -1548.1864014, 1547.9755859
4: -653.7642212, 969.2296143, -654.3821411, 970.1540527, -1623.9182129, 1623.6118164

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1314961, upper bound: 1442.1329292
time: 1.27 seconds

## Relational analysis of NS_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1301434, upper bound: 1442.1350480
time: 0.75 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1317800, upper bound: 1442.1354614
time: 1.02 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -728.9163818, 927.6904297, -707.0169067, 901.7774658, -1630.6937256, 1634.7072754
1: -576.8234863, 746.2800903, -560.1957397, 724.6711426, -1301.4946289, 1306.4758301
2: -500.8928223, 732.9152832, -486.4284668, 712.9572144, -1213.8500977, 1219.3436279
3: -680.5924072, 906.9973755, -661.5222778, 880.3408203, -1560.9332275, 1568.5196533
4: -671.0776367, 992.2803345, -651.4390869, 965.7996216, -1636.8771973, 1643.7194824

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1319109, upper bound: 1442.1319109
time: 0.75 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1319109, upper bound: 1442.1319109
time: 0.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.28 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.28
Output dim: 0, lower bound: -1442.1301434, upper bound: 1442.1350480
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.28
Output dim: 0, lower bound: -1442.1317800, upper bound: 1442.1354614
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.28
Output dim: 0, lower bound: -1442.1319109, upper bound: 1442.1319109
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.28
Output dim: 0, lower bound: -1442.1319109, upper bound: 1442.1319109

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -698.9045410, 891.2833862, -691.0548706, 881.5275879, -1580.4321289, 1582.3382568
1: -553.6767578, 716.1932373, -547.5364990, 708.3796997, -1262.0563965, 1263.7297363
2: -480.7900696, 704.6320190, -475.4703979, 697.0087891, -1177.7988281, 1180.1024170
3: -653.8710938, 870.0370483, -646.7892456, 860.5602417, -1514.4313965, 1516.8262939
4: -643.8659058, 954.4996948, -636.8048096, 944.2108765, -1588.0766602, 1591.3044434

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297975, upper bound: 1442.1305985
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297975, upper bound: 1442.1350480
time: 1.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -709.5701294, 904.9933472, -708.9679565, 904.1557007, -1613.7258301, 1613.9613037
1: -562.2149048, 727.2516479, -561.7214355, 726.5818481, -1288.7962646, 1288.9731445
2: -488.1798401, 715.5029297, -487.7538147, 714.8319092, -1203.0117188, 1203.2565918
3: -663.8747559, 883.4719849, -663.2827148, 882.6629028, -1546.5375977, 1546.7546387
4: -653.7642212, 969.2296143, -653.1882935, 968.3211670, -1622.0854492, 1622.4179688

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1313116, upper bound: 1442.1306374
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1313116, upper bound: 1442.1354614
time: 1.31 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -728.9163818, 927.6904297, -709.5701294, 904.9933472, -1633.9096680, 1637.2602539
1: -576.8234863, 746.2800903, -562.2149048, 727.2516479, -1304.0751953, 1308.4948730
2: -500.8928223, 732.9152832, -488.1798401, 715.5029297, -1216.3956299, 1221.0949707
3: -680.5924072, 906.9973755, -663.8747559, 883.4719849, -1564.0644531, 1570.8720703
4: -671.0776367, 992.2803345, -653.7642212, 969.2296143, -1640.3072510, 1646.0445557

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1313417, upper bound: 1442.1317369
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1313291, upper bound: 1442.1313291
time: 0.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -728.9163818, 927.6904297, -728.9163818, 927.6904297, -1656.6066895, 1656.6066895
1: -576.8234863, 746.2800903, -576.8234863, 746.2800903, -1323.1035156, 1323.1035156
2: -500.8928223, 732.9152832, -500.8928223, 732.9152832, -1233.8079834, 1233.8079834
3: -680.5924072, 906.9973755, -680.5924072, 906.9973755, -1587.5898438, 1587.5898438
4: -671.0776367, 992.2803345, -671.0776367, 992.2803345, -1663.3579102, 1663.3579102

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1313073, upper bound: 1442.1298330
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1314242, upper bound: 1442.1314242
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.11 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1297975, upper bound: 1442.1305985
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1297975, upper bound: 1442.1350480
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1313116, upper bound: 1442.1306374
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1313116, upper bound: 1442.1354614
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1313417, upper bound: 1442.1317369
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1313291, upper bound: 1442.1313291
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1313073, upper bound: 1442.1298330
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.11
Output dim: 0, lower bound: -1442.1314242, upper bound: 1442.1314242

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -690.3966064, 880.6719360, -691.0548706, 881.5275879, -1571.9241943, 1571.7266846
1: -547.0102539, 707.6943970, -547.5364990, 708.3796997, -1255.3897705, 1255.2308350
2: -475.0139465, 696.3309937, -475.4703979, 697.0087891, -1172.0227051, 1171.8013916
3: -646.1658325, 859.7251587, -646.7892456, 860.5602417, -1506.7260742, 1506.5144043
4: -636.1922607, 943.2933350, -636.8048096, 944.2108765, -1580.4030762, 1580.0980225

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297911, upper bound: 1442.1305909
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297757
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -708.3037720, 903.2954712, -691.0548706, 881.5275879, -1589.8312988, 1594.3502197
1: -561.1901245, 725.8921509, -547.5364990, 708.3796997, -1269.5694580, 1273.4287109
2: -487.2931213, 714.1501465, -475.4703979, 697.0087891, -1184.3016357, 1189.6204834
3: -662.6538086, 881.8229980, -646.7892456, 860.5602417, -1523.2141113, 1528.6123047
4: -652.5703735, 967.3967285, -636.8048096, 944.2108765, -1596.7812500, 1604.2015381

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1282253, upper bound: 1442.1338779
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1282605, upper bound: 1442.1348750
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -690.3966064, 880.6719360, -708.9679565, 904.1557007, -1594.5522461, 1589.6396484
1: -547.0102539, 707.6943970, -561.7214355, 726.5818481, -1273.5917969, 1269.4157715
2: -475.0139465, 696.3309937, -487.7538147, 714.8319092, -1189.8458252, 1184.0848389
3: -646.1658325, 859.7251587, -663.2827148, 882.6629028, -1528.8287354, 1523.0075684
4: -636.1922607, 943.2933350, -653.1882935, 968.3211670, -1604.5134277, 1596.4815674

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297911, upper bound: 1442.1306284
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1298211
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -708.3037720, 903.2954712, -708.9679565, 904.1557007, -1612.4594727, 1612.2631836
1: -561.1901245, 725.8921509, -561.7214355, 726.5818481, -1287.7714844, 1287.6135254
2: -487.2931213, 714.1501465, -487.7538147, 714.8319092, -1202.1248779, 1201.9038086
3: -662.6538086, 881.8229980, -663.2827148, 882.6629028, -1545.3166504, 1545.1057129
4: -652.5703735, 967.3967285, -653.1882935, 968.3211670, -1620.8916016, 1620.5849609

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297911, upper bound: 1442.1352952
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1319098
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -728.4357300, 927.0629883, -709.5701294, 904.9933472, -1633.4290771, 1636.6329346
1: -576.4336548, 745.7746582, -562.2149048, 727.2516479, -1303.6853027, 1307.9892578
2: -500.5565491, 732.4179688, -488.1798401, 715.5029297, -1216.0593262, 1220.5977783
3: -680.1347046, 906.3807373, -663.8747559, 883.4719849, -1563.6064453, 1570.2554932
4: -670.6256104, 991.6062622, -653.7642212, 969.2296143, -1639.8552246, 1645.3704834

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1318771, upper bound: 1442.1298753
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1319119, upper bound: 1442.1312674
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -730.1359863, 929.2614746, -709.0498657, 904.3208008, -1634.4567871, 1638.3110352
1: -577.6275024, 747.4442139, -561.7947388, 726.7110596, -1304.3386230, 1309.2390137
2: -501.6637268, 734.1590576, -487.8182983, 714.9701538, -1216.6336670, 1221.9772949
3: -681.7904663, 908.2460938, -663.3833008, 882.8115845, -1564.6020508, 1571.6291504
4: -672.0994263, 994.0568237, -653.2791748, 968.5065308, -1640.6059570, 1647.3359375

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298211, upper bound: 1442.1307953
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1319098, upper bound: 1442.1309094
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -712.4013062, 905.8169556, -719.7376709, 915.6398926, -1628.0410156, 1625.5546875
1: -563.4951172, 728.6843872, -569.4318237, 736.5804443, -1300.0755615, 1298.1162109
2: -489.3368835, 715.5709839, -494.4919739, 723.3489990, -1212.6859131, 1210.0629883
3: -664.8483887, 885.6200562, -671.8610229, 895.2069702, -1560.0552979, 1557.4810791
4: -655.5187378, 968.7247314, -662.4724731, 979.2986450, -1634.8173828, 1631.1972656

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297812, upper bound: 1442.1297812
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297812, upper bound: 1442.1298330
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -726.8963013, 925.0805664, -728.9163818, 927.6904297, -1654.5866699, 1653.9965820
1: -575.1999512, 744.1749268, -576.8234863, 746.2800903, -1321.4799805, 1320.9984131
2: -499.4863892, 730.8354492, -500.8928223, 732.9152832, -1232.4016113, 1231.7282715
3: -678.6852417, 904.4452515, -680.5924072, 906.9973755, -1585.6826172, 1585.0375977
4: -669.1954346, 989.4607544, -671.0776367, 992.2803345, -1661.4755859, 1660.5383301

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298330, upper bound: 1442.1313073
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298330, upper bound: 1442.1314242
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.25 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297911, upper bound: 1442.1305909
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297757
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1282253, upper bound: 1442.1338779
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1282605, upper bound: 1442.1348750
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297911, upper bound: 1442.1306284
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1298211
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297911, upper bound: 1442.1352952
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1319098
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1318771, upper bound: 1442.1298753
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1319119, upper bound: 1442.1312674
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1298211, upper bound: 1442.1307953
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1319098, upper bound: 1442.1309094
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297812, upper bound: 1442.1297812
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1297812, upper bound: 1442.1298330
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1298330, upper bound: 1442.1313073
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 0, lower bound: -1442.1298330, upper bound: 1442.1314242

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -689.9880981, 880.1305542, -691.0548706, 881.5275879, -1571.5156250, 1571.1854248
1: -546.6776733, 707.2580566, -547.5364990, 708.3796997, -1255.0572510, 1254.7945557
2: -474.7271118, 695.9012451, -475.4703979, 697.0087891, -1171.7358398, 1171.3715820
3: -645.7726440, 859.1928101, -646.7892456, 860.5602417, -1506.3328857, 1505.9820557
4: -635.8052979, 942.7105713, -636.8048096, 944.2108765, -1580.0159912, 1579.5152588

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297475, upper bound: 1442.1305835
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -696.8139648, 888.7787476, -690.3899536, 880.6820679, -1577.4960938, 1579.1687012
1: -551.9014893, 714.1286621, -547.0001221, 707.6961060, -1259.5976562, 1261.1287842
2: -479.3235168, 702.7254028, -475.0086060, 696.3369751, -1175.6604004, 1177.7340088
3: -652.0493164, 867.4312134, -646.1660767, 859.7258301, -1511.7750244, 1513.5972900
4: -641.9025269, 951.9027100, -636.1865234, 943.2980347, -1585.2003174, 1588.0891113

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297757
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297757
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -700.8682861, 894.8290405, -677.1918945, 863.4994507, -1564.3676758, 1572.0209961
1: -555.8326416, 719.5971069, -536.4152832, 693.8342896, -1249.6669922, 1256.0124512
2: -482.3995361, 707.8442993, -465.8436890, 682.6361694, -1165.0354004, 1173.6877441
3: -656.1162720, 874.3712769, -633.6397095, 842.8861694, -1499.0023193, 1508.0107422
4: -646.2069092, 958.5946655, -623.8497314, 924.6898193, -1570.8967285, 1582.4443359

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1283335, upper bound: 1442.1334769
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -703.0906982, 896.4506836, -687.4572754, 876.8388672, -1579.9295654, 1583.9078369
1: -556.9985352, 720.3714600, -544.6519165, 704.5910034, -1261.5895996, 1265.0234375
2: -483.6549072, 708.6895752, -472.9653625, 693.2667236, -1176.9216309, 1181.6549072
3: -657.6641846, 875.0997925, -643.3664551, 855.9544678, -1513.6185303, 1518.4663086
4: -647.6637573, 959.9490967, -633.4328613, 939.1119385, -1586.7756348, 1593.3819580

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1283283, upper bound: 1442.1307189
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1282471, upper bound: 1442.1307132
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -689.9880981, 880.1305542, -708.9679565, 904.1557007, -1594.1437988, 1589.0985107
1: -546.6776733, 707.2580566, -561.7214355, 726.5818481, -1273.2592773, 1268.9794922
2: -474.7271118, 695.9012451, -487.7538147, 714.8319092, -1189.5590820, 1183.6550293
3: -645.7726440, 859.1928101, -663.2827148, 882.6629028, -1528.4355469, 1522.4752197
4: -635.8052979, 942.7105713, -653.1882935, 968.3211670, -1604.1264648, 1595.8988037

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307888, upper bound: 1442.1306216
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -696.8139648, 888.7787476, -708.4479980, 903.4838257, -1600.2977295, 1597.2266846
1: -551.9014893, 714.1286621, -561.3016968, 726.0416260, -1277.9431152, 1275.4304199
2: -479.3235168, 702.7254028, -487.3925476, 714.2994995, -1193.6228027, 1190.1179199
3: -652.0493164, 867.4312134, -662.7915649, 882.0031738, -1534.0524902, 1530.2227783
4: -641.9025269, 951.9027100, -652.7036133, 967.5987549, -1609.5012207, 1604.6063232

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307817, upper bound: 1442.1297784
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -707.8444824, 902.6947021, -708.9679565, 904.1557007, -1612.0000000, 1611.6625977
1: -560.8168945, 725.4074097, -561.7214355, 726.5818481, -1287.3984375, 1287.1287842
2: -486.9713745, 713.6730957, -487.7538147, 714.8319092, -1201.8032227, 1201.4268799
3: -662.2156982, 881.2315063, -663.2827148, 882.6629028, -1544.8786621, 1544.5140381
4: -652.1370239, 966.7495728, -653.1882935, 968.3211670, -1620.4582520, 1619.9378662

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1311608, upper bound: 1442.1350123
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -713.2902222, 909.5606079, -708.4479980, 903.4838257, -1616.7740479, 1618.0084229
1: -564.9342651, 730.8696289, -561.3016968, 726.0416260, -1290.9757080, 1292.1712646
2: -490.6171570, 719.1034546, -487.3925476, 714.2994995, -1204.9166260, 1206.4958496
3: -667.2324219, 887.7357788, -662.7915649, 882.0031738, -1549.2355957, 1550.5272217
4: -656.9620361, 974.0505981, -652.7036133, 967.5987549, -1624.5607910, 1626.7541504

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -711.9107056, 905.1764526, -698.9045410, 891.2833862, -1603.1940918, 1604.0810547
1: -563.0972900, 728.1683350, -553.6767578, 716.1932373, -1279.2905273, 1281.8450928
2: -488.9938660, 715.0632324, -480.7900696, 704.6320190, -1193.6258545, 1195.8531494
3: -664.3810425, 884.9911499, -653.8710938, 870.0370483, -1534.4180908, 1538.8621826
4: -655.0578003, 968.0361938, -643.8659058, 954.4996948, -1609.5574951, 1611.9018555

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297829, upper bound: 1442.1297728
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297829, upper bound: 1442.1298753
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -726.4152832, 924.4526367, -709.5701294, 904.9933472, -1631.4083252, 1634.0225830
1: -574.8096924, 743.6691284, -562.2149048, 727.2516479, -1302.0612793, 1305.8839111
2: -499.1498718, 730.3378906, -488.1798401, 715.5029297, -1214.6524658, 1218.5175781
3: -678.2272339, 903.8283081, -663.8747559, 883.4719849, -1561.6992188, 1567.7031250
4: -668.7432251, 988.7861938, -653.7642212, 969.2296143, -1637.9727783, 1642.5504150

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298337, upper bound: 1442.1310876
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298337, upper bound: 1442.1312674
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -721.0434570, 917.3115845, -689.7321167, 879.8262329, -1600.8696289, 1607.0434570
1: -570.3038330, 737.8226929, -546.4739380, 707.0107422, -1277.3145752, 1284.2966309
2: -495.3226318, 724.6693115, -474.5522766, 695.6589966, -1190.9816895, 1199.2215576
3: -673.1362305, 896.5509644, -645.5427856, 858.8906860, -1532.0268555, 1542.0937500
4: -663.5704346, 981.1716309, -635.5739746, 942.3801880, -1605.9505615, 1616.7454834

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297757, upper bound: 1442.1297674
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297715, upper bound: 1442.1307953
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -730.1359863, 929.2614746, -707.7836914, 902.6231689, -1632.7591553, 1637.0449219
1: -577.6275024, 747.4442139, -560.7701416, 725.3518677, -1302.9793701, 1308.2143555
2: -501.6637268, 734.1590576, -486.9317017, 713.6173706, -1215.2810059, 1221.0908203
3: -681.7904663, 908.2460938, -662.1626587, 881.1629639, -1562.9533691, 1570.4086914
4: -672.0994263, 994.0568237, -652.0856323, 966.6739502, -1638.7734375, 1646.1424561

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1318732, upper bound: 1442.1298705
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1318730, upper bound: 1442.1309094
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -712.4013062, 905.8169556, -712.4013062, 905.8169556, -1618.2182617, 1618.2182617
1: -563.4951172, 728.6843872, -563.4951172, 728.6843872, -1292.1794434, 1292.1794434
2: -489.3368835, 715.5709839, -489.3368835, 715.5709839, -1204.9078369, 1204.9078369
3: -664.8483887, 885.6200562, -664.8483887, 885.6200562, -1550.4683838, 1550.4683838
4: -655.5187378, 968.7247314, -655.5187378, 968.7247314, -1624.2434082, 1624.2434082

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297751, upper bound: 1442.1297728
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297674
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -712.4013062, 905.8169556, -726.8963013, 925.0805664, -1637.4814453, 1632.7132568
1: -563.4951172, 728.6843872, -575.1999512, 744.1749268, -1307.6699219, 1303.8842773
2: -489.3368835, 715.5709839, -499.4863892, 730.8354492, -1220.1723633, 1215.0573730
3: -664.8483887, 885.6200562, -678.6852417, 904.4452515, -1569.2934570, 1564.3052979
4: -655.5187378, 968.7247314, -669.1954346, 989.4607544, -1644.9794922, 1637.9199219

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297751, upper bound: 1442.1298204
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297678, upper bound: 1442.1298137
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -726.8963013, 925.0805664, -712.4013062, 905.8169556, -1632.7132568, 1637.4814453
1: -575.1999512, 744.1749268, -563.4951172, 728.6843872, -1303.8842773, 1307.6699219
2: -499.4863892, 730.8354492, -489.3368835, 715.5709839, -1215.0573730, 1220.1723633
3: -678.6852417, 904.4452515, -664.8483887, 885.6200562, -1564.3052979, 1569.2934570
4: -669.1954346, 989.4607544, -655.5187378, 968.7247314, -1637.9197998, 1644.9794922

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298204, upper bound: 1442.1308052
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298137, upper bound: 1442.1307953
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -726.8963013, 925.0805664, -726.8963013, 925.0805664, -1651.9765625, 1651.9766846
1: -575.1999512, 744.1749268, -575.1999512, 744.1749268, -1319.3748779, 1319.3748779
2: -499.4863892, 730.8354492, -499.4863892, 730.8354492, -1230.3217773, 1230.3217773
3: -678.6852417, 904.4452515, -678.6852417, 904.4452515, -1583.1303711, 1583.1303711
4: -669.1954346, 989.4607544, -669.1954346, 989.4607544, -1658.6562500, 1658.6562500

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298257, upper bound: 1442.1312674
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1298137, upper bound: 1442.1308612
time: 1.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.48 seconds
NS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297475, upper bound: 1442.1305835
NS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
NS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297757
NS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297757
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1283335, upper bound: 1442.1334769
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1283283, upper bound: 1442.1307189
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1282471, upper bound: 1442.1307132
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1307888, upper bound: 1442.1306216
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1307817, upper bound: 1442.1297784
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1311608, upper bound: 1442.1350123
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297829, upper bound: 1442.1297728
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297829, upper bound: 1442.1298753
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1298337, upper bound: 1442.1310876
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1298337, upper bound: 1442.1312674
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297757, upper bound: 1442.1297674
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297715, upper bound: 1442.1307953
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1318732, upper bound: 1442.1298705
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1318730, upper bound: 1442.1309094
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297751, upper bound: 1442.1297728
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297674
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297751, upper bound: 1442.1298204
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1297678, upper bound: 1442.1298137
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1298204, upper bound: 1442.1308052
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1298137, upper bound: 1442.1307953
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1298257, upper bound: 1442.1312674
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -1442.1298137, upper bound: 1442.1308612

## BFS NS instance: NS_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -689.9805298, 880.1211548, -687.7807617, 877.3681030, -1567.3486328, 1567.9017334
1: -546.6716309, 707.2503662, -544.9360962, 705.0509033, -1251.7225342, 1252.1865234
2: -474.7218933, 695.8937378, -473.2105408, 693.7230225, -1168.4447021, 1169.1042480
3: -645.7656250, 859.1834717, -643.6875000, 856.5255127, -1502.2911377, 1502.8706055
4: -635.7984009, 942.7003174, -633.7893066, 939.7423096, -1575.5405273, 1576.4895020

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 1.39 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -684.4347534, 872.8839111, -702.3145752, 896.1043091, -1580.5389404, 1575.1984863
1: -542.2254639, 701.4364624, -556.5597534, 720.1960449, -1262.4215088, 1257.9962158
2: -470.8709106, 690.1203003, -483.3245544, 708.4514771, -1179.3221436, 1173.4448242
3: -640.4824829, 852.1217651, -657.4505005, 875.0507202, -1515.5332031, 1509.5722656
4: -630.6512451, 934.8483887, -647.4488525, 959.6824341, -1590.3334961, 1582.2971191

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -696.8139648, 888.7787476, -690.6464844, 880.9864502, -1577.8004150, 1579.4252930
1: -551.9014893, 714.1286621, -547.2041626, 707.9437256, -1259.8452148, 1261.3327637
2: -479.3235168, 702.7254028, -475.1837769, 696.5791626, -1175.9024658, 1177.9091797
3: -652.0493164, 867.4312134, -646.3962402, 860.0280151, -1512.0773926, 1513.8273926
4: -641.9025269, 951.9027100, -636.4180298, 943.6284180, -1585.5308838, 1588.3208008

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297379, upper bound: 1442.1297329
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -696.8139648, 888.7787476, -697.5000000, 889.6689453, -1586.4829102, 1586.2788086
1: -551.9014893, 714.1286621, -552.4519043, 714.8428955, -1266.7443848, 1266.5804443
2: -479.3235168, 702.7254028, -479.8005066, 703.4314575, -1182.7548828, 1182.5258789
3: -652.0493164, 867.4312134, -652.7001343, 868.3010864, -1520.3503418, 1520.1313477
4: -641.9025269, 951.9027100, -642.5428467, 952.8612061, -1594.7636719, 1594.4455566

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297379, upper bound: 1442.1297328
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -700.8606567, 894.8194580, -673.9318848, 859.3472290, -1560.2076416, 1568.7513428
1: -555.8265991, 719.5892944, -533.8241577, 690.5106812, -1246.3372803, 1253.4134521
2: -482.3942566, 707.8365479, -463.5883789, 679.3548584, -1161.7491455, 1171.4249268
3: -656.1090698, 874.3617554, -630.5465088, 838.8566284, -1494.9656982, 1504.9078369
4: -646.1998901, 958.5842896, -620.8426514, 920.2271118, -1566.4270020, 1579.4270020

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334769
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334769
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -695.0895386, 887.4086304, -688.5253906, 878.1471558, -1573.2366943, 1575.9337158
1: -551.2145996, 713.6294556, -545.4984131, 705.7167358, -1256.9312744, 1259.1278076
2: -478.3991394, 701.9288330, -473.7453918, 694.1428833, -1172.5419922, 1175.6741943
3: -650.6641846, 867.1284790, -644.3701782, 857.4572754, -1508.1214600, 1511.4986572
4: -640.8798218, 950.5509644, -634.5590210, 940.2519531, -1581.1317139, 1585.1099854

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -703.0906982, 896.4506836, -687.0449219, 876.2933350, -1579.3839111, 1583.4953613
1: -556.9985352, 720.3714600, -544.3167725, 704.1512451, -1261.1497803, 1264.6879883
2: -483.6549072, 708.6895752, -472.6762695, 692.8337402, -1176.4886475, 1181.3658447
3: -657.6641846, 875.0997925, -642.9699707, 855.4179077, -1513.0820312, 1518.0698242
4: -647.6637573, 959.9490967, -633.0425415, 938.5247192, -1586.1884766, 1592.9916992

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279871, upper bound: 1442.1304373
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279952, upper bound: 1442.1306684
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -702.5786743, 895.7878418, -694.3105469, 885.4866333, -1588.0653076, 1590.0980225
1: -556.5848389, 719.8386230, -549.8942871, 711.4645996, -1268.0494385, 1269.7329102
2: -483.2988281, 708.1643066, -477.5779419, 700.0923462, -1183.3911133, 1185.7419434
3: -657.1799316, 874.4488525, -649.6499023, 864.1909790, -1521.3708496, 1524.0987549
4: -647.1860352, 959.2360840, -639.5445557, 948.3092041, -1595.4952393, 1598.7805176

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1280565, upper bound: 1442.1304441
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1280645, upper bound: 1442.1306755
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -689.9805298, 880.1211548, -705.5938110, 899.8388672, -1589.8192139, 1585.7148438
1: -546.6716309, 707.2503662, -559.0337524, 723.1263428, -1269.7979736, 1266.2841797
2: -474.7218933, 695.8937378, -485.4208984, 711.4219971, -1186.1435547, 1181.3145752
3: -645.7656250, 859.1834717, -660.0744629, 878.4691772, -1524.2348633, 1519.2576904
4: -635.7984009, 942.7003174, -650.0654297, 963.6833496, -1599.4816895, 1592.7657471

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -684.4347534, 872.8839111, -719.3776855, 917.8743896, -1602.3090820, 1592.2615967
1: -542.2254639, 701.4364624, -570.0855103, 737.6779785, -1279.9034424, 1271.5219727
2: -470.8709106, 690.1203003, -495.0349731, 725.5707397, -1196.4415283, 1185.1552734
3: -640.4824829, 852.1217651, -673.2255859, 896.2811890, -1536.7636719, 1525.3474121
4: -630.6512451, 934.8483887, -663.1051636, 982.8279419, -1613.4790039, 1597.9536133

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -696.8038940, 888.7661743, -705.0720215, 899.1647339, -1595.9686279, 1593.8380127
1: -551.8934937, 714.1186523, -558.6126099, 722.5844116, -1274.4779053, 1272.7312012
2: -479.3165588, 702.7153931, -485.0581970, 710.8879395, -1190.2044678, 1187.7735596
3: -652.0397949, 867.4188843, -659.5816040, 877.8074341, -1529.8471680, 1527.0004883
4: -641.8932495, 951.8890991, -649.5791016, 962.9588623, -1604.8520508, 1601.4682617

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -690.8789062, 881.0303345, -718.8586426, 917.1985474, -1608.0771484, 1599.8889160
1: -547.1398315, 707.9031982, -569.6656494, 737.1359253, -1284.2756348, 1277.5688477
2: -475.2011719, 696.5466919, -494.6737976, 725.0365601, -1200.2377930, 1191.2204590
3: -646.4002075, 859.8656006, -672.7338257, 895.6186523, -1542.0187988, 1532.5993652
4: -636.3849487, 943.5023804, -662.6199341, 982.1031494, -1618.4880371, 1606.1221924

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -707.8367310, 902.6850586, -705.5938110, 899.8388672, -1607.6755371, 1608.2788086
1: -560.8107300, 725.3995361, -559.0337524, 723.1263428, -1283.9370117, 1284.4333496
2: -486.9660950, 713.6654663, -485.4208984, 711.4219971, -1198.3880615, 1199.0861816
3: -662.2083740, 881.2218628, -660.0744629, 878.4691772, -1540.6774902, 1541.2962646
4: -652.1300049, 966.7391357, -650.0654297, 963.6833496, -1615.8133545, 1616.8045654

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -702.0393677, 895.2390137, -719.3776855, 917.8743896, -1619.9138184, 1614.6166992
1: -556.1742554, 719.4105225, -570.0855103, 737.6779785, -1293.8520508, 1289.4960938
2: -482.9547424, 707.7271729, -495.0349731, 725.5707397, -1208.5252686, 1202.7618408
3: -656.7381592, 873.9501953, -673.2255859, 896.2811890, -1553.0192871, 1547.1757812
4: -646.7847290, 958.6637573, -663.1051636, 982.8279419, -1629.6121826, 1621.7689209

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -713.2902222, 909.5606079, -708.5078125, 903.5532837, -1616.8433838, 1618.0681152
1: -564.9342651, 730.8696289, -561.3473511, 726.0953369, -1291.0294189, 1292.2166748
2: -490.6171570, 719.1034546, -487.4312134, 714.3536987, -1204.9708252, 1206.5344238
3: -667.2324219, 887.7357788, -662.8431396, 882.0694580, -1549.3018799, 1550.5786133
4: -656.9620361, 974.0505981, -652.7538452, 967.6720581, -1624.6340332, 1626.8044434

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303234, upper bound: 1442.1318669
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -713.2902222, 909.5606079, -713.8974609, 910.3539429, -1623.6441650, 1623.4580078
1: -564.9342651, 730.8696289, -565.4213867, 731.5044556, -1296.4385986, 1296.2906494
2: -490.6171570, 719.1034546, -491.0394287, 719.7327881, -1210.3498535, 1210.1428223
3: -667.2324219, 887.7357788, -667.8109741, 888.5097046, -1555.7420654, 1555.5467529
4: -656.9620361, 974.0505981, -657.5294800, 974.9053345, -1631.8674316, 1631.5800781

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303234, upper bound: 1442.1318669
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -711.9107056, 905.1764526, -690.3966064, 880.6719360, -1592.5823975, 1595.5729980
1: -563.0972900, 728.1683350, -547.0102539, 707.6943970, -1270.7915039, 1275.1785889
2: -488.9938660, 715.0632324, -475.0139465, 696.3309937, -1185.3248291, 1190.0771484
3: -664.3810425, 884.9911499, -646.1658325, 859.7251587, -1524.1059570, 1531.1569824
4: -655.0578003, 968.0361938, -636.1922607, 943.2933350, -1598.3509521, 1604.2282715

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1295541
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1295448
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -711.9107056, 905.1764526, -708.3037720, 903.2954712, -1615.2059326, 1613.4802246
1: -563.0972900, 728.1683350, -561.1901245, 725.8921509, -1288.9895020, 1289.3583984
2: -488.9938660, 715.0632324, -487.2931213, 714.1501465, -1203.1440430, 1202.3560791
3: -664.3810425, 884.9911499, -662.6538086, 881.8229980, -1546.2041016, 1547.6450195
4: -655.0578003, 968.0361938, -652.5703735, 967.3967285, -1622.4544678, 1620.6065674

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1296547
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1296574
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -726.4152832, 924.4526367, -690.3966064, 880.6719360, -1607.0866699, 1614.8492432
1: -574.8096924, 743.6691284, -547.0102539, 707.6943970, -1282.5041504, 1290.6794434
2: -499.1498718, 730.3378906, -475.0139465, 696.3309937, -1195.4808350, 1205.3516846
3: -678.2272339, 903.8283081, -646.1658325, 859.7251587, -1537.9521484, 1549.9941406
4: -668.7432251, 988.7861938, -636.1922607, 943.2933350, -1612.0363770, 1624.9785156

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297526, upper bound: 1442.1302075
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1301964
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -726.4152832, 924.4526367, -708.3037720, 903.2954712, -1629.7102051, 1632.7562256
1: -574.8096924, 743.6691284, -561.1901245, 725.8921509, -1300.7019043, 1304.8591309
2: -499.1498718, 730.3378906, -487.2931213, 714.1501465, -1213.2996826, 1217.6306152
3: -678.2272339, 903.8283081, -662.6538086, 881.8229980, -1560.0502930, 1566.4821777
4: -668.7432251, 988.7861938, -652.5703735, 967.3967285, -1636.1397705, 1641.3565674

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297526, upper bound: 1442.1302885
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1302818
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -714.3161621, 908.2828369, -689.7321167, 879.8262329, -1594.1423340, 1598.0148926
1: -564.8538818, 730.5629883, -546.4739380, 707.0107422, -1271.8646240, 1277.0368652
2: -490.5872803, 717.5067139, -474.5522766, 695.6589966, -1186.2463379, 1192.0589600
3: -666.6965332, 887.7454834, -645.5427856, 858.8906860, -1525.5871582, 1533.2883301
4: -657.1844482, 971.4321899, -635.5739746, 942.3801880, -1599.5645752, 1607.0059814

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1296480
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297200, upper bound: 1442.1296387
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -727.9987793, 926.5054321, -689.7321167, 879.8262329, -1607.8249512, 1616.2373047
1: -575.9147949, 745.2268066, -546.4739380, 707.0107422, -1282.9254150, 1291.7006836
2: -500.1804199, 731.9705811, -474.5522766, 695.6589966, -1195.8393555, 1206.5228271
3: -679.7800293, 905.5570068, -645.5427856, 858.8906860, -1538.6706543, 1551.0997314
4: -670.1142578, 991.0905151, -635.5739746, 942.3801880, -1612.4940186, 1626.6643066

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297368, upper bound: 1442.1304840
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1304712
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -714.3161621, 908.2828369, -707.7836914, 902.6231689, -1616.9390869, 1616.0665283
1: -564.8538818, 730.5629883, -560.7701416, 725.3518677, -1290.2058105, 1291.3331299
2: -490.5872803, 717.5067139, -486.9317017, 713.6173706, -1204.2045898, 1204.4382324
3: -666.6965332, 887.7454834, -662.1626587, 881.1629639, -1547.8594971, 1549.9082031
4: -657.1844482, 971.4321899, -652.0856323, 966.6739502, -1623.8583984, 1623.5178223

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1297487
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1297510
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -727.9987793, 926.5054321, -707.7836914, 902.6231689, -1630.6219482, 1634.2888184
1: -575.9147949, 745.2268066, -560.7701416, 725.3518677, -1301.2666016, 1305.9969482
2: -500.1804199, 731.9705811, -486.9317017, 713.6173706, -1213.7978516, 1218.9020996
3: -679.7800293, 905.5570068, -662.1626587, 881.1629639, -1560.9429932, 1567.7197266
4: -670.1142578, 991.0905151, -652.0856323, 966.6739502, -1636.7880859, 1643.1761475

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297368, upper bound: 1442.1305751
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1305728
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -711.9107056, 905.1764526, -712.4013062, 905.8169556, -1617.7276611, 1617.5776367
1: -563.0972900, 728.1683350, -563.4951172, 728.6843872, -1291.7817383, 1291.6634521
2: -488.9938660, 715.0632324, -489.3368835, 715.5709839, -1204.5648193, 1204.4001465
3: -664.3810425, 884.9911499, -664.8483887, 885.6200562, -1550.0010986, 1549.8394775
4: -655.0578003, 968.0361938, -655.5187378, 968.7247314, -1623.7822266, 1623.5548096

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297264, upper bound: 1442.1295541
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296272, upper bound: 1442.1295415
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -714.3161621, 908.2828369, -711.9229736, 905.1940308, -1619.5102539, 1620.2058105
1: -564.8538818, 730.5629883, -563.1085815, 728.1824951, -1293.0363770, 1293.6716309
2: -490.5872803, 717.5067139, -489.0040894, 715.0765991, -1205.6638184, 1206.5107422
3: -666.6965332, 887.7454834, -664.3955078, 885.0059204, -1551.7023926, 1552.1409912
4: -657.1844482, 971.4321899, -655.0712280, 968.0548706, -1625.2392578, 1626.5032959

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297674
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297674
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -711.9107056, 905.1764526, -726.8963013, 925.0805664, -1636.9908447, 1632.0727539
1: -563.0972900, 728.1683350, -575.1999512, 744.1749268, -1307.2720947, 1303.3682861
2: -488.9938660, 715.0632324, -499.4863892, 730.8354492, -1219.8293457, 1214.5495605
3: -664.3810425, 884.9911499, -678.6852417, 904.4452515, -1568.8261719, 1563.6763916
4: -655.0578003, 968.0361938, -669.1954346, 989.4607544, -1644.5185547, 1637.2313232

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307729, upper bound: 1442.1296017
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304622, upper bound: 1442.1295742
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -714.3161621, 908.2828369, -726.3984985, 924.4350586, -1638.7512207, 1634.6813965
1: -564.8538818, 730.5629883, -574.7987671, 743.6555786, -1308.5095215, 1305.3618164
2: -490.5872803, 717.5067139, -499.1408386, 730.3242798, -1220.9116211, 1216.6473389
3: -666.6965332, 887.7454834, -678.2155762, 903.8099365, -1570.5064697, 1565.9610596
4: -657.1844482, 971.4321899, -668.7313843, 988.7686768, -1645.9531250, 1640.1634521

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307781, upper bound: 1442.1296943
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304678, upper bound: 1442.1296666
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -726.8963013, 925.0805664, -711.9107056, 905.1764526, -1632.0727539, 1636.9909668
1: -575.1999512, 744.1749268, -563.0972900, 728.1683350, -1303.3682861, 1307.2720947
2: -499.4863892, 730.8354492, -488.9938660, 715.0632324, -1214.5495605, 1219.8293457
3: -678.6852417, 904.4452515, -664.3810425, 884.9911499, -1563.6763916, 1568.8261719
4: -669.1954346, 989.4607544, -655.0578003, 968.0361938, -1637.2313232, 1644.5185547

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296017, upper bound: 1442.1307729
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295742, upper bound: 1442.1304622
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -726.3984985, 924.4350586, -714.3161621, 908.2828369, -1634.6813965, 1638.7512207
1: -574.7987671, 743.6555786, -564.8538818, 730.5629883, -1305.3618164, 1308.5095215
2: -499.1408386, 730.3242798, -490.5872803, 717.5067139, -1216.6473389, 1220.9116211
3: -678.2155762, 903.8099365, -666.6965332, 887.7454834, -1565.9610596, 1570.5064697
4: -668.7313843, 988.7686768, -657.1844482, 971.4321899, -1640.1634521, 1645.9531250

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296942, upper bound: 1442.1307781
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296666, upper bound: 1442.1304678
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -726.4152832, 924.4526367, -726.8963013, 925.0805664, -1651.4953613, 1651.3488770
1: -574.8096924, 743.6691284, -575.1999512, 744.1749268, -1318.9846191, 1318.8691406
2: -499.1498718, 730.3378906, -499.4863892, 730.8354492, -1229.9852295, 1229.8242188
3: -678.2272339, 903.8283081, -678.6852417, 904.4452515, -1582.6722412, 1582.5135498
4: -668.7432251, 988.7861938, -669.1954346, 989.4607544, -1658.2039795, 1657.9816895

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1306787, upper bound: 1442.1302612
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303754, upper bound: 1442.1302339
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -727.9987793, 926.5054321, -726.3984985, 924.4350586, -1652.4338379, 1652.9039307
1: -575.9147949, 745.2268066, -574.7987671, 743.6555786, -1319.5703125, 1320.0256348
2: -500.1804199, 731.9705811, -499.1408386, 730.3242798, -1230.5046387, 1231.1112061
3: -679.7800293, 905.5570068, -678.2155762, 903.8099365, -1583.5899658, 1583.7724609
4: -670.1142578, 991.0905151, -668.7313843, 988.7686768, -1658.8829346, 1659.8218994

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308612, upper bound: 1442.1308612
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308612, upper bound: 1442.1308612
time: 1.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.64 seconds
NS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
NS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
NS_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
NS_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
NS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297379, upper bound: 1442.1297329
NS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
NS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297379, upper bound: 1442.1297328
NS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
NS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334769
NS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334769
NS_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
NS_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
NS_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1279871, upper bound: 1442.1304373
NS_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1279952, upper bound: 1442.1306684
NS_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1280565, upper bound: 1442.1304441
NS_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1280645, upper bound: 1442.1306755
NS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
NS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
NS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
NS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
NS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
NS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
NS_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
NS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304712, upper bound: 1442.1297505
NS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
NS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
NS_A1_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
NS_A1_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
NS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
NS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1303234, upper bound: 1442.1318669
NS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1309094, upper bound: 1442.1319098
NS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1303234, upper bound: 1442.1318669
NS_A2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1295541
NS_A2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1295448
NS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1296547
NS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1296574
NS_A2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297526, upper bound: 1442.1302075
NS_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1301964
NS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297526, upper bound: 1442.1302885
NS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1302818
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1296480
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297200, upper bound: 1442.1296387
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297368, upper bound: 1442.1304840
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1304712
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297285, upper bound: 1442.1297487
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1297510
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297368, upper bound: 1442.1305751
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1305728
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297264, upper bound: 1442.1295541
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296272, upper bound: 1442.1295415
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297674
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1297674, upper bound: 1442.1297674
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1307729, upper bound: 1442.1296017
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304622, upper bound: 1442.1295742
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1307781, upper bound: 1442.1296943
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1304678, upper bound: 1442.1296666
NS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296017, upper bound: 1442.1307729
NS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1295742, upper bound: 1442.1304622
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296942, upper bound: 1442.1307781
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1296666, upper bound: 1442.1304678
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1306787, upper bound: 1442.1302612
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1303754, upper bound: 1442.1302339
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1308612, upper bound: 1442.1308612
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -1442.1308612, upper bound: 1442.1308612

## BFS NS instance: NS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -686.7147217, 875.9714966, -687.7807617, 877.3681030, -1564.0827637, 1563.7521973
1: -544.0779419, 703.9296265, -544.9360962, 705.0509033, -1249.1289062, 1248.8656006
2: -472.4678650, 692.6155396, -473.2105408, 693.7230225, -1166.1909180, 1165.8260498
3: -642.6715088, 855.1582031, -643.6875000, 856.5255127, -1499.1968994, 1498.8453369
4: -632.7902832, 938.2423706, -633.7893066, 939.7423096, -1572.5324707, 1572.0313721

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297475, upper bound: 1442.1305811
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297475, upper bound: 1442.1305835
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -701.2518921, 894.7131958, -687.7807617, 877.3681030, -1578.6199951, 1582.4938965
1: -555.7041016, 719.0786133, -544.9360962, 705.0509033, -1260.7550049, 1264.0146484
2: -482.5839844, 707.3482056, -473.2105408, 693.7230225, -1176.3070068, 1180.5583496
3: -656.4377441, 873.6882935, -643.6875000, 856.5255127, -1512.9632568, 1517.3752441
4: -646.4528198, 958.1879883, -633.7893066, 939.7423096, -1586.1950684, 1591.9771729

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297475, upper bound: 1442.1305811
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297475, upper bound: 1442.1305835
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -686.7145996, 875.9712524, -702.3145752, 896.1043091, -1582.8188477, 1578.2858887
1: -544.0778198, 703.9295044, -556.5597534, 720.1960449, -1264.2739258, 1260.4890137
2: -472.4677734, 692.6153564, -483.3245544, 708.4514771, -1180.9191895, 1175.9399414
3: -642.6713257, 855.1580200, -657.4505005, 875.0507202, -1517.7220459, 1512.6083984
4: -632.7902832, 938.2420654, -647.4488525, 959.6824341, -1592.4724121, 1585.6907959

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -701.2518921, 894.7131958, -702.3145752, 896.1043091, -1597.3562012, 1597.0278320
1: -555.7041016, 719.0786133, -556.5597534, 720.1960449, -1275.9001465, 1275.6384277
2: -482.5839844, 707.3482056, -483.3245544, 708.4514771, -1191.0352783, 1190.6727295
3: -656.4377441, 873.6882935, -657.4505005, 875.0507202, -1531.4885254, 1531.1385498
4: -646.4528198, 958.1879883, -647.4488525, 959.6824341, -1606.1352539, 1605.6367188

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296484, upper bound: 1442.1305692
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -696.8038940, 888.7661743, -687.3734741, 876.8280640, -1573.6319580, 1576.1396484
1: -551.8934937, 714.1186523, -544.6045532, 704.6156616, -1256.5091553, 1258.7230225
2: -479.3165588, 702.7153931, -472.9246216, 693.2941895, -1172.6107178, 1175.6400146
3: -652.0397949, 867.4188843, -643.2953491, 855.9940796, -1508.0339355, 1510.7142334
4: -641.8932495, 951.8890991, -633.4034424, 939.1608887, -1581.0541992, 1585.2924805

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -690.8789062, 881.0303345, -701.9092407, 895.5660400, -1586.4444580, 1582.9394531
1: -547.1398315, 707.9031982, -556.2295532, 719.7625122, -1266.9022217, 1264.1328125
2: -475.2011719, 696.5466919, -483.0397949, 708.0242310, -1183.2253418, 1179.5864258
3: -646.4002075, 859.8656006, -657.0599976, 874.5216064, -1520.9218750, 1516.9255371
4: -636.3849487, 943.5023804, -647.0646973, 959.1030884, -1595.4880371, 1590.5665283

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -696.8038940, 888.7661743, -694.0567017, 885.3074951, -1582.1113281, 1582.8226318
1: -551.8934937, 714.1186523, -549.7192383, 711.3475342, -1263.2409668, 1263.8375244
2: -479.3165588, 702.7153931, -477.4199829, 699.9868774, -1179.3033447, 1180.1353760
3: -652.0397949, 867.4188843, -649.4438477, 864.0657959, -1516.1055908, 1516.8627930
4: -641.8932495, 951.8890991, -639.3731689, 948.1769409, -1590.0701904, 1591.2622070

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1296355, upper bound: 1442.1297117
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -690.8789062, 881.0303345, -707.9862061, 903.2471313, -1594.1254883, 1589.0164795
1: -547.1398315, 707.9031982, -560.8492432, 725.8541870, -1272.9940186, 1268.7524414
2: -475.2011719, 696.5466919, -487.1208496, 714.0888672, -1189.2900391, 1183.6674805
3: -646.4002075, 859.8656006, -662.6555176, 881.8013306, -1528.2015381, 1522.5211182
4: -636.3849487, 943.5023804, -652.4592285, 967.2731934, -1603.6579590, 1595.9614258

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1295448, upper bound: 1442.1297117
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -697.5025635, 890.5419312, -673.9318848, 859.3472290, -1556.8496094, 1564.4738770
1: -553.1518555, 716.1623535, -533.8241577, 690.5106812, -1243.6623535, 1249.9864502
2: -480.0737305, 704.4578247, -463.5883789, 679.3548584, -1159.4285889, 1168.0461426
3: -652.9224243, 870.2108154, -630.5465088, 838.8566284, -1491.7790527, 1500.7569580
4: -643.0993652, 953.9895020, -620.8426514, 920.2271118, -1563.3264160, 1574.8321533

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1283335, upper bound: 1442.1334769
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1283335, upper bound: 1442.1334769
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -711.4418335, 908.8177490, -673.9318848, 859.3472290, -1570.7888184, 1582.7496338
1: -564.3276367, 730.9151611, -533.8241577, 690.5106812, -1254.8383789, 1264.7392578
2: -489.8024597, 718.8297729, -463.5883789, 679.3548584, -1169.1573486, 1182.4182129
3: -666.2330322, 888.2699585, -630.5465088, 838.8566284, -1505.0895996, 1518.8161621
4: -656.2850342, 973.4319458, -620.8426514, 920.2271118, -1576.5122070, 1594.2746582

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1280939, upper bound: 1442.1334769
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1280939, upper bound: 1442.1334769
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -697.5025635, 890.5419312, -688.5253906, 878.1471558, -1575.6496582, 1579.0672607
1: -553.1518555, 716.1623535, -545.4984131, 705.7167358, -1258.8682861, 1261.6607666
2: -480.0737305, 704.4578247, -473.7453918, 694.1428833, -1174.2165527, 1178.2032471
3: -652.9224243, 870.2108154, -644.3701782, 857.4572754, -1510.3796387, 1514.5808105
4: -643.0993652, 953.9895020, -634.5590210, 940.2519531, -1583.3513184, 1588.5485840

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334953
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -711.4418335, 908.8177490, -688.5253906, 878.1471558, -1589.5889893, 1597.3428955
1: -564.3276367, 730.9151611, -545.4984131, 705.7167358, -1270.0443115, 1276.4135742
2: -489.8024597, 718.8297729, -473.7453918, 694.1428833, -1183.9453125, 1192.5751953
3: -666.2330322, 888.2699585, -644.3701782, 857.4572754, -1523.6903076, 1532.6400146
4: -656.2850342, 973.4319458, -634.5590210, 940.2519531, -1596.5366211, 1607.9909668

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334769
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1281872, upper bound: 1442.1334769
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -699.7302856, 892.1516113, -687.0373535, 876.2838135, -1576.0141602, 1579.1888428
1: -554.3219604, 716.9304199, -544.3106689, 704.1436768, -1258.4654541, 1261.2410889
2: -481.3314819, 705.2936401, -472.6711121, 692.8262329, -1174.1575928, 1177.9645996
3: -654.4689331, 870.9232788, -642.9627686, 855.4088135, -1509.8774414, 1513.8858643
4: -644.5538330, 955.3308105, -633.0355835, 938.5145874, -1583.0683594, 1588.3663330

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279871, upper bound: 1442.1304373
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279871, upper bound: 1442.1304373
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -713.4364624, 910.0868530, -681.5117798, 869.0739746, -1582.5104980, 1591.5986328
1: -565.3116455, 731.4014893, -539.8817139, 698.3522339, -1263.6636963, 1271.2830811
2: -490.8909302, 719.3641357, -468.8344116, 687.0753784, -1177.9663086, 1188.1982422
3: -667.5494995, 888.6370239, -637.6997070, 848.3745728, -1515.9239502, 1526.3366699
4: -657.5192261, 974.3706665, -627.9075928, 930.6939087, -1588.2131348, 1602.2781982

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279952, upper bound: 1442.1306684
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279952, upper bound: 1442.1306684
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -699.2170410, 891.4880981, -694.3005371, 885.4740601, -1584.6911621, 1585.7885742
1: -553.9075317, 716.3969116, -549.8862915, 711.4544678, -1265.3620605, 1266.2832031
2: -480.9747620, 704.7682495, -477.5708923, 700.0824585, -1181.0568848, 1182.3387451
3: -653.9839478, 870.2717285, -649.6406860, 864.1787109, -1518.1625977, 1519.9123535
4: -644.0752563, 954.6174927, -639.5354004, 948.2958984, -1592.3710938, 1594.1528320

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279871, upper bound: 1442.1304441
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1279871, upper bound: 1442.1304441
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -712.9208374, 909.4154663, -688.3532104, 877.7103271, -1590.6311035, 1597.7685547
1: -564.8944092, 730.8628540, -545.1141968, 705.2153320, -1270.1097412, 1275.9770508
2: -490.5320435, 718.8333740, -473.4396362, 693.8909302, -1184.4229736, 1192.2729492
3: -667.0609741, 887.9791260, -643.9801025, 856.5973511, -1523.6580811, 1531.9592285
4: -657.0372925, 973.6502686, -634.0062866, 939.8778687, -1596.9151611, 1607.6564941

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1280645, upper bound: 1442.1306755
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1280645, upper bound: 1442.1306755
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -686.7147217, 875.9714966, -705.5938110, 899.8388672, -1586.5534668, 1581.5653076
1: -544.0779419, 703.9296265, -559.0337524, 723.1263428, -1267.2041016, 1262.9633789
2: -472.4678650, 692.6155396, -485.4208984, 711.4219971, -1183.8897705, 1178.0363770
3: -642.6715088, 855.1582031, -660.0744629, 878.4691772, -1521.1406250, 1515.2322998
4: -632.7902832, 938.2423706, -650.0654297, 963.6833496, -1596.4736328, 1588.3077393

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307888, upper bound: 1442.1306216
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307888, upper bound: 1442.1306216
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -701.2518921, 894.7131958, -705.5938110, 899.8388672, -1601.0906982, 1600.3070068
1: -555.7041016, 719.0786133, -559.0337524, 723.1263428, -1278.8304443, 1278.1123047
2: -482.5839844, 707.3482056, -485.4208984, 711.4219971, -1194.0058594, 1192.7685547
3: -656.4377441, 873.6882935, -660.0744629, 878.4691772, -1534.9069824, 1533.7623291
4: -646.4528198, 958.1879883, -650.0654297, 963.6833496, -1610.1362305, 1608.2534180

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307888, upper bound: 1442.1306216
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307888, upper bound: 1442.1306216
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -686.7145996, 875.9712524, -719.3776855, 917.8743896, -1604.5888672, 1595.3488770
1: -544.0778198, 703.9295044, -570.0855103, 737.6779785, -1281.7556152, 1274.0148926
2: -472.4677734, 692.6153564, -495.0349731, 725.5707397, -1198.0385742, 1187.6503906
3: -642.6713257, 855.1580200, -673.2255859, 896.2811890, -1538.9525146, 1528.3835449
4: -632.7902832, 938.2420654, -663.1051636, 982.8279419, -1615.6179199, 1601.3471680

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -701.2518921, 894.7131958, -719.3776855, 917.8743896, -1619.1262207, 1614.0908203
1: -555.7041016, 719.0786133, -570.0855103, 737.6779785, -1293.3820801, 1289.1640625
2: -482.5839844, 707.3482056, -495.0349731, 725.5707397, -1208.1547852, 1202.3828125
3: -656.4377441, 873.6882935, -673.2255859, 896.2811890, -1552.7189941, 1546.9138184
4: -646.4528198, 958.1879883, -663.1051636, 982.8279419, -1629.2806396, 1621.2932129

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304788, upper bound: 1442.1305904
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -693.3710327, 884.4179688, -705.0720215, 899.1647339, -1592.5357666, 1589.4898682
1: -549.1693115, 710.6338501, -558.6126099, 722.5844116, -1271.7536621, 1269.2464600
2: -476.9433899, 699.2813110, -485.0581970, 710.8879395, -1187.8312988, 1184.3394775
3: -648.7937622, 863.1965942, -659.5816040, 877.8074341, -1526.6010742, 1522.7781982
4: -638.7336426, 947.2191772, -649.5791016, 962.9588623, -1601.6921387, 1596.7983398

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307817, upper bound: 1442.1297784
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307817, upper bound: 1442.1297784
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -707.3050537, 902.3626709, -705.0720215, 899.1647339, -1606.4697266, 1607.4345703
1: -560.3031006, 725.1447754, -558.6126099, 722.5844116, -1282.8873291, 1283.7573242
2: -486.6474304, 713.3880005, -485.0581970, 710.8879395, -1197.5354004, 1198.4460449
3: -662.0098267, 880.9370117, -659.5816040, 877.8074341, -1539.8171387, 1540.5185547
4: -651.8230591, 966.3221436, -649.5791016, 962.9588623, -1614.7819824, 1615.9012451

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307817, upper bound: 1442.1297784
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307817, upper bound: 1442.1297784
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -693.3710327, 884.4179688, -718.8586426, 917.1985474, -1610.5695801, 1603.2766113
1: -549.1693115, 710.6338501, -569.6656494, 737.1359253, -1286.3051758, 1280.2995605
2: -476.9433899, 699.2813110, -494.6737976, 725.0365601, -1201.9799805, 1193.9550781
3: -648.7937622, 863.1965942, -672.7338257, 895.6186523, -1544.4123535, 1535.9304199
4: -638.7336426, 947.2191772, -662.6199341, 982.1031494, -1620.8364258, 1609.8391113

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1301964, upper bound: 1442.1297350
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1301964, upper bound: 1442.1297350
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -707.3050537, 902.3626709, -718.8586426, 917.1985474, -1624.5036621, 1621.2213135
1: -560.3031006, 725.1447754, -569.6656494, 737.1359253, -1297.4387207, 1294.8104248
2: -486.6474304, 713.3880005, -494.6737976, 725.0365601, -1211.6839600, 1208.0615234
3: -662.0098267, 880.9370117, -672.7338257, 895.6186523, -1557.6284180, 1553.6708984
4: -651.8230591, 966.3221436, -662.6199341, 982.1031494, -1633.9262695, 1628.9420166

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1301964, upper bound: 1442.1297350
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1301964, upper bound: 1442.1297350
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -704.4721069, 898.3794556, -705.5938110, 899.8388672, -1604.3105469, 1603.9730225
1: -558.1304321, 721.9530029, -559.0337524, 723.1263428, -1281.2565918, 1280.9868164
2: -484.6394348, 710.2642212, -485.4208984, 711.4219971, -1196.0611572, 1195.6848145
3: -659.0085449, 877.0390625, -660.0744629, 878.4691772, -1537.4777832, 1537.1134033
4: -649.0157471, 962.1132202, -650.0654297, 963.6833496, -1612.6990967, 1612.1787109

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1311608, upper bound: 1442.1350123
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1311608, upper bound: 1442.1350123
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -718.3028564, 916.4730225, -705.5938110, 899.8388672, -1618.1417236, 1622.0668945
1: -569.2205200, 736.5518188, -559.0337524, 723.1263428, -1292.3466797, 1295.5855713
2: -494.2863159, 724.4593506, -485.4208984, 711.4219971, -1205.7082520, 1209.8801270
3: -672.2037964, 894.9083862, -660.0744629, 878.4691772, -1550.6729736, 1554.9827881
4: -662.0991821, 981.3210449, -650.0654297, 963.6833496, -1625.7824707, 1631.3864746

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1311608, upper bound: 1442.1350123
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1311608, upper bound: 1442.1350123
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -704.4720459, 898.3792725, -719.3776855, 917.8743896, -1622.3459473, 1617.7567139
1: -558.1303711, 721.9528809, -570.0855103, 737.6779785, -1295.8081055, 1292.0383301
2: -484.6393433, 710.2640991, -495.0349731, 725.5707397, -1210.2099609, 1205.2989502
3: -659.0085449, 877.0390625, -673.2255859, 896.2811890, -1555.2897949, 1550.2646484
4: -649.0157471, 962.1132202, -663.1051636, 982.8279419, -1631.8433838, 1625.2183838

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -718.3028564, 916.4730225, -719.3776855, 917.8743896, -1636.1772461, 1635.8507080
1: -569.2205200, 736.5518188, -570.0855103, 737.6779785, -1306.8981934, 1306.6373291
2: -494.2863159, 724.4593506, -495.0349731, 725.5707397, -1219.8570557, 1219.4942627
3: -672.2037964, 894.9083862, -673.2255859, 896.2811890, -1568.4849854, 1568.1340332
4: -662.0991821, 981.3210449, -663.1051636, 982.8279419, -1644.9268799, 1644.4262695

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1308467, upper bound: 1442.1349835
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -713.2800903, 909.5480347, -705.1349487, 899.2377319, -1612.5178223, 1614.6828613
1: -564.9262695, 730.8594360, -558.6607666, 722.6409912, -1287.5670166, 1289.5201416
2: -490.6101990, 719.0935059, -485.0992126, 710.9446411, -1201.5548096, 1204.1925049
3: -667.2231445, 887.7233887, -659.6358032, 877.8771362, -1545.1003418, 1547.3590088
4: -656.9527588, 974.0371094, -649.6321411, 963.0355835, -1619.9882812, 1623.6691895

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303162, upper bound: 1442.1316851
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303162, upper bound: 1442.1318669
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -707.7091064, 902.3357544, -718.9428711, 917.3018188, -1625.0109863, 1621.2785645
1: -560.4670410, 725.0665283, -569.7321777, 737.2168579, -1297.6838379, 1294.7987061
2: -486.7533264, 713.3432617, -494.7301331, 725.1169434, -1211.8701172, 1208.0731201
3: -661.9468384, 880.6878662, -672.8098145, 895.7186890, -1557.6655273, 1553.4976807
4: -651.8032227, 966.2215576, -662.6946411, 982.2123413, -1634.0156250, 1628.9162598

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303162, upper bound: 1442.1316851
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303162, upper bound: 1442.1318669
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -713.2800903, 909.5480347, -710.4070435, 905.9110718, -1619.1911621, 1619.9550781
1: -564.9262695, 730.8594360, -562.6477051, 727.9439697, -1292.8702393, 1293.5070801
2: -490.6101990, 719.0935059, -488.6249390, 716.2257690, -1206.8359375, 1207.7182617
3: -667.2231445, 887.7233887, -664.5023193, 884.1903076, -1551.4134521, 1552.2257080
4: -656.9527588, 974.0371094, -654.3065796, 970.1362305, -1627.0889893, 1628.3435059

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305251, upper bound: 1442.1316851
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305251, upper bound: 1442.1316851
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -707.7091064, 902.3357544, -724.4738770, 924.1488647, -1631.8579102, 1626.8094482
1: -560.4670410, 725.0665283, -573.9055176, 742.6952515, -1303.1623535, 1298.9720459
2: -486.7533264, 713.3432617, -498.4262085, 730.5652466, -1217.3186035, 1211.7694092
3: -661.9468384, 880.6878662, -677.8799438, 902.2337646, -1564.1806641, 1558.5676270
4: -651.8032227, 966.2215576, -667.5729980, 989.5516968, -1641.3547363, 1633.7944336

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305251, upper bound: 1442.1316851
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305251, upper bound: 1442.1318669
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -711.9038696, 905.1677856, -687.1222534, 876.5118408, -1588.4152832, 1592.2900391
1: -563.0917358, 728.1613770, -544.4095459, 704.3650513, -1267.4566650, 1272.5709229
2: -488.9890747, 715.0563354, -472.7539673, 693.0444946, -1182.0335693, 1187.8101807
3: -664.3744507, 884.9825439, -643.0637817, 855.6896973, -1520.0642090, 1528.0462646
4: -655.0513916, 968.0269165, -633.1766357, 938.8239746, -1593.8751221, 1601.2033691

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1295448
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1295448
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -706.1530151, 897.6214600, -701.6580811, 895.2524414, -1601.4055176, 1599.2795410
1: -558.4882202, 722.1083984, -556.0348511, 719.5127563, -1278.0009766, 1278.1433105
2: -484.9970703, 709.0484009, -482.8692322, 707.7759399, -1192.7729492, 1191.9176025
3: -658.8917847, 877.6290283, -656.8289185, 874.2181396, -1533.1097412, 1534.4580078
4: -649.6913452, 959.8591919, -646.8377686, 958.7681274, -1608.4591064, 1606.6967773

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1295448
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1295448
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -711.9038696, 905.1677856, -704.9301147, 898.9790649, -1610.8828125, 1610.0979004
1: -563.0917358, 728.1613770, -558.5026245, 722.4370728, -1285.5288086, 1286.6640625
2: -488.9890747, 715.0563354, -484.9604797, 710.7402954, -1199.7293701, 1200.0167236
3: -664.3744507, 884.9825439, -659.4459839, 877.6298218, -1542.0041504, 1544.4284668
4: -655.0513916, 968.0269165, -649.4478760, 962.7593384, -1617.8105469, 1617.4744873

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1316691, upper bound: 1442.1296547
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1316691, upper bound: 1442.1296547
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -706.1530151, 897.6214600, -718.7175293, 917.0205688, -1623.1735840, 1616.3389893
1: -558.4882202, 722.1083984, -569.5573120, 736.9924927, -1295.4804688, 1291.6656494
2: -484.9970703, 709.0484009, -494.5769348, 724.8933105, -1209.8903809, 1203.6253662
3: -658.8917847, 877.6290283, -672.6010742, 895.4461670, -1554.3377686, 1550.2301025
4: -649.6913452, 959.8591919, -662.4907837, 981.9095459, -1631.6007080, 1622.3498535

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1318513, upper bound: 1442.1296574
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1318513, upper bound: 1442.1296574
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -726.4082031, 924.4437256, -687.1222534, 876.5118408, -1602.9196777, 1611.5659180
1: -574.8040771, 743.6618652, -544.4095459, 704.3650513, -1279.1691895, 1288.0714111
2: -499.1449890, 730.3307495, -472.7539673, 693.0444946, -1192.1894531, 1203.0843506
3: -678.2203979, 903.8195190, -643.0637817, 855.6896973, -1533.9101562, 1546.8833008
4: -668.7366943, 988.7764282, -633.1766357, 938.8239746, -1607.5601807, 1621.9531250

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1301964
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1301964
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -720.5557861, 916.8103027, -701.6580811, 895.2524414, -1615.8081055, 1618.4683838
1: -570.1253662, 737.5362549, -556.0348511, 719.5127563, -1289.6378174, 1293.5710449
2: -495.0900574, 724.2508545, -482.8692322, 707.7759399, -1202.8659668, 1207.1201172
3: -672.6672363, 896.3801880, -656.8289185, 874.2181396, -1546.8853760, 1553.2091064
4: -663.2985229, 980.5155640, -646.8377686, 958.7681274, -1622.0665283, 1627.3532715

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1301964
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1301964
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -726.4082031, 924.4437256, -704.9301147, 898.9790649, -1625.3872070, 1629.3737793
1: -574.8040771, 743.6618652, -558.5026245, 722.4370728, -1297.2412109, 1302.1645508
2: -499.1449890, 730.3307495, -484.9604797, 710.7402954, -1209.8850098, 1215.2908936
3: -678.2203979, 903.8195190, -659.4459839, 877.6298218, -1555.8500977, 1563.2655029
4: -668.7366943, 988.7764282, -649.4478760, 962.7593384, -1631.4957275, 1638.2243652

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1314770, upper bound: 1442.1302818
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1314771, upper bound: 1442.1302818
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -720.5557861, 916.8103027, -718.7175293, 917.0205688, -1637.5764160, 1635.5278320
1: -570.1253662, 737.5362549, -569.5573120, 736.9924927, -1307.1174316, 1307.0933838
2: -495.0900574, 724.2508545, -494.5769348, 724.8933105, -1219.9832764, 1218.8277588
3: -672.6672363, 896.3801880, -672.6010742, 895.4461670, -1568.1134033, 1568.9812012
4: -663.2985229, 980.5155640, -662.4907837, 981.9095459, -1645.2080078, 1643.0063477

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1314770, upper bound: 1442.1302818
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1314771, upper bound: 1442.1302818
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -714.3065186, 908.2707520, -686.4577637, 875.6676636, -1589.9741211, 1594.7285156
1: -564.8461914, 730.5531616, -543.8737793, 703.6824951, -1268.5286865, 1274.4267578
2: -490.5805969, 717.4971313, -472.2922974, 692.3738403, -1182.9544678, 1189.7894287
3: -666.6875000, 887.7333984, -642.4410400, 854.8572388, -1521.5446777, 1530.1744385
4: -657.1755371, 971.4191895, -632.5589600, 937.9124756, -1595.0876465, 1603.9780273

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1296387
time: 1.00 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297200, upper bound: 1442.1296387
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -708.4073486, 900.5398560, -701.0026245, 894.4093018, -1602.8166504, 1601.5424805
1: -560.1237183, 724.3533325, -555.5048828, 718.8336182, -1278.9572754, 1279.8581543
2: -486.4862976, 711.3436890, -482.4136047, 707.1083374, -1193.5943604, 1193.7573242
3: -661.0693359, 880.2009277, -656.2119141, 873.3894043, -1534.4587402, 1536.4128418
4: -651.6824951, 963.0578003, -646.2258301, 957.8610229, -1609.5432129, 1609.2833252

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1296387
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297117, upper bound: 1442.1296387
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -727.9891357, 926.4932861, -686.4577637, 875.6676636, -1603.6567383, 1612.9509277
1: -575.9071655, 745.2171021, -543.8737793, 703.6824951, -1279.5894775, 1289.0906982
2: -500.1737366, 731.9609375, -472.2922974, 692.3738403, -1192.5476074, 1204.2531738
3: -679.7709351, 905.5451660, -642.4410400, 854.8572388, -1534.6281738, 1547.9862061
4: -670.1052856, 991.0773926, -632.5589600, 937.9124756, -1608.0173340, 1623.6363525

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297350, upper bound: 1442.1304712
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297505, upper bound: 1442.1304712
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -722.2969971, 919.0643921, -701.0026245, 894.4093018, -1616.7062988, 1620.0670166
1: -571.3610229, 739.2590332, -555.5048828, 718.8336182, -1290.1945801, 1294.7635498
2: -496.2315369, 726.0483398, -482.4136047, 707.1083374, -1203.3397217, 1208.4619141
3: -674.3695068, 898.3095093, -656.2119141, 873.3894043, -1547.7589111, 1554.5214844
4: -664.8197021, 983.0454712, -646.2258301, 957.8610229, -1622.6801758, 1629.2712402

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26

Time for candidate selection: 0.12 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.53 + 416.56 = 420.09 seconds
