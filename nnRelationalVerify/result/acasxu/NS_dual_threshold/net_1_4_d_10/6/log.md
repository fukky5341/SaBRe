## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 2204.5111029827913


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-780.0952148, 1252.1962891, -780.0952148, 1252.1962891, -2032.2915039, 2032.2915039)
1: (-876.9357910, 1279.8708496, -876.9357910, 1279.8708496, -2156.8066406, 2156.8066406)
2: (-884.4795532, 1279.2316895, -884.4795532, 1279.2316895, -2163.7106934, 2163.7106934)
3: (-1073.1099854, 1476.0378418, -1073.1099854, 1476.0378418, -2549.1479492, 2549.1479492)
4: (-971.7084351, 1472.0739746, -971.7084351, 1472.0739746, -2443.7822266, 2443.7824707)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.63 + 2.66 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -2204.5772403, upper bound: 2204.5772403

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5406616, upper bound: 2204.5074876
time: 1.09 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5771713, upper bound: 2204.5771712
time: 1.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.38 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.38
Output dim: 3, lower bound: -2204.5406616, upper bound: 2204.5074876
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.38
Output dim: 3, lower bound: -2204.5771713, upper bound: 2204.5771712

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -777.6053467, 1248.1035156, -746.0270996, 1195.2812500, -1972.8864746, 1994.1304932
1: -874.1688232, 1275.6961670, -838.7745361, 1221.9118652, -2096.0798340, 2114.4702148
2: -881.6210938, 1275.0347900, -845.3336792, 1221.3171387, -2102.9377441, 2120.3684082
3: -1069.7530518, 1471.2385254, -1026.6439209, 1409.3378906, -2479.0905762, 2497.8811035
4: -968.4663086, 1467.2651367, -927.8871460, 1405.6160889, -2374.0825195, 2395.1523438

Time for backsubstitution: 0.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.5070976, upper bound: 2204.5070976
time: 1.06 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.5070976, upper bound: 2204.5074876
time: 1.18 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -780.0952148, 1252.1962891, -780.0045166, 1252.0537109, -2032.1486816, 2032.2006836
1: -876.9357910, 1279.8708496, -876.8342896, 1279.7246094, -2156.6604004, 2156.7050781
2: -884.4795532, 1279.2316895, -884.3770752, 1279.0852051, -2163.5646973, 2163.6086426
3: -1073.1099854, 1476.0378418, -1072.9860840, 1475.8693848, -2548.9792480, 2549.0239258
4: -971.7084351, 1472.0739746, -971.5972900, 1471.9053955, -2443.6137695, 2443.6711426

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5260077, upper bound: 2204.5454825
time: 1.38 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5278023, upper bound: 2204.5278023
time: 1.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.03 seconds
NS_B1_A1, status: Status.VERIFIED, split count: 2, time: 3.03
Output dim: 3, lower bound: -2204.5070976, upper bound: 2204.5070976
NS_B1_A2, status: Status.VERIFIED, split count: 2, time: 3.03
Output dim: 3, lower bound: -2204.5070976, upper bound: 2204.5074876
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 3, lower bound: -2204.5260077, upper bound: 2204.5454825
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 3, lower bound: -2204.5278023, upper bound: 2204.5278023

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -780.0952148, 1252.1962891, -765.6796265, 1227.1473389, -2007.2424316, 2017.8756104
1: -876.9357910, 1279.8708496, -860.5095825, 1254.3399658, -2131.2756348, 2140.3803711
2: -884.4795532, 1279.2316895, -868.0309448, 1253.9733887, -2138.4523926, 2147.2626953
3: -1073.1099854, 1476.0378418, -1052.6195068, 1446.8366699, -2519.9467773, 2528.6572266
4: -971.7084351, 1472.0739746, -953.7347412, 1442.8133545, -2414.5217285, 2425.8085938

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5077706, upper bound: 2204.5396915
time: 1.35 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5244650, upper bound: 2204.5444219
time: 1.07 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -778.1378784, 1248.8664551, -895.3894043, 1430.3947754, -2208.5324707, 2144.2553711
1: -874.7072144, 1276.4447021, -1006.0085449, 1462.3494873, -2337.0563965, 2282.4531250
2: -882.2340698, 1275.7270508, -1015.0972290, 1461.1177979, -2343.3518066, 2290.8242188
3: -1070.3848877, 1472.2448730, -1227.2722168, 1688.0335693, -2758.4184570, 2699.5170898
4: -969.2272339, 1468.1748047, -1118.3990479, 1681.9255371, -2651.1518555, 2586.5734863

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5278023, upper bound: 2204.5260077
time: 1.19 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5278023, upper bound: 2204.5278023
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.80 seconds
NS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 3, lower bound: -2204.5077706, upper bound: 2204.5396915
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 3, lower bound: -2204.5244650, upper bound: 2204.5444219
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 3, lower bound: -2204.5278023, upper bound: 2204.5260077
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 3, lower bound: -2204.5278023, upper bound: 2204.5278023

## BFS NS instance: NS_B2_B1_A1

### Backsubstitution after applying NS history:
0: -767.3154297, 1231.0120850, -762.8641968, 1222.6284180, -1989.9438477, 1993.8762207
1: -862.5448608, 1258.3776855, -857.3363647, 1249.7271729, -2112.2714844, 2115.7133789
2: -869.8714600, 1257.6816406, -864.8281860, 1249.3510742, -2119.2221680, 2122.5097656
3: -1055.5042725, 1451.2878418, -1048.7880859, 1441.5212402, -2497.0253906, 2500.0759277
4: -955.7076416, 1447.2781982, -950.1782227, 1437.4941406, -2393.2016602, 2397.4560547

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5067410, upper bound: 2204.5364426
time: 1.03 seconds

## Relational analysis of NS_B2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
time: 1.09 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: -779.8619385, 1251.8294678, -765.6796265, 1227.1473389, -2007.0092773, 2017.5086670
1: -876.6753540, 1279.4943848, -860.5095825, 1254.3399658, -2131.0153809, 2140.0039062
2: -884.2153931, 1278.8559570, -868.0309448, 1253.9733887, -2138.1884766, 2146.8869629
3: -1072.7932129, 1475.6033936, -1052.6195068, 1446.8366699, -2519.6298828, 2528.2229004
4: -971.4189453, 1471.6417236, -953.7347412, 1442.8133545, -2414.2324219, 2425.3764648

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5244650, upper bound: 2204.5444219
time: 1.07 seconds

## Relational analysis of NS_B2_B1_A2_A2

### Relational analysis result of NS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5244650, upper bound: 2204.5444219
time: 1.12 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -765.7679443, 1227.2880859, -895.3894043, 1430.3947754, -2196.1625977, 2122.6765137
1: -860.6093140, 1254.4846191, -1006.0085449, 1462.3494873, -2322.9584961, 2260.4926758
2: -868.1311646, 1254.1179199, -1015.0972290, 1461.1177979, -2329.2490234, 2269.2150879
3: -1052.7419434, 1447.0034180, -1227.2722168, 1688.0335693, -2740.7753906, 2674.2756348
4: -953.8444824, 1442.9805908, -1118.3990479, 1681.9255371, -2635.7690430, 2561.3793945

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5253781, upper bound: 2204.5219489
time: 1.03 seconds

## Relational analysis of NS_B2_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5219489
time: 1.22 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -895.4413452, 1430.4816895, -895.3894043, 1430.3947754, -2325.8361816, 2325.8701172
1: -1006.0681763, 1462.4388428, -1006.0085449, 1462.3494873, -2468.4174805, 2468.4472656
2: -1015.1561890, 1461.2092285, -1015.0972290, 1461.1177979, -2476.2739258, 2476.3063965
3: -1227.3483887, 1688.1374512, -1227.2722168, 1688.0335693, -2915.3818359, 2915.4096680
4: -1118.4635010, 1682.0313721, -1118.3990479, 1681.9255371, -2800.3881836, 2800.4304199

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5253781, upper bound: 2204.5236460
time: 1.14 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5236460
time: 0.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.14 seconds
NS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5067410, upper bound: 2204.5364426
NS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
NS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5244650, upper bound: 2204.5444219
NS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5244650, upper bound: 2204.5444219
NS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5253781, upper bound: 2204.5219489
NS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5219489
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5253781, upper bound: 2204.5236460
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5236460

## BFS NS instance: NS_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -751.1117554, 1205.3540039, -751.5965576, 1204.6949463, -1955.8066406, 1956.9505615
1: -844.3523560, 1232.0784912, -844.6928101, 1231.3564453, -2075.7082520, 2076.7709961
2: -851.5693970, 1231.4956055, -852.0974731, 1231.0534668, -2082.6228027, 2083.5927734
3: -1033.2420654, 1420.9379883, -1033.2857666, 1420.3509521, -2453.5930176, 2454.2236328
4: -935.7701416, 1417.1870117, -936.3186035, 1416.4788818, -2352.2485352, 2353.5056152

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
time: 1.27 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
time: 1.14 seconds

## BFS NS instance: NS_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -950.0399780, 1524.7962646, -748.0736694, 1198.8287354, -2148.8686523, 2272.8698730
1: -1067.5468750, 1558.5821533, -840.7996826, 1225.4281006, -2292.9750977, 2399.3818359
2: -1076.2495117, 1556.9345703, -848.1297607, 1224.9749756, -2301.2246094, 2405.0642090
3: -1305.6639404, 1797.3331299, -1028.3195801, 1413.4842529, -2719.1481934, 2825.6528320
4: -1179.2060547, 1791.8012695, -931.9989014, 1409.3486328, -2588.5541992, 2723.8000488

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A1_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
time: 1.19 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
time: 1.27 seconds

## BFS NS instance: NS_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -765.5635986, 1226.9681396, -765.6796265, 1227.1473389, -1992.7109375, 1992.6473389
1: -860.3803101, 1254.1552734, -860.5095825, 1254.3399658, -2114.7202148, 2114.6647949
2: -867.8999023, 1253.7900391, -868.0309448, 1253.9733887, -2121.8730469, 2121.8210449
3: -1052.4631348, 1446.6229248, -1052.6195068, 1446.8366699, -2499.2998047, 2499.2424316
4: -953.5901489, 1442.6029053, -953.7347412, 1442.8133545, -2396.4035645, 2396.3376465

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A1_A1

### Relational analysis result of NS_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237456, upper bound: 2204.5411407
time: 1.10 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2

### Relational analysis result of NS_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
time: 1.26 seconds

## BFS NS instance: NS_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -895.2436523, 1430.1636963, -765.6796265, 1227.1473389, -2122.3906250, 2195.8427734
1: -1005.8454590, 1462.1127930, -860.5095825, 1254.3399658, -2260.1855469, 2322.6220703
2: -1014.9321899, 1460.8818359, -868.0309448, 1253.9733887, -2268.9055176, 2328.9128418
3: -1227.0740967, 1687.7619629, -1052.6195068, 1446.8366699, -2673.9106445, 2740.3811035
4: -1118.2197266, 1681.6555176, -953.7347412, 1442.8133545, -2561.0332031, 2635.3901367

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5439395
time: 1.40 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
time: 1.13 seconds

## BFS NS instance: NS_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -749.3162842, 1201.1799316, -883.7517090, 1411.7296143, -2161.0458984, 2084.9316406
1: -842.1754150, 1227.7322998, -992.9671021, 1443.2984619, -2285.4738770, 2220.6994629
2: -849.5438843, 1227.4807129, -1001.9619751, 1442.1173096, -2291.6608887, 2229.4426270
3: -1030.1467285, 1416.1391602, -1211.1920166, 1666.0797119, -2696.2265625, 2627.3308105
4: -933.5875244, 1412.3710938, -1104.1816406, 1660.1064453, -2593.6938477, 2516.5522461

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
time: 1.17 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
time: 0.95 seconds

## BFS NS instance: NS_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -947.7670898, 1520.2650146, -880.3927612, 1405.8702393, -2353.6369629, 2400.6574707
1: -1064.8012695, 1553.7296143, -989.2687988, 1437.4731445, -2502.2744141, 2542.9985352
2: -1073.6849365, 1552.4042969, -998.1936646, 1436.0764160, -2509.7612305, 2550.5979004
3: -1301.9499512, 1791.9694824, -1206.4279785, 1659.3503418, -2961.3002930, 2998.3969727
4: -1176.3287354, 1786.4251709, -1100.1324463, 1652.9736328, -2829.3020020, 2885.9721680

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A1_A2_B1

### Relational analysis result of NS_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
time: 1.29 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
time: 1.05 seconds

## BFS NS instance: NS_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -878.4448242, 1403.2790527, -883.7517090, 1411.7296143, -2290.1743164, 2287.0307617
1: -987.0339966, 1434.6789551, -992.9671021, 1443.2984619, -2430.3325195, 2427.6459961
2: -995.9737549, 1433.5321045, -1001.9619751, 1442.1173096, -2438.0910645, 2435.4941406
3: -1203.9013672, 1656.1054688, -1211.1920166, 1666.0797119, -2869.9807129, 2867.2968750
4: -1097.6905518, 1650.2377930, -1104.1816406, 1660.1064453, -2757.7968750, 2754.4194336

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5031313, upper bound: 2204.5236460
time: 1.00 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5236460
time: 1.22 seconds

## BFS NS instance: NS_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1075.5578613, 1721.3309326, -880.3927612, 1405.8702393, -2481.4282227, 2601.7231445
1: -1207.9868164, 1759.3670654, -989.2687988, 1437.4731445, -2645.4599609, 2748.6357422
2: -1218.4506836, 1757.5551758, -998.1936646, 1436.0764160, -2654.5270996, 2755.7487793
3: -1474.0344238, 2029.9600830, -1206.4279785, 1659.3503418, -3133.3842773, 3236.3874512
4: -1338.1461182, 2022.6143799, -1100.1324463, 1652.9736328, -2991.1196289, 3122.0312500

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B2_A2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5031310, upper bound: 2204.5236460
time: 1.08 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5236460
time: 1.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.00 seconds
NS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
NS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
NS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
NS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5038298, upper bound: 2204.5364426
NS_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5237456, upper bound: 2204.5411407
NS_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
NS_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5439395
NS_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
NS_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
NS_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
NS_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
NS_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5421519, upper bound: 2204.5219494
NS_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5031313, upper bound: 2204.5236460
NS_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5236460
NS_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5031310, upper bound: 2204.5236460
NS_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 3, lower bound: -2204.5219489, upper bound: 2204.5236460

## BFS NS instance: NS_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -751.1117554, 1205.3540039, -746.4564209, 1196.5949707, -1947.7067871, 1951.8101807
1: -844.3523560, 1232.0784912, -838.9436035, 1223.0583496, -2067.4104004, 2071.0219727
2: -851.5693970, 1231.4956055, -846.2908325, 1222.7972412, -2074.3662109, 2077.7863770
3: -1033.2420654, 1420.9379883, -1026.2580566, 1410.7540283, -2443.9960938, 2447.1955566
4: -935.7701416, 1417.1870117, -929.9824829, 1406.9802246, -2342.7500000, 2347.1694336

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5040352, upper bound: 2204.5325534
time: 1.22 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4960948, upper bound: 2204.5319422
time: 1.23 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -751.1117554, 1205.3540039, -945.0031738, 1515.7468262, -2266.8586426, 2150.3571777
1: -844.3523560, 1232.0784912, -1061.7116699, 1549.1315918, -2393.4831543, 2293.7900391
2: -851.5693970, 1231.4956055, -1070.5357666, 1547.8046875, -2399.3740234, 2302.0312500
3: -1033.2420654, 1420.9379883, -1298.1798096, 1786.6791992, -2819.9213867, 2719.1176758
4: -935.7701416, 1417.1870117, -1172.8282471, 1781.1234131, -2716.8928223, 2590.0151367

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5059904, upper bound: 2204.5355103
time: 1.32 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5055371, upper bound: 2204.5358810
time: 1.12 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -950.0399780, 1524.7962646, -746.4564209, 1196.5949707, -2146.6350098, 2271.2521973
1: -1067.5468750, 1558.5821533, -838.9436035, 1223.0583496, -2290.6052246, 2397.5258789
2: -1076.2495117, 1556.9345703, -846.2908325, 1222.7972412, -2299.0461426, 2403.2250977
3: -1305.6639404, 1797.3331299, -1026.2580566, 1410.7540283, -2716.4179688, 2823.5908203
4: -1179.2060547, 1791.8012695, -929.9824829, 1406.9802246, -2586.1857910, 2721.7836914

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5008131, upper bound: 2204.5325534
time: 1.33 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4924523, upper bound: 2204.5319422
time: 1.58 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -950.0399780, 1524.7962646, -945.0031738, 1515.7468262, -2465.7866211, 2469.7993164
1: -1067.5468750, 1558.5821533, -1061.7116699, 1549.1315918, -2616.6782227, 2620.2939453
2: -1076.2495117, 1556.9345703, -1070.5357666, 1547.8046875, -2624.0541992, 2627.4699707
3: -1305.6639404, 1797.3331299, -1298.1798096, 1786.6791992, -3092.3432617, 3095.5129395
4: -1179.2060547, 1791.8012695, -1172.8282471, 1781.1234131, -2960.3288574, 2964.6293945

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5008131, upper bound: 2204.5325534
time: 1.19 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4924523, upper bound: 2204.5319422
time: 0.99 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -749.1112671, 1200.8592529, -754.3788452, 1209.1601562, -1958.2711182, 1955.2380371
1: -841.9461060, 1227.4023438, -847.8377686, 1235.9063721, -2077.8520508, 2075.2397461
2: -849.3118896, 1227.1525879, -855.2628174, 1235.6130371, -2084.9248047, 2082.4147949
3: -1029.8669434, 1415.7583008, -1037.0678711, 1425.5930176, -2455.4594727, 2452.8261719
4: -933.3326416, 1411.9925537, -939.8285522, 1421.7260742, -2355.0585938, 2351.8210449

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A1_A1_B1

### Relational analysis result of NS_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
time: 1.10 seconds

## Relational analysis of NS_B2_B1_A2_A1_A1_B2

### Relational analysis result of NS_B2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
time: 1.08 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -947.5657349, 1519.9526367, -750.8504639, 1203.2819824, -2150.8471680, 2270.8029785
1: -1064.5758057, 1553.4077148, -843.9389038, 1229.9667969, -2294.5424805, 2397.3466797
2: -1073.4567871, 1552.0844727, -851.2885742, 1229.5241699, -2302.9809570, 2403.3730469
3: -1301.6760254, 1791.5980225, -1032.0936279, 1418.7126465, -2720.3886719, 2823.6916504
4: -1176.0791016, 1786.0567627, -935.5029297, 1414.5836182, -2590.6625977, 2721.5595703

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A1_A2_B1

### Relational analysis result of NS_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
time: 0.93 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2_B2

### Relational analysis result of NS_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
time: 1.09 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -883.6069336, 1411.5008545, -749.2294312, 1201.0413818, -2084.6484375, 2160.7302246
1: -992.8060303, 1443.0640869, -842.0776367, 1227.5903320, -2220.3964844, 2285.1416016
2: -1001.7983398, 1441.8846436, -849.4456177, 1227.3383789, -2229.1367188, 2291.3300781
3: -1210.9953613, 1665.8110352, -1030.0261230, 1415.9755859, -2626.9707031, 2695.8371582
4: -1104.0039062, 1659.8387451, -933.4797974, 1412.2065430, -2516.2104492, 2593.3186035

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B1_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
time: 1.02 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_A2

### Relational analysis result of NS_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
time: 1.42 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -880.2460938, 1405.6376953, -947.6765747, 1520.1228027, -2400.3686523, 2353.3142090
1: -989.1049194, 1437.2348633, -1064.6992188, 1553.5836182, -2542.6884766, 2501.9340820
2: -998.0276489, 1435.8399658, -1073.5823975, 1552.2584229, -2550.2861328, 2509.4223633
3: -1206.2283936, 1659.0778809, -1301.8259277, 1791.8009033, -2998.0292969, 2960.9035645
4: -1099.9525146, 1652.7019043, -1176.2172852, 1786.2565918, -2885.5988770, 2828.9189453

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_B1_A2_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
time: 1.35 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
time: 1.34 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -749.3162842, 1201.1799316, -878.3933716, 1403.1925049, -2152.5087891, 2079.5732422
1: -842.1754150, 1227.7322998, -986.9750977, 1434.5900879, -2276.7656250, 2214.7075195
2: -849.5438843, 1227.4807129, -995.9152832, 1433.4410400, -2282.9841309, 2223.3957520
3: -1030.1467285, 1416.1391602, -1203.8254395, 1656.0020752, -2686.1486816, 2619.9645996
4: -933.5875244, 1412.3710938, -1097.6263428, 1650.1330566, -2583.7207031, 2509.9975586

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5445760, upper bound: 2204.5209686
time: 1.11 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5445619, upper bound: 2204.5213602
time: 1.05 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -749.3162842, 1201.1799316, -1075.5012207, 1721.2386475, -2470.5549316, 2276.6806641
1: -842.1754150, 1227.7322998, -1207.9224854, 1759.2728271, -2601.4482422, 2435.6547852
2: -849.5438843, 1227.4807129, -1218.3870850, 1757.4587402, -2607.0026855, 2445.8669434
3: -1030.1467285, 1416.1391602, -1473.9526367, 2029.8526611, -3059.9990234, 2890.0917969
4: -933.5875244, 1412.3710938, -1338.0777588, 2022.5037842, -2956.0913086, 2750.4487305

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5445760, upper bound: 2204.5209686
time: 1.18 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5445619, upper bound: 2204.5213602
time: 1.15 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -947.7670898, 1520.2650146, -878.3933716, 1403.1925049, -2350.9592285, 2398.6582031
1: -1064.8012695, 1553.7296143, -986.9750977, 1434.5900879, -2499.3913574, 2540.7045898
2: -1073.6849365, 1552.4042969, -995.9152832, 1433.4410400, -2507.1250000, 2548.3195801
3: -1301.9499512, 1791.9694824, -1203.8254395, 1656.0020752, -2957.9521484, 2995.7949219
4: -1176.3287354, 1786.4251709, -1097.6263428, 1650.1330566, -2826.4616699, 2883.4372559

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_A2_B1_A1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5326655, upper bound: 2204.5153418
time: 1.01 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5404259, upper bound: 2204.5153418
time: 1.31 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -947.7670898, 1520.2650146, -1075.5012207, 1721.2386475, -2669.0051270, 2595.7658691
1: -1064.8012695, 1553.7296143, -1207.9224854, 1759.2728271, -2824.0742188, 2761.6520996
2: -1073.6849365, 1552.4042969, -1218.3870850, 1757.4587402, -2831.1435547, 2770.7912598
3: -1301.9499512, 1791.9694824, -1473.9526367, 2029.8526611, -3331.8024902, 3265.9218750
4: -1176.3287354, 1786.4251709, -1338.0777588, 2022.5037842, -3198.8322754, 3122.5883789

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A1_A2_B2_A1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5419296, upper bound: 2204.5192530
time: 1.32 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5419468, upper bound: 2204.5212833
time: 1.22 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -878.4448242, 1403.2790527, -878.3933716, 1403.1925049, -2281.6372070, 2281.6723633
1: -987.0339966, 1434.6789551, -986.9750977, 1434.5900879, -2421.6240234, 2421.6540527
2: -995.9737549, 1433.5321045, -995.9152832, 1433.4410400, -2429.4143066, 2429.4472656
3: -1203.9013672, 1656.1054688, -1203.8254395, 1656.0020752, -2859.9028320, 2859.9309082
4: -1097.6905518, 1650.2377930, -1097.6263428, 1650.1330566, -2747.8237305, 2747.8642578

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4985453, upper bound: 2204.5168143
time: 1.07 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5189289, upper bound: 2204.5168143
time: 1.17 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -878.4448242, 1403.2790527, -1075.5012207, 1721.2386475, -2599.6833496, 2478.7800293
1: -987.0339966, 1434.6789551, -1207.9224854, 1759.2728271, -2746.3068848, 2642.6015625
2: -995.9737549, 1433.5321045, -1218.3870850, 1757.4587402, -2753.4326172, 2651.9191895
3: -1203.9013672, 1656.1054688, -1473.9526367, 2029.8526611, -3233.7529297, 3130.0576172
4: -1097.6905518, 1650.2377930, -1338.0777588, 2022.5037842, -3119.4340820, 2988.3154297

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5249306, upper bound: 2204.5226002
time: 0.95 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5245245, upper bound: 2204.5230168
time: 1.17 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1075.5578613, 1721.3309326, -878.3933716, 1403.1925049, -2478.7504883, 2599.7236328
1: -1207.9868164, 1759.3670654, -986.9750977, 1434.5900879, -2642.5769043, 2746.3422852
2: -1218.4506836, 1757.5551758, -995.9152832, 1433.4410400, -2651.8908691, 2753.4699707
3: -1474.0344238, 2029.9600830, -1203.8254395, 1656.0020752, -3130.0361328, 3233.7856445
4: -1338.1461182, 2022.6143799, -1097.6263428, 1650.1330566, -2988.2792969, 3119.4965820

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5161138, upper bound: 2204.5168252
time: 1.31 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5168252
time: 1.25 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1075.5578613, 1721.3309326, -1075.5012207, 1721.2386475, -2796.7890625, 2796.8293457
1: -1207.9868164, 1759.3670654, -1207.9224854, 1759.2728271, -2967.2595215, 2967.2895508
2: -1218.4506836, 1757.5551758, -1218.3870850, 1757.4587402, -2975.9094238, 2975.9416504
3: -1474.0344238, 2029.9600830, -1473.9526367, 2029.8526611, -3503.8864746, 3503.9123535
4: -1338.1461182, 2022.6143799, -1338.0777588, 2022.5037842, -3358.5895996, 3358.6479492

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5161138, upper bound: 2204.5168251
time: 1.49 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5168252
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.19 seconds
NS_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5040352, upper bound: 2204.5325534
NS_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.4960948, upper bound: 2204.5319422
NS_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5059904, upper bound: 2204.5355103
NS_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5055371, upper bound: 2204.5358810
NS_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5008131, upper bound: 2204.5325534
NS_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.4924523, upper bound: 2204.5319422
NS_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5008131, upper bound: 2204.5325534
NS_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.4924523, upper bound: 2204.5319422
NS_B2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
NS_B2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
NS_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
NS_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5707803, upper bound: 2204.5707802
NS_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
NS_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
NS_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
NS_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5202463, upper bound: 2204.5411363
NS_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5445760, upper bound: 2204.5209686
NS_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5445619, upper bound: 2204.5213602
NS_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5445760, upper bound: 2204.5209686
NS_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5445619, upper bound: 2204.5213602
NS_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5326655, upper bound: 2204.5153418
NS_B2_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5404259, upper bound: 2204.5153418
NS_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5419296, upper bound: 2204.5192530
NS_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5419468, upper bound: 2204.5212833
NS_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.4985453, upper bound: 2204.5168143
NS_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5189289, upper bound: 2204.5168143
NS_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5249306, upper bound: 2204.5226002
NS_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5245245, upper bound: 2204.5230168
NS_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5161138, upper bound: 2204.5168252
NS_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5168252
NS_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5161138, upper bound: 2204.5168251
NS_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5168252

## BFS NS instance: NS_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -739.8888550, 1187.2478027, -737.3473511, 1181.9729004, -1921.8616943, 1924.5952148
1: -831.8164062, 1213.2066650, -828.6954346, 1207.8969727, -2039.7131348, 2041.9019775
2: -838.6459961, 1212.8765869, -835.8386841, 1207.7348633, -2046.3807373, 2048.7153320
3: -1018.1198730, 1399.0572510, -1013.8854980, 1393.1887207, -2411.3085938, 2412.9428711
4: -920.3421631, 1396.0484619, -917.7453613, 1389.7955322, -2310.1376953, 2313.7939453

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5267824
time: 1.28 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5357645
time: 1.33 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -746.6878662, 1198.1086426, -744.5747681, 1193.5312500, -1940.2191162, 1942.6833496
1: -839.3478394, 1224.6741943, -836.8280640, 1219.9150391, -2059.2629395, 2061.5021973
2: -846.5509644, 1224.1058350, -844.1539917, 1219.6604004, -2066.2111816, 2068.2597656
3: -1027.0643311, 1412.3280029, -1023.6383057, 1407.1032715, -2434.1672363, 2435.9663086
4: -930.2369385, 1408.7332764, -927.6253662, 1403.3869629, -2333.6232910, 2336.3583984

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5267824
time: 1.01 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5357645
time: 1.18 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -739.8997803, 1187.4226074, -935.1323853, 1499.9420166, -2239.8417969, 2122.5546875
1: -831.8004761, 1213.7646484, -1050.6319580, 1533.0158691, -2364.8159180, 2264.3964844
2: -838.8721924, 1213.2177734, -1059.3850098, 1531.6567383, -2370.5288086, 2272.6025391
3: -1017.9088135, 1399.7661133, -1284.5554199, 1768.0664062, -2785.9750977, 2684.3215332
4: -921.8178711, 1396.1634521, -1160.7006836, 1762.5334473, -2684.3513184, 2556.8640137

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4946178, upper bound: 2204.5231007
time: 1.32 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
time: 1.40 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -823.7167969, 1326.8155518, -935.3297119, 1500.0623779, -2323.7792969, 2262.1452637
1: -926.5855713, 1356.8237305, -1050.8370361, 1533.0853271, -2459.6708984, 2407.6606445
2: -934.2412720, 1354.4296875, -1059.5760498, 1531.7298584, -2465.9711914, 2414.0056152
3: -1134.2304688, 1565.1447754, -1284.8006592, 1768.2214355, -2902.4519043, 2849.9450684
4: -1028.6525879, 1557.8830566, -1160.7661133, 1762.6361084, -2791.2883301, 2718.6486816

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5028606, upper bound: 2204.5323381
time: 1.00 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4952319, upper bound: 2204.5318728
time: 1.23 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -940.2817993, 1509.0428467, -737.3473511, 1181.9729004, -2122.2546387, 2246.3898926
1: -1056.6192627, 1542.2465820, -828.6954346, 1207.8969727, -2264.5161133, 2370.9418945
2: -1065.0073242, 1540.7747803, -835.8386841, 1207.7348633, -2272.7416992, 2376.6135254
3: -1292.4550781, 1778.4005127, -1013.8854980, 1393.1887207, -2685.6433105, 2792.2854004
4: -1165.8300781, 1773.4591064, -917.7453613, 1389.7955322, -2555.6252441, 2691.2045898

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A2_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5267470
time: 1.06 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5357626
time: 1.15 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -945.0764160, 1516.7320557, -744.5747681, 1193.5312500, -2138.6076660, 2261.3063965
1: -1061.9445801, 1550.2939453, -836.8280640, 1219.9150391, -2281.8596191, 2387.1218262
2: -1070.6265869, 1548.6778564, -844.1539917, 1219.6604004, -2290.2868652, 2392.8317871
3: -1298.7576904, 1787.7120361, -1023.6383057, 1407.1032715, -2705.8603516, 2811.3503418
4: -1173.0065918, 1782.3546143, -927.6253662, 1403.3869629, -2576.3935547, 2709.9799805

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5267470
time: 0.98 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5357626
time: 1.18 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -940.2817993, 1509.0428467, -936.9418335, 1502.7854004, -2443.0671387, 2445.9846191
1: -1056.6192627, 1542.2465820, -1052.6549072, 1535.7484131, -2592.3676758, 2594.9013672
2: -1065.0073242, 1540.7747803, -1061.2882080, 1534.4967041, -2599.5034180, 2602.0629883
3: -1292.4550781, 1778.4005127, -1287.1998291, 1771.2313232, -3063.6855469, 3065.5996094
4: -1165.8300781, 1773.4591064, -1161.9748535, 1765.9656982, -2931.7958984, 2935.4338379

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5227887
time: 1.32 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5319422
time: 1.33 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -945.0764160, 1516.7320557, -942.8804321, 1512.3172607, -2457.3933105, 2459.6120605
1: -1061.9445801, 1550.2939453, -1059.3159180, 1545.6064453, -2607.5507812, 2609.6098633
2: -1070.6265869, 1548.6778564, -1068.1302490, 1544.2927246, -2614.9191895, 2616.8076172
3: -1298.7576904, 1787.7120361, -1295.2308350, 1782.5905762, -3081.3481445, 3082.9426270
4: -1173.0065918, 1782.3546143, -1170.1734619, 1777.1026611, -2950.1088867, 2952.5280762

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5227887
time: 1.08 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5319422
time: 1.32 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -749.1112671, 1200.8592529, -749.2294312, 1201.0413818, -1950.1525879, 1950.0886230
1: -841.9461060, 1227.4023438, -842.0776367, 1227.5903320, -2069.5361328, 2069.4797363
2: -849.3118896, 1227.1525879, -849.4456177, 1227.3383789, -2076.6503906, 2076.5979004
3: -1029.8669434, 1415.7583008, -1030.0261230, 1415.9755859, -2445.8425293, 2445.7844238
4: -933.3326416, 1411.9925537, -933.4797974, 1412.2065430, -2345.5390625, 2345.4721680

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5543248, upper bound: 2204.5607205
time: 1.03 seconds

## Relational analysis of NS_B2_B1_A2_A1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5723263, upper bound: 2204.5688122
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -749.1112671, 1200.8592529, -947.6765747, 1520.1228027, -2269.2338867, 2148.5358887
1: -841.9461060, 1227.4023438, -1064.6992188, 1553.5836182, -2395.5297852, 2292.1015625
2: -849.3118896, 1227.1525879, -1073.5823975, 1552.2584229, -2401.5703125, 2300.7343750
3: -1029.8669434, 1415.7583008, -1301.8259277, 1791.8009033, -2821.6679688, 2717.5842285
4: -933.3326416, 1411.9925537, -1176.2172852, 1786.2565918, -2719.5893555, 2588.2094727

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5740317, upper bound: 2204.5705666
time: 1.19 seconds

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5739271, upper bound: 2204.5705788
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -947.5657349, 1519.9526367, -749.2294312, 1201.0413818, -2148.6071777, 2269.1818848
1: -1064.5758057, 1553.4077148, -842.0776367, 1227.5903320, -2292.1660156, 2395.4853516
2: -1073.4567871, 1552.0844727, -849.4456177, 1227.3383789, -2300.7951660, 2401.5300293
3: -1301.6760254, 1791.5980225, -1030.0261230, 1415.9755859, -2717.6516113, 2821.6237793
4: -1176.0791016, 1786.0567627, -933.4797974, 1412.2065430, -2588.2856445, 2719.5366211

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5605929
time: 1.15 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5688122, upper bound: 2204.5688122
time: 0.99 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -947.5657349, 1519.9526367, -947.6765747, 1520.1228027, -2467.6882324, 2467.6291504
1: -1064.5758057, 1553.4077148, -1064.6992188, 1553.5836182, -2618.1594238, 2618.1069336
2: -1073.4567871, 1552.0844727, -1073.5823975, 1552.2584229, -2625.7153320, 2625.6669922
3: -1301.6760254, 1791.5980225, -1301.8259277, 1791.8009033, -3093.4770508, 3093.4238281
4: -1176.0791016, 1786.0567627, -1176.2172852, 1786.2565918, -2962.3356934, 2962.2739258

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5605929
time: 1.15 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5688122, upper bound: 2204.5688122
time: 0.95 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -878.2485962, 1402.9638672, -749.2294312, 1201.0413818, -2079.2900391, 2152.1933594
1: -986.8137817, 1434.3555908, -842.0776367, 1227.5903320, -2214.4040527, 2276.4331055
2: -995.7517090, 1433.2080078, -849.4456177, 1227.3383789, -2223.0900879, 2282.6533203
3: -1203.6290283, 1655.7335205, -1030.0261230, 1415.9755859, -2619.6044922, 2685.7597656
4: -1097.4490967, 1649.8654785, -933.4797974, 1412.2065430, -2509.6555176, 2583.3452148

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5151914, upper bound: 2204.5423905
time: 1.68 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5423905
time: 4.89 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1075.3442383, 1720.9935303, -749.2294312, 1201.0413818, -2276.3857422, 2470.2226562
1: -1207.7465820, 1759.0201416, -842.0776367, 1227.5903320, -2435.3369141, 2601.0976562
2: -1218.2094727, 1757.2082520, -849.4456177, 1227.3383789, -2445.5478516, 2606.6538086
3: -1473.7388916, 2029.5604248, -1030.0261230, 1415.9755859, -2889.7143555, 3059.5861816
4: -1337.8846436, 2022.2154541, -933.4797974, 1412.2065430, -2750.0905762, 2955.6950684

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5192376, upper bound: 2204.5435654
time: 1.05 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5196372, upper bound: 2204.5434948
time: 1.19 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -878.2485962, 1402.9638672, -947.6765747, 1520.1228027, -2398.3708496, 2350.6403809
1: -986.8137817, 1434.3555908, -1064.6992188, 1553.5836182, -2540.3974609, 2499.0546875
2: -995.7517090, 1433.2080078, -1073.5823975, 1552.2584229, -2548.0102539, 2506.7905273
3: -1203.6290283, 1655.7335205, -1301.8259277, 1791.8009033, -2995.4299316, 2957.5595703
4: -1097.4490967, 1649.8654785, -1176.2172852, 1786.2565918, -2883.0661621, 2826.0825195

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5190023, upper bound: 2204.5400206
time: 1.19 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5195256, upper bound: 2204.5379998
time: 1.35 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1075.3442383, 1720.9935303, -947.6765747, 1520.1228027, -2595.4670410, 2668.6699219
1: -1207.7465820, 1759.0201416, -1064.6992188, 1553.5836182, -2761.3300781, 2823.7192383
2: -1218.2094727, 1757.2082520, -1073.5823975, 1552.2584229, -2770.4675293, 2830.7905273
3: -1473.7388916, 2029.5604248, -1301.8259277, 1791.8009033, -3265.5397949, 3331.3862305
4: -1337.8846436, 2022.2154541, -1176.2172852, 1786.2565918, -3122.1997070, 3198.4321289

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B2_A2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5151914, upper bound: 2204.5394201
time: 1.59 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_A2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5394201
time: 1.73 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -738.0010376, 1183.0332031, -870.9686890, 1391.1269531, -2129.1279297, 2054.0017090
1: -829.5087891, 1209.2052002, -978.6676636, 1422.3179932, -2251.8266602, 2187.8725586
2: -836.7335815, 1208.9919434, -987.5139771, 1421.1885986, -2257.9221191, 2196.5058594
3: -1014.6687622, 1394.7266846, -1193.6259766, 1641.8015137, -2656.4702148, 2588.3525391
4: -919.5277100, 1391.1138916, -1088.4412842, 1636.0660400, -2555.5937500, 2479.5551758

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5243128
time: 1.21 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5243128
time: 1.24 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -821.7074585, 1322.4025879, -859.6686401, 1372.6882324, -2194.3957520, 2182.0710449
1: -924.1153564, 1352.1656494, -965.9656982, 1403.5502930, -2327.6655273, 2318.1313477
2: -931.9538574, 1350.0850830, -974.7888794, 1402.3504639, -2334.3041992, 2324.8735352
3: -1130.8178711, 1560.0310059, -1177.6999512, 1620.1695557, -2750.9873047, 2737.7309570
4: -1026.1292725, 1552.7178955, -1074.6512451, 1614.2838135, -2640.4130859, 2627.3691406

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5245245
time: 1.04 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5245245
time: 1.22 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -738.0010376, 1183.0332031, -1065.1414795, 1704.4692383, -2442.4702148, 2248.1748047
1: -829.5087891, 1209.2052002, -1196.3333740, 1742.2194824, -2571.7282715, 2405.5385742
2: -836.7335815, 1208.9919434, -1206.7033691, 1740.3916016, -2577.1250000, 2415.6953125
3: -1014.6687622, 1394.7266846, -1459.5972900, 2010.1710205, -3024.8398438, 2854.3239746
4: -919.5277100, 1391.1138916, -1325.4632568, 2002.8966064, -2922.4243164, 2716.5771484

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5349306, upper bound: 2204.5115155
time: 0.96 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
time: 1.15 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -821.7074585, 1322.4025879, -1065.6898193, 1705.2412109, -2526.9487305, 2388.0920410
1: -924.1153564, 1352.1656494, -1196.9073486, 1742.9638672, -2667.0788574, 2549.0729980
2: -931.9538574, 1350.0850830, -1207.2749023, 1741.1071777, -2673.0607910, 2557.3598633
3: -1130.8178711, 1560.0310059, -1460.3450928, 2011.0606689, -3141.8784180, 3020.3759766
4: -1026.1292725, 1552.7178955, -1325.9035645, 2003.6643066, -3029.7934570, 2878.3684082

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5349306, upper bound: 2204.5119271
time: 1.04 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_A2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
time: 1.27 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -941.1321411, 1509.4635010, -865.9602661, 1383.4871826, -2324.6184082, 2375.4235840
1: -1057.3305664, 1542.5191650, -972.9472656, 1414.0999756, -2471.4306641, 2515.4663086
2: -1065.9932861, 1541.3685303, -981.6586914, 1413.2529297, -2479.2460938, 2523.0273438
3: -1292.9707031, 1778.9918213, -1186.9516602, 1632.1510010, -2925.1213379, 2965.9433594
4: -1166.7474365, 1773.9707031, -1081.0112305, 1627.0207520, -2793.7680664, 2853.5769043

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5326655, upper bound: 2204.5189246
time: 1.20 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5326655, upper bound: 2204.5189289
time: 0.93 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -942.5073242, 1511.7509766, -876.8381348, 1400.5789795, -2343.0861816, 2388.5888672
1: -1058.8673096, 1544.9920654, -985.2229614, 1431.9395752, -2490.8068848, 2530.2145996
2: -1067.7215576, 1543.7102051, -994.1583862, 1430.7684326, -2498.4899902, 2537.8686523
3: -1294.6511230, 1781.8165283, -1201.6396484, 1652.9440918, -2947.5952148, 2983.4558105
4: -1169.7451172, 1776.4477539, -1095.7207031, 1647.0858154, -2816.8298340, 2871.5646973

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5404381, upper bound: 2204.5189246
time: 1.37 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5404381, upper bound: 2204.5189289
time: 1.28 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -940.0057373, 1507.7000732, -1071.2994385, 1714.3829346, -2654.3879395, 2578.9992676
1: -1056.0407715, 1540.8543701, -1203.2015381, 1752.3071289, -2808.3479004, 2744.0559082
2: -1064.8690186, 1539.5784912, -1213.6319580, 1750.4733887, -2815.3420410, 2753.2104492
3: -1291.1767578, 1777.1774902, -1468.0921631, 2021.8571777, -3313.0339355, 3245.2695312
4: -1166.7269287, 1771.5815430, -1332.9807129, 2014.4129639, -3181.1398926, 3102.5700684

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5115064
time: 1.25 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5192530
time: 1.28 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -956.4277954, 1533.5697021, -1069.4327393, 1711.3342285, -2667.7619629, 2603.0021973
1: -1074.2648926, 1566.9367676, -1201.0607910, 1749.1546631, -2823.4194336, 2767.9975586
2: -1083.3348389, 1566.1177979, -1211.5008545, 1747.3602295, -2830.6950684, 2777.6186523
3: -1313.1861572, 1807.4716797, -1465.4249268, 2018.2667236, -3331.4521484, 3272.8964844
4: -1187.0090332, 1802.0288086, -1330.6202393, 2010.7962646, -3197.8044434, 3130.6699219

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5115064
time: 0.86 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A2_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5212833
time: 1.26 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -863.0832520, 1378.5733643, -865.9602661, 1383.4871826, -2246.5703125, 2244.5336914
1: -969.6864624, 1408.8481445, -972.9472656, 1414.0999756, -2383.7863770, 2381.7954102
2: -978.2928467, 1408.2373047, -981.6586914, 1413.2529297, -2391.5454102, 2389.8959961
3: -1183.0726318, 1625.9886475, -1186.9516602, 1632.1510010, -2815.2236328, 2812.9404297
4: -1076.6649170, 1621.3752441, -1081.0112305, 1627.0207520, -2703.6855469, 2702.3864746

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5190891, upper bound: 2204.5199750
time: 1.28 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5190891, upper bound: 2204.5210252
time: 1.23 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -874.5387573, 1396.7172852, -876.8381348, 1400.5789795, -2275.1174316, 2273.5554199
1: -982.6329346, 1428.0239258, -985.2229614, 1431.9395752, -2414.5725098, 2413.2468262
2: -991.5609131, 1426.8238525, -994.1583862, 1430.7684326, -2422.3293457, 2420.9821777
3: -1198.4099121, 1648.4262695, -1201.6396484, 1652.9440918, -2851.3537598, 2850.0659180
4: -1092.9044189, 1642.5850830, -1095.7207031, 1647.0858154, -2739.9897461, 2738.3051758

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5192572, upper bound: 2204.5199750
time: 1.08 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4978787, upper bound: 2204.5210252
time: 1.12 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -866.8386230, 1384.4084473, -1065.1414795, 1704.4692383, -2571.3078613, 2449.5498047
1: -974.0506592, 1415.4921875, -1196.3333740, 1742.2194824, -2716.2700195, 2611.8254395
2: -982.8425903, 1414.3786621, -1206.7033691, 1740.3916016, -2723.2341309, 2621.0820312
3: -1187.9548340, 1633.8980713, -1459.5972900, 2010.1710205, -3198.1259766, 3093.4951172
4: -1083.3437500, 1628.2454834, -1325.4632568, 2002.8966064, -3085.3715820, 2953.7087402

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.4940937, upper bound: 2204.4885600
time: 1.06 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5154044, upper bound: 2204.5129780
time: 1.23 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -949.7016602, 1522.8638916, -1065.6898193, 1705.2412109, -2654.9428711, 2588.5537109
1: -1067.6151123, 1557.1064453, -1196.9073486, 1742.9638672, -2810.5783691, 2754.0136719
2: -1076.8326416, 1554.7235107, -1207.2749023, 1741.1071777, -2817.9399414, 2761.9982910
3: -1303.0878906, 1797.1711426, -1460.3450928, 2011.0606689, -3314.1479492, 3257.5156250
4: -1188.0065918, 1788.7879639, -1325.9035645, 2003.6643066, -3189.2695312, 3114.2990723

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5189810, upper bound: 2204.5168101
time: 0.97 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5185374, upper bound: 2204.5168101
time: 1.07 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1061.6512451, 1699.0894775, -865.9602661, 1383.4871826, -2445.1384277, 2565.0490723
1: -1192.2880859, 1736.2871094, -972.9472656, 1414.0999756, -2606.3881836, 2709.2343750
2: -1202.5068359, 1734.7565918, -981.6586914, 1413.2529297, -2615.7587891, 2716.4152832
3: -1455.1474609, 2003.2508545, -1186.9516602, 1632.1510010, -3087.2980957, 3190.2026367
4: -1319.3314209, 1996.5783691, -1081.0112305, 1627.0207520, -2946.3520508, 3076.2158203

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5198005
time: 1.23 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5209445
time: 1.69 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1071.1383057, 1714.0062256, -876.8381348, 1400.5789795, -2471.7172852, 2590.8442383
1: -1203.0167236, 1751.9155273, -985.2229614, 1431.9395752, -2634.9562988, 2737.1384277
2: -1213.4631348, 1750.0543213, -994.1583862, 1430.7684326, -2644.2314453, 2744.2126465
3: -1467.8559570, 2021.3729248, -1201.6396484, 1652.9440918, -3120.7998047, 3223.0126953
4: -1332.7282715, 2014.0590820, -1095.7207031, 1647.0858154, -2979.8137207, 3109.0576172

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5198005
time: 1.47 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5209445
time: 1.09 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1061.6512451, 1699.0894775, -1063.9853516, 1703.0384521, -2764.2822266, 2762.6748047
1: -1192.2880859, 1736.2871094, -1194.9295654, 1740.4188232, -2932.7070312, 2931.2167969
2: -1202.5068359, 1734.7565918, -1205.2092285, 1738.8068848, -2941.0468750, 2939.6967773
3: -1455.1474609, 2003.2508545, -1458.3117676, 2007.9833984, -3463.1308594, 3461.5625000
4: -1319.3314209, 1996.5783691, -1322.7722168, 2001.1303711, -3318.0334473, 3316.6950684

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5159439
time: 1.18 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5168252
time: 1.16 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1071.1383057, 1714.0062256, -1073.7482910, 1718.3304443, -2789.4687500, 2787.7543945
1: -1203.0167236, 1751.9155273, -1205.9517822, 1756.3132324, -2959.3298340, 2957.8671875
2: -1213.4631348, 1750.0543213, -1216.4079590, 1754.4797363, -2967.9428711, 2966.4621582
3: -1467.8559570, 2021.3729248, -1471.5023193, 2026.4426270, -3494.2983398, 3492.8752441
4: -1332.7282715, 2014.0590820, -1335.9257812, 2019.1052246, -3349.7548828, 3347.9624023

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5159439
time: 1.13 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5168252
time: 1.17 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.06 seconds
NS_B2_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5267824
NS_B2_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5357645
NS_B2_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5267824
NS_B2_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4961071, upper bound: 2204.5357645
NS_B2_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4946178, upper bound: 2204.5231007
NS_B2_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
NS_B2_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5028606, upper bound: 2204.5323381
NS_B2_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4952319, upper bound: 2204.5318728
NS_B2_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5267470
NS_B2_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5357626
NS_B2_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5267470
NS_B2_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919978, upper bound: 2204.5357626
NS_B2_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5227887
NS_B2_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5319422
NS_B2_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5227887
NS_B2_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4919663, upper bound: 2204.5319422
NS_B2_B1_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5543248, upper bound: 2204.5607205
NS_B2_B1_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5723263, upper bound: 2204.5688122
NS_B2_B1_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5740317, upper bound: 2204.5705666
NS_B2_B1_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5739271, upper bound: 2204.5705788
NS_B2_B1_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5605929
NS_B2_B1_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5688122, upper bound: 2204.5688122
NS_B2_B1_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5605929
NS_B2_B1_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5688122, upper bound: 2204.5688122
NS_B2_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5151914, upper bound: 2204.5423905
NS_B2_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5423905
NS_B2_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5192376, upper bound: 2204.5435654
NS_B2_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5196372, upper bound: 2204.5434948
NS_B2_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5190023, upper bound: 2204.5400206
NS_B2_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5195256, upper bound: 2204.5379998
NS_B2_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5151914, upper bound: 2204.5394201
NS_B2_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5394201
NS_B2_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5243128
NS_B2_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5243128
NS_B2_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5245245
NS_B2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5442856, upper bound: 2204.5245245
NS_B2_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5349306, upper bound: 2204.5115155
NS_B2_B2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
NS_B2_B2_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5349306, upper bound: 2204.5119271
NS_B2_B2_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
NS_B2_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5326655, upper bound: 2204.5189246
NS_B2_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5326655, upper bound: 2204.5189289
NS_B2_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5404381, upper bound: 2204.5189246
NS_B2_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5404381, upper bound: 2204.5189289
NS_B2_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5115064
NS_B2_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5192530
NS_B2_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5115064
NS_B2_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5242751, upper bound: 2204.5212833
NS_B2_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5190891, upper bound: 2204.5199750
NS_B2_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5190891, upper bound: 2204.5210252
NS_B2_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5192572, upper bound: 2204.5199750
NS_B2_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4978787, upper bound: 2204.5210252
NS_B2_B2_A2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.4940937, upper bound: 2204.4885600
NS_B2_B2_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5154044, upper bound: 2204.5129780
NS_B2_B2_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5189810, upper bound: 2204.5168101
NS_B2_B2_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5185374, upper bound: 2204.5168101
NS_B2_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5198005
NS_B2_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5209445
NS_B2_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5198005
NS_B2_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5209445
NS_B2_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5159439
NS_B2_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5151595, upper bound: 2204.5168252
NS_B2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5159439
NS_B2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 3, lower bound: -2204.5153222, upper bound: 2204.5168252

## BFS NS instance: NS_B2_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -739.8888550, 1187.2478027, -737.6021118, 1182.1845703, -1922.0734863, 1924.8498535
1: -831.8164062, 1213.2066650, -828.9519043, 1207.9633789, -2039.7795410, 2042.1585693
2: -838.6459961, 1212.8765869, -836.0655518, 1207.9832764, -2046.6291504, 2048.9418945
3: -1018.1198730, 1399.0572510, -1014.2639160, 1393.2148438, -2411.3347168, 2413.3212891
4: -920.3421631, 1396.0484619, -917.5365601, 1390.1973877, -2310.5395508, 2313.5849609

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5036381, upper bound: 2204.5349018
time: 0.91 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5028606, upper bound: 2204.5351728
time: 1.26 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -739.8888550, 1187.2478027, -741.7020874, 1188.8485107, -1928.7373047, 1928.9498291
1: -831.8164062, 1213.2066650, -833.5987549, 1215.1098633, -2046.9260254, 2046.8054199
2: -838.6459961, 1212.8765869, -840.8916626, 1214.8664551, -2053.5122070, 2053.7680664
3: -1018.1198730, 1399.0572510, -1019.6406250, 1401.5244141, -2419.6442871, 2418.6977539
4: -920.3421631, 1396.0484619, -924.0239868, 1397.8974609, -2318.2397461, 2320.0722656

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5036381, upper bound: 2204.5353876
time: 2.67 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5028606, upper bound: 2204.5356826
time: 0.90 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -746.6878662, 1198.1086426, -737.6021118, 1182.1845703, -1928.8724365, 1935.7106934
1: -839.3478394, 1224.6741943, -828.9519043, 1207.9633789, -2047.3111572, 2053.6259766
2: -846.5509644, 1224.1058350, -836.0655518, 1207.9832764, -2054.5341797, 2060.1713867
3: -1027.0643311, 1412.3280029, -1014.2639160, 1393.2148438, -2420.2792969, 2426.5915527
4: -930.2369385, 1408.7332764, -917.5365601, 1390.1973877, -2320.4343262, 2326.2695312

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4958067, upper bound: 2204.5255730
time: 1.10 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4949185, upper bound: 2204.5260227
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -746.6878662, 1198.1086426, -741.7020874, 1188.8485107, -1935.5363770, 1939.8106689
1: -839.3478394, 1224.6741943, -833.5987549, 1215.1098633, -2054.4577637, 2058.2727051
2: -846.5509644, 1224.1058350, -840.8916626, 1214.8664551, -2061.4174805, 2064.9975586
3: -1027.0643311, 1412.3280029, -1019.6406250, 1401.5244141, -2428.5886230, 2431.9687500
4: -930.2369385, 1408.7332764, -924.0239868, 1397.8974609, -2328.1342773, 2332.7568359

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4958067, upper bound: 2204.5255730
time: 1.03 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4949185, upper bound: 2204.5260227
time: 1.16 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -726.9191284, 1167.3677979, -926.2474976, 1486.1414795, -2213.0605469, 2093.6149902
1: -817.3423462, 1193.5314941, -1040.7229004, 1519.0811768, -2336.4235840, 2234.2543945
2: -824.2847290, 1192.6926270, -1049.3581543, 1517.4848633, -2341.7695312, 2242.0507812
3: -1000.1508789, 1376.4033203, -1272.4035645, 1752.0227051, -2752.1735840, 2648.8068848
4: -906.2462158, 1372.3460693, -1149.9782715, 1746.0886230, -2652.3342285, 2522.3239746

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
time: 1.11 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
time: 1.61 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -759.7911377, 1218.4149170, -929.2493896, 1490.4885254, -2250.2792969, 2147.6633301
1: -853.8340454, 1245.1942139, -1044.0629883, 1523.3868408, -2377.2204590, 2289.2573242
2: -861.1880493, 1244.5672607, -1052.7357178, 1522.0052490, -2383.1931152, 2297.3029785
3: -1044.8392334, 1436.2651367, -1276.5015869, 1756.9451904, -2801.7844238, 2712.7666016
4: -945.3560181, 1432.5877686, -1153.5109863, 1751.4467773, -2696.8027344, 2586.0983887

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
time: 1.23 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
time: 1.33 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -814.1363525, 1311.7781982, -927.2593994, 1487.0930176, -2301.2292480, 2239.0375977
1: -915.8702393, 1341.2696533, -1041.7695312, 1519.6999512, -2435.5703125, 2383.0390625
2: -923.1989136, 1338.9216309, -1050.3190918, 1518.4118652, -2441.6105957, 2389.2407227
3: -1121.2977295, 1547.1569824, -1273.8074951, 1752.7775879, -2874.0751953, 2820.9643555
4: -1015.7491455, 1540.2565918, -1149.9039307, 1747.4649658, -2763.2141113, 2690.1599121

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4948096, upper bound: 2204.5227766
time: 1.10 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4948638, upper bound: 2204.5318728
time: 1.10 seconds

## BFS NS instance: NS_B2_B1_A1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -818.7861938, 1318.6790771, -933.2036743, 1496.6267090, -2315.4128418, 2251.8828125
1: -921.0164185, 1348.4278564, -1048.4377441, 1529.5529785, -2450.5693359, 2396.8657227
2: -928.6351318, 1346.1127930, -1057.1666260, 1528.2102051, -2456.8452148, 2403.2792969
3: -1127.3626709, 1555.4355469, -1281.8480225, 1764.1263428, -2891.4890137, 2837.2834473
4: -1022.4277344, 1548.3857422, -1158.1077881, 1758.6087646, -2781.0366211, 2706.4934082

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4948638, upper bound: 2204.5227766
time: 0.89 seconds

## Relational analysis of NS_B2_B1_A1_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_B1_A1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4948638, upper bound: 2204.5318728
time: 1.00 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -940.2817993, 1509.0428467, -737.6021118, 1182.1845703, -2122.4663086, 2246.6450195
1: -1056.6192627, 1542.2465820, -828.9519043, 1207.9633789, -2264.5825195, 2371.1984863
2: -1065.0073242, 1540.7747803, -836.0655518, 1207.9832764, -2272.9907227, 2376.8403320
3: -1292.4550781, 1778.4005127, -1014.2639160, 1393.2148438, -2685.6699219, 2792.6638184
4: -1165.8300781, 1773.4591064, -917.5365601, 1390.1973877, -2556.0273438, 2690.9956055

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_B1_A1_B1_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4785301, upper bound: 2204.5157362
time: 1.42 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A1_B1_B2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4781161, upper bound: 2204.5149497
time: 1.19 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -940.2817993, 1509.0428467, -741.7020874, 1188.8485107, -2129.1303711, 2250.7448730
1: -1056.6192627, 1542.2465820, -833.5987549, 1215.1098633, -2271.7290039, 2375.8452148
2: -1065.0073242, 1540.7747803, -840.8916626, 1214.8664551, -2279.8735352, 2381.6665039
3: -1292.4550781, 1778.4005127, -1019.6406250, 1401.5244141, -2693.9790039, 2798.0410156
4: -1165.8300781, 1773.4591064, -924.0239868, 1397.8974609, -2563.7275391, 2697.4831543

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_B1_A1_B2_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4785301, upper bound: 2204.5157362
time: 1.28 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A1_B2_B2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4781161, upper bound: 2204.5249152
time: 1.04 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -945.0764160, 1516.7320557, -737.6021118, 1182.1845703, -2127.2609863, 2254.3342285
1: -1061.9445801, 1550.2939453, -828.9519043, 1207.9633789, -2269.9079590, 2379.2458496
2: -1070.6265869, 1548.6778564, -836.0655518, 1207.9832764, -2278.6098633, 2384.7431641
3: -1298.7576904, 1787.7120361, -1014.2639160, 1393.2148438, -2691.9726562, 2801.9758301
4: -1173.0065918, 1782.3546143, -917.5365601, 1390.1973877, -2563.2041016, 2699.8911133

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_B1_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4786788, upper bound: 2204.5157362
time: 0.86 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A2_B1_B2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4782407, upper bound: 2204.5149497
time: 0.94 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -945.0764160, 1516.7320557, -741.7020874, 1188.8485107, -2133.9248047, 2258.4340820
1: -1061.9445801, 1550.2939453, -833.5987549, 1215.1098633, -2277.0544434, 2383.8925781
2: -1070.6265869, 1548.6778564, -840.8916626, 1214.8664551, -2285.4931641, 2389.5693359
3: -1298.7576904, 1787.7120361, -1019.6406250, 1401.5244141, -2700.2817383, 2807.3525391
4: -1173.0065918, 1782.3546143, -924.0239868, 1397.8974609, -2570.9040527, 2706.3784180

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_B1_A2_B2_B1

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4786788, upper bound: 2204.5177033
time: 1.47 seconds

## Relational analysis of NS_B2_B1_A1_A2_B1_A2_B2_B2

### Relational analysis result of NS_B2_B1_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4782407, upper bound: 2204.5169918
time: 0.85 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -940.2817993, 1509.0428467, -937.9713745, 1504.3095703, -2444.5913086, 2447.0141602
1: -1056.6192627, 1542.2465820, -1053.8027344, 1537.2694092, -2593.8886719, 2596.0493164
2: -1065.0073242, 1540.7747803, -1062.3947754, 1536.1148682, -2601.1220703, 2603.1694336
3: -1292.4550781, 1778.4005127, -1288.6613770, 1772.9471436, -3065.4018555, 3067.0617676
4: -1165.8300781, 1773.4591064, -1162.7652588, 1767.9299316, -2933.7600098, 2936.2238770

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4933583, upper bound: 2204.5220440
time: 1.05 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4894998, upper bound: 2204.5199754
time: 1.06 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -940.2817993, 1509.0428467, -939.6588135, 1507.1131592, -2447.3950195, 2448.7016602
1: -1056.6192627, 1542.2465820, -1055.6811523, 1540.2624512, -2596.8818359, 2597.9277344
2: -1065.0073242, 1540.7747803, -1064.4797363, 1538.9781494, -2603.9851074, 2605.2543945
3: -1292.4550781, 1778.4005127, -1290.7579346, 1776.3839111, -3068.8383789, 3069.1584473
4: -1165.8300781, 1773.4591064, -1166.1451416, 1771.0000000, -2936.8300781, 2939.6042480

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4933583, upper bound: 2204.5242300
time: 0.94 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4894998, upper bound: 2204.5226230
time: 0.94 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -945.0764160, 1516.7320557, -937.9713745, 1504.3095703, -2449.3859863, 2454.7031250
1: -1061.9445801, 1550.2939453, -1053.8027344, 1537.2694092, -2599.2138672, 2604.0966797
2: -1070.6265869, 1548.6778564, -1062.3947754, 1536.1148682, -2606.7414551, 2611.0720215
3: -1298.7576904, 1787.7120361, -1288.6613770, 1772.9471436, -3071.7045898, 3076.3735352
4: -1173.0065918, 1782.3546143, -1162.7652588, 1767.9299316, -2940.9365234, 2945.1191406

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B1_B1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4795312, upper bound: 2204.5122855
time: 1.00 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B1_B2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4909391, upper bound: 2204.5224853
time: 1.12 seconds

## BFS NS instance: NS_B2_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -945.0764160, 1516.7320557, -939.6588135, 1507.1131592, -2452.1894531, 2456.3906250
1: -1061.9445801, 1550.2939453, -1055.6811523, 1540.2624512, -2602.2070312, 2605.9750977
2: -1070.6265869, 1548.6778564, -1064.4797363, 1538.9781494, -2609.6044922, 2613.1574707
3: -1298.7576904, 1787.7120361, -1290.7579346, 1776.3839111, -3075.1413574, 3078.4699707
4: -1173.0065918, 1782.3546143, -1166.1451416, 1771.0000000, -2944.0065918, 2948.4995117

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4826804, upper bound: 2204.5149393
time: 1.06 seconds

## Relational analysis of NS_B2_B1_A1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_B1_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4795267, upper bound: 2204.5133209
time: 1.13 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -740.5338745, 1186.9154053, -739.9984131, 1186.2333984, -1926.7673340, 1926.9135742
1: -832.2148438, 1212.7844238, -831.6488037, 1212.2381592, -2044.4530029, 2044.4331055
2: -839.4078979, 1212.8225098, -838.8596802, 1212.0867920, -2051.4943848, 2051.6821289
3: -1018.2453613, 1398.7630615, -1017.4846191, 1398.1948242, -2416.4401855, 2416.2475586
4: -921.2334595, 1395.7592773, -921.1033936, 1394.8110352, -2316.0441895, 2316.8627930

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5546682, upper bound: 2204.5546682
time: 1.14 seconds

## Relational analysis of NS_B2_B1_A2_A1_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5546682, upper bound: 2204.5643744
time: 1.21 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -744.3553467, 1193.1093750, -747.3485107, 1197.9761963, -1942.3315430, 1940.4577637
1: -836.6074219, 1219.4516602, -839.9632568, 1224.4453125, -2061.0522461, 2059.4147949
2: -843.9105835, 1219.2302246, -847.3095703, 1224.2012939, -2068.1118164, 2066.5390625
3: -1023.2776489, 1406.5345459, -1027.4079590, 1412.3265381, -2435.6042480, 2433.9423828
4: -927.3684692, 1402.9313965, -931.1221924, 1408.6134033, -2335.9816895, 2334.0529785

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5643744, upper bound: 2204.5546682
time: 1.12 seconds

## Relational analysis of NS_B2_B1_A2_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5643744, upper bound: 2204.5723292
time: 1.30 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -737.7979126, 1182.7158203, -937.7956543, 1504.2883301, -2242.0861816, 2120.5112305
1: -829.2812500, 1208.8784180, -1053.6088867, 1537.4361572, -2366.7172852, 2262.4870605
2: -836.5034790, 1208.6667480, -1062.4171143, 1536.0893555, -2372.5927734, 2271.0839844
3: -1014.3916626, 1394.3491211, -1288.1936035, 1773.1544189, -2787.5456543, 2682.5427246
4: -919.2750854, 1390.7391357, -1164.0690918, 1767.6353760, -2686.9104004, 2554.8081055

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A1_A1

### Relational analysis result of NS_B2_B1_A2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5578837, upper bound: 2204.5531879
time: 1.22 seconds

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A1_A2

### Relational analysis result of NS_B2_B1_A2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5574764, upper bound: 2204.5531239
time: 0.97 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -821.5031128, 1322.0836182, -938.0238037, 1504.4667969, -2325.9697266, 2260.1074219
1: -923.8870239, 1351.8380127, -1053.8471680, 1537.5626221, -2461.4497070, 2405.6848145
2: -931.7229004, 1349.7586670, -1062.6458740, 1536.2133789, -2467.9357910, 2412.4045410
3: -1130.5399170, 1559.6519775, -1288.4755859, 1773.3795166, -2903.9194336, 2848.1274414
4: -1025.8765869, 1552.3409424, -1164.1818848, 1767.8055420, -2793.6821289, 2716.5227051

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A2_A1

### Relational analysis result of NS_B2_B1_A2_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5578837, upper bound: 2204.5532098
time: 1.15 seconds

## Relational analysis of NS_B2_B1_A2_A1_A1_B2_A2_A2

### Relational analysis result of NS_B2_B1_A2_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5570450, upper bound: 2204.5528669
time: 1.08 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -940.9617920, 1509.1997070, -739.9984131, 1186.2333984, -2127.1950684, 2249.1979980
1: -1057.1395264, 1542.2476807, -831.6488037, 1212.2381592, -2269.3774414, 2373.8964844
2: -1065.8005371, 1541.0988770, -838.8596802, 1212.0867920, -2277.8872070, 2379.9584961
3: -1292.7384033, 1778.6776123, -1017.4846191, 1398.1948242, -2690.9331055, 2796.1621094
4: -1166.5354004, 1773.6585693, -921.1033936, 1394.8110352, -2561.3459473, 2694.7619629

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_A1_B1

### Relational analysis result of NS_B2_B1_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512493, upper bound: 2204.5543248
time: 1.45 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_A1_B2

### Relational analysis result of NS_B2_B1_A2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512493, upper bound: 2204.5636928
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -942.3135986, 1511.4501953, -747.3485107, 1197.9761963, -2140.2895508, 2258.7983398
1: -1058.6510010, 1544.6823730, -839.9632568, 1224.4453125, -2283.0961914, 2384.6452637
2: -1067.5026855, 1543.4020996, -847.3095703, 1224.2012939, -2291.7038574, 2390.7109375
3: -1294.3880615, 1781.4591064, -1027.4079590, 1412.3265381, -2706.7143555, 2808.8664551
4: -1169.5056152, 1776.0922852, -931.1221924, 1408.6134033, -2578.1186523, 2707.2143555

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_A2_B1

### Relational analysis result of NS_B2_B1_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5607205, upper bound: 2204.5543248
time: 1.14 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2_B1_A2_B2

### Relational analysis result of NS_B2_B1_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5607205, upper bound: 2204.5723263
time: 0.98 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -940.9617920, 1509.1997070, -939.5111694, 1507.0028076, -2447.9643555, 2448.7102051
1: -1057.1395264, 1542.2476807, -1055.5273438, 1540.0412598, -2597.1804199, 2597.7746582
2: -1065.8005371, 1541.0988770, -1064.2187500, 1538.7885742, -2604.5888672, 2605.3176270
3: -1292.7384033, 1778.6776123, -1290.7064209, 1776.1739502, -3068.9123535, 3069.3840332
4: -1166.5354004, 1773.6585693, -1165.2409668, 1770.9030762, -2937.4377441, 2938.8994141

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_B1_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5512193
time: 1.18 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_B1_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5605929
time: 1.09 seconds

## BFS NS instance: NS_B2_B1_A2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -942.3135986, 1511.4501953, -945.5847168, 1516.7375488, -2459.0512695, 2457.0341797
1: -1058.6510010, 1544.6823730, -1062.3392334, 1550.1030273, -2608.7539062, 2607.0212402
2: -1067.5026855, 1543.4020996, -1071.2106934, 1548.8017578, -2616.3039551, 2614.6125488
3: -1294.3880615, 1781.4591064, -1298.9228516, 1787.7633057, -3082.1511230, 3080.3818359
4: -1169.5056152, 1776.0922852, -1173.5976562, 1782.2905273, -2951.7956543, 2949.6892090

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A1_A2_B2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5605929, upper bound: 2204.5512193
time: 1.01 seconds

## Relational analysis of NS_B2_B1_A2_A1_A2_B2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5605929, upper bound: 2204.5688122
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -862.8977051, 1378.2769775, -739.9984131, 1186.2333984, -2049.1311035, 2118.2753906
1: -969.4775391, 1408.5432129, -831.6488037, 1212.2381592, -2181.7158203, 2240.1918945
2: -978.0824585, 1407.9313965, -838.8596802, 1212.0867920, -2190.1691895, 2246.7910156
3: -1182.8151855, 1625.6370850, -1017.4846191, 1398.1948242, -2581.0097656, 2643.1215820
4: -1076.4349365, 1621.0249023, -921.1033936, 1394.8110352, -2471.2458496, 2542.1279297

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B1_A1_A1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5177558, upper bound: 2204.5346044
time: 1.45 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_A1_A1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5177558, upper bound: 2204.5424775
time: 1.27 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -874.3494873, 1396.4132080, -747.3485107, 1197.9761963, -2072.3254395, 2143.7614746
1: -982.4202881, 1427.7114258, -839.9632568, 1224.4453125, -2206.8657227, 2267.6740723
2: -991.3464966, 1426.5113525, -847.3095703, 1224.2012939, -2215.5478516, 2273.8208008
3: -1198.1468506, 1648.0671387, -1027.4079590, 1412.3265381, -2610.4733887, 2675.4748535
4: -1092.6711426, 1642.2257080, -931.1221924, 1408.6134033, -2501.2844238, 2573.3476562

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B1_A1_A2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5128233, upper bound: 2204.5372688
time: 1.24 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_A1_A2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5119372, upper bound: 2204.5359160
time: 1.29 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1064.9875488, 1704.2282715, -737.9144287, 1182.8953857, -2247.8828125, 2442.1425781
1: -1196.1608887, 1741.9715576, -829.4112549, 1209.0637207, -2405.2246094, 2571.3823242
2: -1206.5290527, 1740.1461182, -836.6354370, 1208.8502197, -2415.3791504, 2576.7814941
3: -1459.3875732, 2009.8841553, -1014.5485840, 1394.5634766, -2853.9511719, 3024.4326172
4: -1325.2730713, 2002.6140137, -919.4204102, 1390.9499512, -2716.2231445, 2922.0341797

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5148791, upper bound: 2204.5422468
time: 1.08 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_B2_B1_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5135577, upper bound: 2204.5422468
time: 1.07 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1065.5346680, 1704.9985352, -821.6220703, 1322.2681885, -2387.8027344, 2526.6203613
1: -1196.7337646, 1742.7136230, -924.0200195, 1352.0280762, -2548.7617188, 2666.7336426
2: -1207.0992432, 1740.8594971, -931.8574829, 1349.9475098, -2557.0468750, 2672.7170410
3: -1460.1339111, 2010.7709961, -1130.7011719, 1559.8718262, -3020.0058594, 3141.4721680
4: -1325.7116699, 2003.3792725, -1026.0245361, 1552.5588379, -2877.9916992, 3029.4038086

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5151354, upper bound: 2204.5422373
time: 1.23 seconds

## Relational analysis of NS_B2_B1_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137732, upper bound: 2204.5422373
time: 1.23 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -877.0031738, 1400.9088135, -941.6265869, 1510.2904053, -2387.2934570, 2342.5354004
1: -985.4108276, 1432.2630615, -1057.8532715, 1543.5014648, -2528.9121094, 2490.1162109
2: -994.3398438, 1431.1055908, -1066.7080078, 1542.2037354, -2536.5434570, 2497.8134766
3: -1201.8797607, 1653.3492432, -1293.4429932, 1780.2526855, -2982.1323242, 2946.7912598
4: -1095.9294434, 1647.4345703, -1168.7252197, 1774.6134033, -2869.8459473, 2816.1591797

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_B1_A2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5178228, upper bound: 2204.5384593
time: 1.15 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_B1_A2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5165487, upper bound: 2204.5382341
time: 0.94 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -873.1979980, 1394.5684814, -963.6494751, 1542.8265381, -2416.0244141, 2358.2180176
1: -981.1211548, 1425.7984619, -1082.5576172, 1576.9102783, -2558.0310059, 2508.3559570
2: -990.0188599, 1424.6163330, -1091.6462402, 1575.5119629, -2565.5307617, 2516.2624512
3: -1196.6323242, 1645.8687744, -1322.7485352, 1818.7474365, -3015.3793945, 2968.6171875
4: -1091.1595459, 1639.9104004, -1196.5091553, 1813.3192139, -2903.9367676, 2836.4194336

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_B1_A2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5178358, upper bound: 2204.5335731
time: 1.03 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_B1_A2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5167717, upper bound: 2204.5316945
time: 1.02 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1061.4566650, 1698.7755127, -939.5111694, 1507.0028076, -2568.1918945, 2638.2858887
1: -1192.0683594, 1735.9645996, -1055.5273438, 1540.0412598, -2732.1091309, 2791.4914551
2: -1202.2851562, 1734.4344482, -1064.2187500, 1538.7885742, -2740.9577637, 2798.6533203
3: -1454.8771973, 2002.8786621, -1290.7064209, 1776.1739502, -3231.0512695, 3293.5849609
4: -1319.0875244, 1996.2081299, -1165.2409668, 1770.9030762, -3087.6289062, 3161.4492188

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B2_A2_A1_B1

### Relational analysis result of NS_B2_B1_A2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5315384
time: 1.14 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_A2_A1_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5394201
time: 1.17 seconds

## BFS NS instance: NS_B2_B1_A2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1070.9241943, 1713.6669922, -945.5847168, 1516.7375488, -2587.6616211, 2659.2512207
1: -1202.7758789, 1751.5672607, -1062.3392334, 1550.1030273, -2752.8789062, 2813.9062500
2: -1213.2213135, 1749.7055664, -1071.2106934, 1548.8017578, -2762.0224609, 2820.9162598
3: -1467.5590820, 2020.9715576, -1298.9228516, 1787.7633057, -3255.3222656, 3319.8942871
4: -1332.4658203, 2013.6577148, -1173.5976562, 1782.2905273, -3112.7932129, 3187.2546387

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B1_A2_A2_B2_A2_A2_B1

### Relational analysis result of NS_B2_B1_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5315384
time: 1.07 seconds

## Relational analysis of NS_B2_B1_A2_A2_B2_A2_A2_B2

### Relational analysis result of NS_B2_B1_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5394201
time: 1.27 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -738.0010376, 1183.0332031, -866.7879028, 1384.3229980, -2122.3239746, 2049.8210449
1: -829.5087891, 1209.2052002, -973.9923706, 1415.4042969, -2244.9130859, 2183.1975098
2: -836.7335815, 1208.9919434, -982.7849731, 1414.2880859, -2251.0214844, 2191.7768555
3: -1014.6687622, 1394.7266846, -1187.8803711, 1633.7955322, -2648.4643555, 2582.6069336
4: -919.5277100, 1391.1138916, -1083.2805176, 1628.1420898, -2547.6699219, 2474.3945312

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5442358, upper bound: 2204.5225997
time: 1.19 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5442358, upper bound: 2204.5236649
time: 1.15 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -738.0010376, 1183.0332031, -949.6515503, 1522.7784424, -2260.7790527, 2132.6848145
1: -829.5087891, 1209.2052002, -1067.5577393, 1557.0194092, -2386.5283203, 2276.7629395
2: -836.7335815, 1208.9919434, -1076.7757568, 1554.6339111, -2391.3674316, 2285.7673340
3: -1014.6687622, 1394.7266846, -1303.0151367, 1797.0718994, -2811.7407227, 2697.7416992
4: -919.5277100, 1391.1138916, -1187.9434814, 1788.6859131, -2708.2136230, 2579.0573730

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5374449, upper bound: 2204.5165858
time: 1.39 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5363333, upper bound: 2204.5164416
time: 0.88 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -821.7074585, 1322.4025879, -866.7879028, 1384.3229980, -2206.0305176, 2189.1904297
1: -924.1153564, 1352.1656494, -973.9923706, 1415.4042969, -2339.5195312, 2326.1579590
2: -931.9538574, 1350.0850830, -982.7849731, 1414.2880859, -2346.2419434, 2332.8701172
3: -1130.8178711, 1560.0310059, -1187.8803711, 1633.7955322, -2764.6130371, 2747.9113770
4: -1026.1292725, 1552.7178955, -1083.2805176, 1628.1420898, -2654.2712402, 2635.9985352

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5435403, upper bound: 2204.5235643
time: 0.99 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5415011, upper bound: 2204.5237958
time: 1.20 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -821.7074585, 1322.4025879, -949.6515503, 1522.7784424, -2344.4858398, 2272.0541992
1: -924.1153564, 1352.1656494, -1067.5577393, 1557.0194092, -2481.1347656, 2419.7233887
2: -931.9538574, 1350.0850830, -1076.7757568, 1554.6339111, -2486.5878906, 2426.8603516
3: -1130.8178711, 1560.0310059, -1303.0151367, 1797.0718994, -2927.8896484, 2863.0461426
4: -1026.1292725, 1552.7178955, -1187.9434814, 1788.6859131, -2814.8151855, 2740.1118164

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_B2_B2_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5440350, upper bound: 2204.5223183
time: 1.03 seconds

## Relational analysis of NS_B2_B2_A1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_B2_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5440350, upper bound: 2204.5239335
time: 1.25 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -724.8449707, 1162.7159424, -1056.0963135, 1690.4055176, -2415.2497559, 2218.8117676
1: -814.8334961, 1188.7115479, -1186.2130127, 1728.0408936, -2542.8745117, 2374.9238281
2: -821.9531250, 1188.1956787, -1196.5003662, 1725.9659424, -2547.9184570, 2384.6953125
3: -996.6428223, 1371.0705566, -1447.1956787, 1993.8131104, -2990.4560547, 2818.2661133
4: -903.7799072, 1366.9611816, -1314.5712891, 1986.1343994, -2889.9143066, 2681.5322266

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
time: 1.19 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
time: 0.92 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -757.5258789, 1213.4370117, -1059.2991943, 1695.0046387, -2452.5305176, 2272.7363281
1: -851.0978394, 1240.0061035, -1189.8250732, 1732.6026611, -2583.7004395, 2429.8308105
2: -858.6420898, 1239.7155762, -1200.1093750, 1730.7398682, -2589.3818359, 2439.8249512
3: -1041.1392822, 1430.5830078, -1451.5958252, 1999.0694580, -3040.2087402, 2882.1787109
4: -942.6447754, 1426.8299561, -1318.3714600, 1991.8298340, -2934.4743652, 2745.2011719

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
time: 1.03 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
time: 1.10 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -808.9209595, 1302.6428223, -1057.2833252, 1692.1943359, -2501.1152344, 2359.9262695
1: -909.8918457, 1332.2072754, -1187.5058594, 1729.8137207, -2639.7055664, 2519.7131348
2: -917.5776367, 1329.7988281, -1197.7962646, 1727.7225342, -2645.2998047, 2527.5947266
3: -1113.3408203, 1537.0091553, -1448.8184814, 1995.8872070, -3109.2280273, 2985.8276367
4: -1010.8040161, 1529.1622314, -1315.8110352, 1988.1081543, -2998.9116211, 2844.7099609

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
time: 1.12 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_A1_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
time: 1.25 seconds

## BFS NS instance: NS_B2_B2_A1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -824.2951660, 1325.1387939, -1059.6740723, 1695.5623779, -2519.8574219, 2384.8125000
1: -926.5723877, 1354.4716797, -1190.2036133, 1733.1246338, -2659.6970215, 2544.6752930
2: -934.6006470, 1352.6856689, -1200.4869385, 1731.2324219, -2665.8330078, 2553.1723633
3: -1133.8502197, 1563.1243896, -1452.1193848, 1999.7086182, -3133.5581055, 3015.2436523
4: -1027.9331055, 1555.9338379, -1318.6046143, 1992.3295898, -3020.2622070, 2874.3278809

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
time: 0.96 seconds

## Relational analysis of NS_B2_B2_A1_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_B2_A1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
time: 1.71 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -941.1321411, 1509.4635010, -863.0300903, 1378.4851074, -2319.6169434, 2372.4936523
1: -1057.3305664, 1542.5191650, -969.6259766, 1408.7573242, -2466.0878906, 2512.1450195
2: -1065.9932861, 1541.3685303, -978.2324829, 1408.1430664, -2474.1362305, 2519.6010742
3: -1292.9707031, 1778.9918213, -1182.9952393, 1625.8833008, -2918.8540039, 2961.9870605
4: -1166.7474365, 1773.9707031, -1076.5989990, 1621.2689209, -2788.0151367, 2849.4492188

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_A2_B1_A1_B1_A1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5316734, upper bound: 2204.5174806
time: 0.88 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A1_B1_A2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5314209, upper bound: 2204.5182751
time: 1.61 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -941.1321411, 1509.4635010, -874.4929199, 1396.6396484, -2337.7707520, 2383.9565430
1: -1057.3305664, 1542.5191650, -982.5802612, 1427.9438477, -2485.2741699, 2525.0993652
2: -1065.9932861, 1541.3685303, -991.5087891, 1426.7418213, -2492.7351074, 2532.8774414
3: -1292.9707031, 1778.9918213, -1198.3416748, 1648.3330078, -2941.3037109, 2977.3334961
4: -1166.7474365, 1773.9707031, -1092.8470459, 1642.4907227, -2809.2375488, 2864.7302246

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_A2_B1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5316734, upper bound: 2204.5174806
time: 1.02 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5314209, upper bound: 2204.5182915
time: 1.46 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -942.5073242, 1511.7509766, -863.0300903, 1378.4851074, -2320.9924316, 2374.7807617
1: -1058.8673096, 1544.9920654, -969.6259766, 1408.7573242, -2467.6245117, 2514.6176758
2: -1067.7215576, 1543.7102051, -978.2324829, 1408.1430664, -2475.8645020, 2521.9426270
3: -1294.6511230, 1781.8165283, -1182.9952393, 1625.8833008, -2920.5344238, 2964.8117676
4: -1169.7451172, 1776.4477539, -1076.5989990, 1621.2689209, -2791.0124512, 2851.9934082

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A1_A2_B1_A2_B1_B1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5286374, upper bound: 2204.5081132
time: 0.90 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A2_B1_B2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5283842, upper bound: 2204.5073357
time: 1.15 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -942.5073242, 1511.7509766, -874.4929199, 1396.6396484, -2339.1467285, 2386.2438965
1: -1058.8673096, 1544.9920654, -982.5802612, 1427.9438477, -2486.8110352, 2527.5717773
2: -1067.7215576, 1543.7102051, -991.5087891, 1426.7418213, -2494.4633789, 2535.2189941
3: -1294.6511230, 1781.8165283, -1198.3416748, 1648.3330078, -2942.9841309, 2980.1582031
4: -1169.7451172, 1776.4477539, -1092.8470459, 1642.4907227, -2812.2348633, 2868.6726074

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A1_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5391973, upper bound: 2204.5180840
time: 1.12 seconds

## Relational analysis of NS_B2_B2_A1_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_B2_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5327280, upper bound: 2204.5182751
time: 1.09 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -940.0057373, 1507.7000732, -1068.2529297, 1709.3980713, -2649.4038086, 2575.9528809
1: -1056.0407715, 1540.8543701, -1199.7783203, 1747.2451172, -2803.2858887, 2740.6328125
2: -1064.8690186, 1539.5784912, -1210.1839600, 1745.3962402, -2810.2648926, 2749.7624512
3: -1291.1767578, 1777.1774902, -1463.8399658, 2016.0474854, -3307.2241211, 3241.0173340
4: -1166.7269287, 1771.5815430, -1329.2818604, 2008.5332031, -3175.2600098, 3098.8305664

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_B2_A1_A2_B2_A1_B1_B1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5267827, upper bound: 2204.5066394
time: 1.10 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A1_B1_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5275859, upper bound: 2204.5059043
time: 1.29 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -940.0057373, 1507.7000732, -1087.0156250, 1738.4395752, -2678.4453125, 2594.7150879
1: -1056.0407715, 1540.8543701, -1220.5894775, 1776.8530273, -2832.8937988, 2761.4436035
2: -1064.8690186, 1539.5784912, -1231.3466797, 1775.2271729, -2840.0959473, 2770.9252930
3: -1291.1767578, 1777.1774902, -1488.8068848, 2050.5114746, -3341.6877441, 3265.9843750
4: -1166.7269287, 1771.5815430, -1352.8092041, 2042.9000244, -3209.6269531, 3121.7832031

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_B2_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5267827, upper bound: 2204.5127837
time: 1.15 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A1_B2_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5275859, upper bound: 2204.5131046
time: 1.14 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -956.4277954, 1533.5697021, -1068.2529297, 1709.3980713, -2665.8256836, 2601.8222656
1: -1074.2648926, 1566.9367676, -1199.7783203, 1747.2451172, -2821.5100098, 2766.7150879
2: -1083.3348389, 1566.1177979, -1210.1839600, 1745.3962402, -2828.7307129, 2776.3012695
3: -1313.1861572, 1807.4716797, -1463.8399658, 2016.0474854, -3329.2329102, 3271.3112793
4: -1187.0090332, 1802.0288086, -1329.2818604, 2008.5332031, -3195.5417480, 3129.2880859

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_B2_A1_A2_B2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153213, upper bound: 2204.5056807
time: 1.14 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A2_B1_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5183000, upper bound: 2204.5052571
time: 1.19 seconds

## BFS NS instance: NS_B2_B2_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -956.4277954, 1533.5697021, -1087.0156250, 1738.4395752, -2694.8674316, 2620.5578613
1: -1074.2648926, 1566.9367676, -1220.5894775, 1776.8530273, -2851.1179199, 2787.5261230
2: -1083.3348389, 1566.1177979, -1231.3466797, 1775.2271729, -2858.5617676, 2797.4641113
3: -1313.1861572, 1807.4716797, -1488.8068848, 2050.5114746, -3363.6970215, 3296.2783203
4: -1187.0090332, 1802.0288086, -1352.8092041, 2042.9000244, -3229.9091797, 3152.2407227

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_B2_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153213, upper bound: 2204.5127920
time: 0.94 seconds

## Relational analysis of NS_B2_B2_A1_A2_B2_A2_B2_B2

### Relational analysis result of NS_B2_B2_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5153213, upper bound: 2204.5132222
time: 0.88 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -863.0832520, 1378.5733643, -863.0300903, 1378.4851074, -2241.5683594, 2241.6035156
1: -969.6864624, 1408.8481445, -969.6259766, 1408.7573242, -2378.4438477, 2378.4741211
2: -978.2928467, 1408.2373047, -978.2324829, 1408.1430664, -2386.4360352, 2386.4694824
3: -1183.0726318, 1625.9886475, -1182.9952393, 1625.8833008, -2808.9560547, 2808.9838867
4: -1076.6649170, 1621.3752441, -1076.5989990, 1621.2689209, -2697.9331055, 2697.9738770

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B1_A1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5192264, upper bound: 2204.5195667
time: 1.09 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B1_A2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5189792, upper bound: 2204.5197107
time: 0.97 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -863.0832520, 1378.5733643, -874.4929199, 1396.6396484, -2259.7229004, 2253.0664062
1: -969.6864624, 1408.8481445, -982.5802612, 1427.9438477, -2397.6301270, 2391.4284668
2: -978.2928467, 1408.2373047, -991.5087891, 1426.7418213, -2405.0346680, 2399.7460938
3: -1183.0726318, 1625.9886475, -1198.3416748, 1648.3330078, -2831.4057617, 2824.3303223
4: -1076.6649170, 1621.3752441, -1092.8470459, 1642.4907227, -2719.1552734, 2714.2221680

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5192264, upper bound: 2204.5203583
time: 0.95 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5189792, upper bound: 2204.5205230
time: 1.22 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -874.5387573, 1396.7172852, -863.0300903, 1378.4851074, -2253.0239258, 2259.7473145
1: -982.6329346, 1428.0239258, -969.6259766, 1408.7573242, -2391.3901367, 2397.6496582
2: -991.5609131, 1426.8238525, -978.2324829, 1408.1430664, -2399.7041016, 2405.0559082
3: -1198.4099121, 1648.4262695, -1182.9952393, 1625.8833008, -2824.2927246, 2831.4213867
4: -1092.9044189, 1642.5850830, -1076.5989990, 1621.2689209, -2714.1726074, 2719.1835938

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5166363, upper bound: 2204.5188315
time: 1.10 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5186447, upper bound: 2204.5197037
time: 0.97 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -874.5387573, 1396.7172852, -874.4929199, 1396.6396484, -2271.1779785, 2271.2102051
1: -982.6329346, 1428.0239258, -982.5802612, 1427.9438477, -2410.5766602, 2410.6037598
2: -991.5609131, 1426.8238525, -991.5087891, 1426.7418213, -2418.3027344, 2418.3325195
3: -1198.4099121, 1648.4262695, -1198.3416748, 1648.3330078, -2846.7429199, 2846.7680664
4: -1092.9044189, 1642.5850830, -1092.8470459, 1642.4907227, -2735.3947754, 2735.4318848

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5189213, upper bound: 2204.5190797
time: 1.06 seconds

## Relational analysis of NS_B2_B2_A2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5185487, upper bound: 2204.5193561
time: 1.19 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -885.1538086, 1412.7916260, -1059.2991943, 1695.0046387, -2580.1582031, 2472.0905762
1: -994.2966309, 1444.2792969, -1189.8250732, 1732.6026611, -2726.8994141, 2634.1040039
2: -1003.3848877, 1443.2576904, -1200.1093750, 1730.7398682, -2734.1245117, 2643.3664551
3: -1212.7448730, 1667.3001709, -1451.5958252, 1999.0694580, -3211.8144531, 3118.8959961
4: -1104.8387451, 1661.7143555, -1318.3714600, 1991.8298340, -3095.8200684, 2980.0859375

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5154044, upper bound: 2204.5129780
time: 1.15 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5154044, upper bound: 2204.5129780
time: 1.44 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -936.1536865, 1501.6313477, -1054.1732178, 1687.0561523, -2623.2089844, 2555.8044434
1: -1052.3066406, 1535.0629883, -1183.9147949, 1724.1260986, -2776.4326172, 2718.9775391
2: -1061.3132324, 1532.7849121, -1194.0944824, 1722.4703369, -2783.7834473, 2726.8793945
3: -1284.7531738, 1771.6727295, -1444.6995850, 1989.2120361, -3273.9645996, 3216.3723145
4: -1169.7921143, 1763.6520996, -1310.5952148, 1982.3038330, -3149.3146973, 3073.1166992

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5183449, upper bound: 2204.5158890
time: 0.96 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_A1_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5183449, upper bound: 2204.5168101
time: 1.33 seconds

## BFS NS instance: NS_B2_B2_A2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -945.3922729, 1515.5843506, -1063.9345703, 1702.3288574, -2647.7209473, 2579.5190430
1: -1062.7562256, 1549.6903076, -1194.9339600, 1740.0004883, -2802.7565918, 2744.6242676
2: -1071.9542236, 1547.2956543, -1205.2934570, 1738.1237793, -2810.0781250, 2752.5891113
3: -1297.0401611, 1788.6059570, -1457.8916016, 2007.6459961, -3304.6860352, 3246.4973145
4: -1182.6572266, 1780.3306885, -1323.7491455, 2000.2609863, -3180.4987793, 3103.7253418

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5185374, upper bound: 2204.5158890
time: 1.59 seconds

## Relational analysis of NS_B2_B2_A2_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_B2_A2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5185374, upper bound: 2204.5168101
time: 1.31 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1061.6512451, 1699.0894775, -863.0300903, 1378.4851074, -2440.1362305, 2562.1188965
1: -1192.2880859, 1736.2871094, -969.6259766, 1408.7573242, -2601.0454102, 2705.9130859
2: -1202.5068359, 1734.7565918, -978.2324829, 1408.1430664, -2610.6494141, 2712.9890137
3: -1455.1474609, 2003.2508545, -1182.9952393, 1625.8833008, -3081.0305176, 3186.2460938
4: -1319.3314209, 1996.5783691, -1076.5989990, 1621.2689209, -2940.5991211, 3072.0883789

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A2_B1_A1_B1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4994490, upper bound: 2204.5114942
time: 1.07 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A1_B1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5155400, upper bound: 2204.5197732
time: 1.20 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1061.6512451, 1699.0894775, -874.4929199, 1396.6396484, -2458.2910156, 2573.5815430
1: -1192.2880859, 1736.2871094, -982.5802612, 1427.9438477, -2620.2319336, 2718.8671875
2: -1202.5068359, 1734.7565918, -991.5087891, 1426.7418213, -2629.2482910, 2726.2653809
3: -1455.1474609, 2003.2508545, -1198.3416748, 1648.3330078, -3103.4804688, 3201.5925293
4: -1319.3314209, 1996.5783691, -1092.8470459, 1642.4907227, -2961.8215332, 3087.3691406

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A2_B1_A1_B2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4994490, upper bound: 2204.5114942
time: 1.26 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A1_B2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5155400, upper bound: 2204.5205648
time: 1.15 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1071.1383057, 1714.0062256, -863.0300903, 1378.4851074, -2449.6235352, 2577.0363770
1: -1203.0167236, 1751.9155273, -969.6259766, 1408.7573242, -2611.7739258, 2721.5415039
2: -1213.4631348, 1750.0543213, -978.2324829, 1408.1430664, -2621.6062012, 2728.2863770
3: -1467.8559570, 2021.3729248, -1182.9952393, 1625.8833008, -3093.7392578, 3204.3681641
4: -1332.7282715, 2014.0590820, -1076.5989990, 1621.2689209, -2953.9965820, 3089.5004883

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5005296, upper bound: 2204.5114942
time: 1.07 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146481, upper bound: 2204.5195205
time: 0.98 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1071.1383057, 1714.0062256, -874.4929199, 1396.6396484, -2467.7778320, 2588.4990234
1: -1203.0167236, 1751.9155273, -982.5802612, 1427.9438477, -2630.9604492, 2734.4956055
2: -1213.4631348, 1750.0543213, -991.5087891, 1426.7418213, -2640.2050781, 2741.5629883
3: -1467.8559570, 2021.3729248, -1198.3416748, 1648.3330078, -3116.1889648, 3219.7145996
4: -1332.7282715, 2014.0590820, -1092.8470459, 1642.4907227, -2975.2187500, 3106.1655273

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5005297, upper bound: 2204.5114942
time: 1.24 seconds

## Relational analysis of NS_B2_B2_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_B2_B2_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146481, upper bound: 2204.5195288
time: 1.15 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1061.6512451, 1699.0894775, -1061.5988770, 1699.0002441, -2760.2722168, 2760.3132324
1: -1192.2880859, 1736.2871094, -1192.2280273, 1736.1956787, -2928.4838867, 2928.5151367
2: -1202.5068359, 1734.7565918, -1202.4468994, 1734.6623535, -2936.9489746, 2936.9863281
3: -1455.1474609, 2003.2508545, -1455.0706787, 2003.1468506, -3458.2944336, 3458.3215332
4: -1319.3314209, 1996.5783691, -1319.2655029, 1996.4702148, -3313.3942871, 3313.4519043

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.4944328, upper bound: 2204.4908418
time: 1.06 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.4878203, upper bound: 2204.4889876
time: 1.23 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1061.6512451, 1699.0894775, -1071.0825195, 1713.9149170, -2775.0991211, 2769.6965332
1: -1192.2880859, 1736.2871094, -1202.9534912, 1751.8222656, -2944.1103516, 2939.2404785
2: -1202.5068359, 1734.7565918, -1213.4005127, 1749.9582520, -2952.1245117, 2947.7463379
3: -1455.1474609, 2003.2508545, -1467.7751465, 2021.2663574, -3476.4133301, 3471.0258789
4: -1319.3314209, 1996.5783691, -1332.6608887, 2013.9488525, -3330.8046875, 3325.9338379

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.4944328, upper bound: 2204.4908418
time: 1.09 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.4878203, upper bound: 2204.5046592
time: 0.97 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1071.1383057, 1714.0062256, -1061.5988770, 1699.0002441, -2769.6589355, 2775.1425781
1: -1203.0167236, 1751.9155273, -1192.2280273, 1736.1956787, -2939.2124023, 2944.1435547
2: -1213.4631348, 1750.0543213, -1202.4468994, 1734.6623535, -2947.7121582, 2952.1633301
3: -1467.8559570, 2021.3729248, -1455.0706787, 2003.1468506, -3471.0029297, 3476.4436035
4: -1332.7282715, 2014.0590820, -1319.2655029, 1996.4702148, -3325.8784180, 3330.8640137

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B1_B1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5125546, upper bound: 2204.5150602
time: 1.19 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B1_B2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5145468, upper bound: 2204.5152988
time: 1.47 seconds

## BFS NS instance: NS_B2_B2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1071.1383057, 1714.0062256, -1071.0825195, 1713.9149170, -2785.0532227, 2785.0888672
1: -1203.0167236, 1751.9155273, -1202.9534912, 1751.8222656, -2954.8388672, 2954.8691406
2: -1213.4631348, 1750.0543213, -1213.4005127, 1749.9582520, -2963.4213867, 2963.4548340
3: -1467.8559570, 2021.3729248, -1467.7751465, 2021.2663574, -3489.1215820, 3489.1479492
4: -1332.7282715, 2014.0590820, -1332.6608887, 2013.9488525, -3344.6206055, 3344.6779785

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.4943805, upper bound: 2204.5060965
time: 1.47 seconds

## Relational analysis of NS_B2_B2_A2_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_B2_A2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.4894385, upper bound: 2204.5047369
time: 1.36 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.40 seconds
NS_B2_B1_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5036381, upper bound: 2204.5349018
NS_B2_B1_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5028606, upper bound: 2204.5351728
NS_B2_B1_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5036381, upper bound: 2204.5353876
NS_B2_B1_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5028606, upper bound: 2204.5356826
NS_B2_B1_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4958067, upper bound: 2204.5255730
NS_B2_B1_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4949185, upper bound: 2204.5260227
NS_B2_B1_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4958067, upper bound: 2204.5255730
NS_B2_B1_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4949185, upper bound: 2204.5260227
NS_B2_B1_A1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
NS_B2_B1_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
NS_B2_B1_A1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
NS_B2_B1_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4941005, upper bound: 2204.5230245
NS_B2_B1_A1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4948096, upper bound: 2204.5227766
NS_B2_B1_A1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4948638, upper bound: 2204.5318728
NS_B2_B1_A1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4948638, upper bound: 2204.5227766
NS_B2_B1_A1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4948638, upper bound: 2204.5318728
NS_B2_B1_A1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4785301, upper bound: 2204.5157362
NS_B2_B1_A1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4781161, upper bound: 2204.5149497
NS_B2_B1_A1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4785301, upper bound: 2204.5157362
NS_B2_B1_A1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4781161, upper bound: 2204.5249152
NS_B2_B1_A1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4786788, upper bound: 2204.5157362
NS_B2_B1_A1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4782407, upper bound: 2204.5149497
NS_B2_B1_A1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4786788, upper bound: 2204.5177033
NS_B2_B1_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4782407, upper bound: 2204.5169918
NS_B2_B1_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4933583, upper bound: 2204.5220440
NS_B2_B1_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4894998, upper bound: 2204.5199754
NS_B2_B1_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4933583, upper bound: 2204.5242300
NS_B2_B1_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4894998, upper bound: 2204.5226230
NS_B2_B1_A1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4795312, upper bound: 2204.5122855
NS_B2_B1_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4909391, upper bound: 2204.5224853
NS_B2_B1_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4826804, upper bound: 2204.5149393
NS_B2_B1_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4795267, upper bound: 2204.5133209
NS_B2_B1_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5546682, upper bound: 2204.5546682
NS_B2_B1_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5546682, upper bound: 2204.5643744
NS_B2_B1_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5643744, upper bound: 2204.5546682
NS_B2_B1_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5643744, upper bound: 2204.5723292
NS_B2_B1_A2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5578837, upper bound: 2204.5531879
NS_B2_B1_A2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5574764, upper bound: 2204.5531239
NS_B2_B1_A2_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5578837, upper bound: 2204.5532098
NS_B2_B1_A2_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5570450, upper bound: 2204.5528669
NS_B2_B1_A2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5512493, upper bound: 2204.5543248
NS_B2_B1_A2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5512493, upper bound: 2204.5636928
NS_B2_B1_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5607205, upper bound: 2204.5543248
NS_B2_B1_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5607205, upper bound: 2204.5723263
NS_B2_B1_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5512193
NS_B2_B1_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5512194, upper bound: 2204.5605929
NS_B2_B1_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5605929, upper bound: 2204.5512193
NS_B2_B1_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5605929, upper bound: 2204.5688122
NS_B2_B1_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5177558, upper bound: 2204.5346044
NS_B2_B1_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5177558, upper bound: 2204.5424775
NS_B2_B1_A2_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5128233, upper bound: 2204.5372688
NS_B2_B1_A2_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5119372, upper bound: 2204.5359160
NS_B2_B1_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5148791, upper bound: 2204.5422468
NS_B2_B1_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5135577, upper bound: 2204.5422468
NS_B2_B1_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5151354, upper bound: 2204.5422373
NS_B2_B1_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5137732, upper bound: 2204.5422373
NS_B2_B1_A2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5178228, upper bound: 2204.5384593
NS_B2_B1_A2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5165487, upper bound: 2204.5382341
NS_B2_B1_A2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5178358, upper bound: 2204.5335731
NS_B2_B1_A2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5167717, upper bound: 2204.5316945
NS_B2_B1_A2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5315384
NS_B2_B1_A2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5394201
NS_B2_B1_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5315384
NS_B2_B1_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5137873, upper bound: 2204.5394201
NS_B2_B2_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5442358, upper bound: 2204.5225997
NS_B2_B2_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5442358, upper bound: 2204.5236649
NS_B2_B2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5374449, upper bound: 2204.5165858
NS_B2_B2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5363333, upper bound: 2204.5164416
NS_B2_B2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5435403, upper bound: 2204.5235643
NS_B2_B2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5415011, upper bound: 2204.5237958
NS_B2_B2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5440350, upper bound: 2204.5223183
NS_B2_B2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5440350, upper bound: 2204.5239335
NS_B2_B2_A1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
NS_B2_B2_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
NS_B2_B2_A1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
NS_B2_B2_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5340634, upper bound: 2204.5114693
NS_B2_B2_A1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
NS_B2_B2_A1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
NS_B2_B2_A1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
NS_B2_B2_A1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5327192, upper bound: 2204.5112196
NS_B2_B2_A1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5316734, upper bound: 2204.5174806
NS_B2_B2_A1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5314209, upper bound: 2204.5182751
NS_B2_B2_A1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5316734, upper bound: 2204.5174806
NS_B2_B2_A1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5314209, upper bound: 2204.5182915
NS_B2_B2_A1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5286374, upper bound: 2204.5081132
NS_B2_B2_A1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5283842, upper bound: 2204.5073357
NS_B2_B2_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5391973, upper bound: 2204.5180840
NS_B2_B2_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5327280, upper bound: 2204.5182751
NS_B2_B2_A1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5267827, upper bound: 2204.5066394
NS_B2_B2_A1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5275859, upper bound: 2204.5059043
NS_B2_B2_A1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5267827, upper bound: 2204.5127837
NS_B2_B2_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5275859, upper bound: 2204.5131046
NS_B2_B2_A1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5153213, upper bound: 2204.5056807
NS_B2_B2_A1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5183000, upper bound: 2204.5052571
NS_B2_B2_A1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5153213, upper bound: 2204.5127920
NS_B2_B2_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5153213, upper bound: 2204.5132222
NS_B2_B2_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5192264, upper bound: 2204.5195667
NS_B2_B2_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5189792, upper bound: 2204.5197107
NS_B2_B2_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5192264, upper bound: 2204.5203583
NS_B2_B2_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5189792, upper bound: 2204.5205230
NS_B2_B2_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5166363, upper bound: 2204.5188315
NS_B2_B2_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5186447, upper bound: 2204.5197037
NS_B2_B2_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5189213, upper bound: 2204.5190797
NS_B2_B2_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5185487, upper bound: 2204.5193561
NS_B2_B2_A2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5154044, upper bound: 2204.5129780
NS_B2_B2_A2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5154044, upper bound: 2204.5129780
NS_B2_B2_A2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5183449, upper bound: 2204.5158890
NS_B2_B2_A2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5183449, upper bound: 2204.5168101
NS_B2_B2_A2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5185374, upper bound: 2204.5158890
NS_B2_B2_A2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5185374, upper bound: 2204.5168101
NS_B2_B2_A2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4994490, upper bound: 2204.5114942
NS_B2_B2_A2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5155400, upper bound: 2204.5197732
NS_B2_B2_A2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4994490, upper bound: 2204.5114942
NS_B2_B2_A2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5155400, upper bound: 2204.5205648
NS_B2_B2_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5005296, upper bound: 2204.5114942
NS_B2_B2_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5146481, upper bound: 2204.5195205
NS_B2_B2_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5005297, upper bound: 2204.5114942
NS_B2_B2_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5146481, upper bound: 2204.5195288
NS_B2_B2_A2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4944328, upper bound: 2204.4908418
NS_B2_B2_A2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4878203, upper bound: 2204.4889876
NS_B2_B2_A2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4944328, upper bound: 2204.4908418
NS_B2_B2_A2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4878203, upper bound: 2204.5046592
NS_B2_B2_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5125546, upper bound: 2204.5150602
NS_B2_B2_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.5145468, upper bound: 2204.5152988
NS_B2_B2_A2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4943805, upper bound: 2204.5060965
NS_B2_B2_A2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 4.40
Output dim: 3, lower bound: -2204.4894385, upper bound: 2204.5047369

## BFS NS instance: NS_B2_B1_A1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -728.4932861, 1169.0129395, -730.1983032, 1170.3134766, -1898.8067627, 1899.2111816
1: -819.0516357, 1194.5781250, -820.6551514, 1195.8411865, -2014.8928223, 2015.2332764
2: -825.7406616, 1194.2868652, -827.6820068, 1195.8839111, -2021.6245117, 2021.9688721
3: -1002.5234375, 1377.5269775, -1004.1336670, 1379.2167969, -2381.7397461, 2381.6606445
4: -906.1610718, 1374.6492920, -908.3312988, 1376.2755127, -2282.4365234, 2282.9804688

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5029622, upper bound: 2204.5349912
time: 1.14 seconds

## Relational analysis of NS_B2_B1_A1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5029622, upper bound: 2204.5349912
time: 1.02 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.29 + 416.99 = 420.28 seconds
