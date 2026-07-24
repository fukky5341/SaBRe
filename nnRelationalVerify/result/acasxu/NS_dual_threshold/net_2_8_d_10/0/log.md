## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 24053.457489852968


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13645.4052734, 14396.3037109, -13645.4052734, 14396.3037109, -28041.7089844, 28041.7089844)
1: (-1456.9052734, 1049.7773438, -1456.9052734, 1049.7773438, -2506.6826172, 2506.6826172)
2: (-2374.5815430, 2731.1796875, -2374.5815430, 2731.1796875, -5105.7607422, 5105.7607422)
3: (-2753.4853516, 1715.0015869, -2753.4853516, 1715.0015869, -4468.4868164, 4468.4868164)
4: (-2079.3156738, 2193.2937012, -2079.3156738, 2193.2937012, -4272.6093750, 4272.6093750)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.66 + 2.22 = 4.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -24054.9007839, upper bound: 24054.9007839

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8852932, upper bound: 24054.8862473
time: 1.05 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8851038, upper bound: 24054.8851038
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.16 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 0, lower bound: -24054.8852932, upper bound: 24054.8862473
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.16
Output dim: 0, lower bound: -24054.8851038, upper bound: 24054.8851038

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13342.4511719, 14096.1416016, -13457.2392578, 14209.7343750, -27552.1855469, 27553.3789062
1: -1425.9169922, 1026.3793945, -1437.6535645, 1035.2438965, -2461.1606445, 2464.0329590
2: -2322.6171875, 2673.5187988, -2342.3442383, 2695.3393555, -5017.9565430, 5015.8632812
3: -2694.0593262, 1678.2844238, -2716.6882324, 1692.1850586, -4386.2441406, 4394.9726562
4: -2033.7200928, 2147.3442383, -2051.0771484, 2164.7180176, -4198.4370117, 4198.4213867

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8845177, upper bound: 24054.8845177
time: 0.79 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8845177, upper bound: 24054.8846438
time: 0.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13869.1933594, 14612.4423828, -13101.1904297, 13837.3330078, -27706.5234375, 27713.6308594
1: -1480.0068359, 1066.2563477, -1399.8630371, 1007.3348999, -2487.3413086, 2466.1193848
2: -2416.3530273, 2771.3415527, -2281.6853027, 2624.6042480, -5040.9570312, 5053.0263672
3: -2802.2524414, 1742.3112793, -2646.4514160, 1648.0278320, -4450.2802734, 4388.7626953
4: -2115.7363281, 2226.2524414, -1997.1914062, 2108.0175781, -4223.7534180, 4223.4438477

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8846438, upper bound: 24054.8849860
time: 0.72 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8846438, upper bound: 24054.8851038
time: 0.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.28 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -24054.8845177, upper bound: 24054.8845177
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -24054.8845177, upper bound: 24054.8846438
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -24054.8846438, upper bound: 24054.8849860
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -24054.8846438, upper bound: 24054.8851038

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -13342.4511719, 14096.1416016, -13342.4511719, 14096.1416016, -27438.5937500, 27438.5937500
1: -1425.9169922, 1026.3793945, -1425.9169922, 1026.3793945, -2452.2963867, 2452.2963867
2: -2322.6171875, 2673.5187988, -2322.6171875, 2673.5187988, -4996.1357422, 4996.1357422
3: -2694.0593262, 1678.2844238, -2694.0593262, 1678.2844238, -4372.3432617, 4372.3432617
4: -2033.7200928, 2147.3442383, -2033.7200928, 2147.3442383, -4181.0634766, 4181.0634766

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8796815, upper bound: 24054.8839683
time: 0.83 seconds

## Relational analysis of NS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8750659, upper bound: 24054.8778432
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8807761, upper bound: 24054.8823834
time: 0.74 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -13342.4511719, 14096.1416016, -13867.5761719, 14610.9082031, -27953.3593750, 27963.7128906
1: -1425.9169922, 1026.3793945, -1479.8498535, 1066.1276855, -2492.0441895, 2506.2292480
2: -2322.6171875, 2673.5187988, -2416.0803223, 2771.0493164, -5093.6660156, 5089.5991211
3: -2694.0593262, 1678.2844238, -2801.9584961, 1742.1221924, -4436.1811523, 4480.2421875
4: -2033.7200928, 2147.3442383, -2115.5019531, 2226.0173340, -4259.7368164, 4262.8452148

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7763305, upper bound: 24054.7632091
time: 0.79 seconds

## Relational analysis of NS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8845485, upper bound: 24054.8859662
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8849821, upper bound: 24054.8859788
time: 0.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13869.1933594, 14612.4423828, -13342.4511719, 14096.1416016, -27965.3300781, 27954.8945312
1: -1480.0068359, 1066.2563477, -1425.9169922, 1026.3793945, -2506.3862305, 2492.1733398
2: -2416.3530273, 2771.3415527, -2322.6171875, 2673.5187988, -5089.8720703, 5093.9589844
3: -2802.2524414, 1742.3112793, -2694.0593262, 1678.2844238, -4480.5366211, 4436.3701172
4: -2115.7363281, 2226.2524414, -2033.7200928, 2147.3442383, -4263.0800781, 4259.9721680

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8763354, upper bound: 24054.8764845
time: 0.77 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8764951, upper bound: 24054.8764423
time: 1.11 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13869.1933594, 14612.4423828, -13869.1933594, 14612.4423828, -28481.6328125, 28481.6328125
1: -1480.0068359, 1066.2563477, -1480.0068359, 1066.2563477, -2546.2631836, 2546.2631836
2: -2416.3530273, 2771.3415527, -2416.3530273, 2771.3415527, -5187.6943359, 5187.6943359
3: -2802.2524414, 1742.3112793, -2802.2524414, 1742.3112793, -4544.5634766, 4544.5629883
4: -2115.7363281, 2226.2524414, -2115.7363281, 2226.2524414, -4341.9887695, 4341.9882812

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8647381, upper bound: 24054.8439360
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8809428, upper bound: 24054.8815312
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.09 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8750659, upper bound: 24054.8778432
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8807761, upper bound: 24054.8823834
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8845485, upper bound: 24054.8859662
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8849821, upper bound: 24054.8859788
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8763354, upper bound: 24054.8764845
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8764951, upper bound: 24054.8764423
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8647381, upper bound: 24054.8439360
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -24054.8809428, upper bound: 24054.8815312

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12955.4189453, 13693.2548828, -13039.9501953, 13778.8115234, -26734.2304688, 26733.2050781
1: -1385.4168701, 996.4285278, -1393.8607178, 1003.0620117, -2388.4790039, 2390.2890625
2: -2256.5134277, 2597.6437988, -2270.4831543, 2613.9621582, -4870.4755859, 4868.1269531
3: -2618.5075684, 1630.9163818, -2634.5183105, 1640.6440430, -4259.1513672, 4265.4340820
4: -1976.0906982, 2085.4936523, -1988.3824463, 2099.0266113, -4075.1171875, 4073.8757324

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8649056, upper bound: 24054.8654055
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8799376, upper bound: 24054.8816896
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13085.2841797, 13834.6689453, -13172.7851562, 13922.9267578, -27008.2109375, 27007.4531250
1: -1399.0742188, 1006.5308838, -1408.1651611, 1013.2791748, -2412.3532715, 2414.6960449
2: -2277.5136719, 2623.4997559, -2292.8603516, 2640.3703613, -4917.8837891, 4916.3603516
3: -2641.1506348, 1646.6270752, -2659.1416016, 1657.3413086, -4298.4921875, 4305.7685547
4: -1993.7349854, 2107.4829102, -2007.3707275, 2120.9272461, -4114.6616211, 4114.8535156

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8659643, upper bound: 24054.8660021
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8819638, upper bound: 24054.8819638
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13167.5068359, 13923.4003906, -13716.8212891, 14454.8603516, -27622.3652344, 27640.2089844
1: -1408.1717529, 1012.8718872, -1463.9282227, 1054.5587158, -2462.7304688, 2476.7995605
2: -2292.4526367, 2640.3764648, -2390.4038086, 2741.4465332, -5033.8994141, 5030.7802734
3: -2659.5412598, 1657.2019043, -2772.6376953, 1723.2960205, -4382.8374023, 4429.8398438
4: -2007.1970215, 2120.9255371, -2093.1220703, 2202.4113770, -4209.6083984, 4214.0478516

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8776881, upper bound: 24054.8803529
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8785122, upper bound: 24054.8806849
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13786.6484375, 14601.1884766, -13708.0185547, 14447.3876953, -28234.0351562, 28309.2031250
1: -1474.9703369, 1060.5343018, -1463.2321777, 1053.8007812, -2528.7709961, 2523.7663574
2: -2398.9462891, 2763.8977051, -2388.8776855, 2740.2421875, -5139.1879883, 5152.7753906
3: -2776.9780273, 1735.3404541, -2771.1555176, 1722.4190674, -4499.3969727, 4506.4960938
4: -2096.2194824, 2223.6469727, -2091.9963379, 2201.0788574, -4297.2973633, 4315.6430664

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8782800, upper bound: 24054.8807065
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8784854, upper bound: 24054.8806479
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -13266.7626953, 13932.9716797, -12472.0908203, 13055.4951172, -26322.2519531, 26405.0605469
1: -1412.7967529, 1019.9873657, -1324.8291016, 959.4407959, -2372.2373047, 2344.8164062
2: -2311.9609375, 2644.5771484, -2171.1157227, 2481.1069336, -4793.0678711, 4815.6928711
3: -2683.1799316, 1664.1010742, -2520.0539551, 1560.8719482, -4244.0517578, 4184.1547852
4: -2025.9898682, 2123.3537598, -1903.9719238, 1990.8504639, -4016.8403320, 4027.3251953

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8805979, upper bound: 24054.8782631
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8805979, upper bound: 24054.8782800
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -13724.2431641, 14457.6845703, -13132.0644531, 13877.3300781, -27601.5703125, 27589.7500000
1: -1464.3634033, 1055.1676025, -1403.7093506, 1010.1965942, -2474.5600586, 2458.8767090
2: -2391.2460938, 2742.3557129, -2286.3735352, 2632.4738770, -5023.7192383, 5028.7290039
3: -2773.6784668, 1723.8555908, -2653.1586914, 1652.0883789, -4425.7666016, 4377.0141602
4: -2094.0854492, 2202.9516602, -2002.5141602, 2114.1879883, -4208.2734375, 4205.4648438

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8806849, upper bound: 24054.8784567
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8806849, upper bound: 24054.8785122
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13634.4638672, 14371.0957031, -13655.6064453, 14387.5087891, -28021.9726562, 28026.7031250
1: -1455.1502686, 1049.0706787, -1457.2849121, 1049.7412109, -2504.8913574, 2506.3554688
2: -2375.2089844, 2724.2946777, -2379.5009766, 2729.1064453, -5104.3154297, 5103.7954102
3: -2752.8295898, 1713.5400391, -2759.6704102, 1715.6684570, -4468.4980469, 4473.2104492
4: -2078.6237793, 2188.4738770, -2083.4101562, 2192.0991211, -4270.7221680, 4271.8837891

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8286936, upper bound: 24054.8075108
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8646964, upper bound: 24054.8432956
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13393.3017578, 14118.1445312, -13524.1728516, 14250.2197266, -27643.5214844, 27642.3164062
1: -1429.9232178, 1029.7365723, -1443.4577637, 1039.8051758, -2469.7282715, 2473.1938477
2: -2333.3525391, 2677.9626465, -2356.1660156, 2703.1049805, -5036.4575195, 5034.1284180
3: -2706.9335938, 1683.3228760, -2733.2121582, 1699.3253174, -4406.2587891, 4416.5351562
4: -2043.2471924, 2150.7902832, -2063.3117676, 2171.0012207, -4214.2480469, 4214.1020508

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8712761, upper bound: 24054.8704866
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8725090, upper bound: 24054.8729142
time: 0.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.38 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8649056, upper bound: 24054.8654055
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8799376, upper bound: 24054.8816896
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8659643, upper bound: 24054.8660021
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8819638, upper bound: 24054.8819638
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8776881, upper bound: 24054.8803529
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8785122, upper bound: 24054.8806849
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8782800, upper bound: 24054.8807065
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8784854, upper bound: 24054.8806479
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8805979, upper bound: 24054.8782631
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8805979, upper bound: 24054.8782800
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8806849, upper bound: 24054.8784567
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8806849, upper bound: 24054.8785122
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8646964, upper bound: 24054.8432956
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8712761, upper bound: 24054.8704866
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -24054.8725090, upper bound: 24054.8729142

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12832.2734375, 13569.0400391, -12788.3115234, 13518.2910156, -26350.5644531, 26357.3515625
1: -1372.6970215, 986.9024048, -1367.3583984, 982.7151489, -2355.4121094, 2354.2607422
2: -2235.2929688, 2574.3439941, -2227.1745605, 2565.4113770, -4800.7041016, 4801.5185547
3: -2594.5241699, 1615.7955322, -2585.0981445, 1608.9255371, -4203.4497070, 4200.8935547
4: -1957.5842285, 2066.8486328, -1950.7424316, 2060.8090820, -4018.3933105, 4017.5903320

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8584882, upper bound: 24054.8590940
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8607713, upper bound: 24054.8606382
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12647.9375000, 13336.3623047, -12523.5107422, 13169.9482422, -25817.8867188, 25859.8730469
1: -1350.6326904, 972.8373413, -1334.7235107, 963.4800415, -2314.1127930, 2307.5607910
2: -2204.9321289, 2533.0344238, -2183.4719238, 2504.2448730, -4709.1767578, 4716.5063477
3: -2560.1196289, 1590.0103760, -2535.8569336, 1571.2410889, -4131.3608398, 4125.8671875
4: -1932.3404541, 2032.9986572, -1914.7138672, 2009.7082520, -3942.0473633, 3947.7124023

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8705114, upper bound: 24054.8746975
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8748910, upper bound: 24054.8750780
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -12839.4394531, 13580.6660156, -13045.9794922, 13795.4873047, -26634.9257812, 26626.6445312
1: -1373.2185059, 986.6179199, -1395.1594238, 1003.4655762, -2376.6835938, 2381.7773438
2: -2235.4948730, 2576.4689941, -2271.2524414, 2616.5205078, -4852.0151367, 4847.7216797
3: -2593.6928711, 1615.7166748, -2634.9318848, 1641.8253174, -4235.5180664, 4250.6484375
4: -1957.3819580, 2070.2087402, -1988.6129150, 2101.7790527, -4059.1611328, 4058.8212891

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8578320, upper bound: 24054.8616961
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8458022, upper bound: 24054.8459303
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -12558.1767578, 13213.7382812, -12858.0205078, 13556.3984375, -26114.5703125, 26071.7578125
1: -1338.6820068, 966.1250000, -1372.3941650, 989.0989380, -2327.7810059, 2338.5190430
2: -2188.7148438, 2511.4912109, -2239.9724121, 2574.0241699, -4762.7392578, 4751.4633789
3: -2540.4770508, 1575.8251953, -2599.2968750, 1615.3730469, -4155.8500977, 4175.1220703
4: -1918.4913330, 2016.2056885, -1962.4569092, 2066.9399414, -3985.4311523, 3978.6623535

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8730745, upper bound: 24054.8769676
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8774128, upper bound: 24054.8774128
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -12275.8974609, 12852.7968750, -13091.4853516, 13748.5166016, -26024.4140625, 25944.2812500
1: -1304.3310547, 944.3629761, -1394.1838379, 1006.4874878, -2310.8183594, 2338.5466309
2: -2137.4736328, 2442.9379883, -2282.0852051, 2609.9299316, -4747.4028320, 4725.0229492
3: -2482.0854492, 1536.5905762, -2649.3027344, 1642.0648193, -4124.1503906, 4185.8935547
4: -1874.8510742, 1960.0054932, -2000.1484375, 2095.5151367, -3970.3662109, 3960.1538086

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8776881, upper bound: 24054.8803528
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8776881, upper bound: 24054.8803528
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -12972.9052734, 13719.3583984, -13584.2529297, 14314.3974609, -27287.3027344, 27303.6113281
1: -1387.4932861, 997.9098511, -1449.7312012, 1044.3758545, -2431.8691406, 2447.6408691
2: -2258.8027344, 2602.2397461, -2367.3332520, 2715.1035156, -4973.9052734, 4969.5732422
3: -2621.5263672, 1632.8524170, -2746.4973145, 1706.5461426, -4328.0717773, 4379.3496094
4: -1978.2132568, 2090.0517578, -2073.2304688, 2181.1723633, -4159.3857422, 4163.2817383

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8784567, upper bound: 24054.8806849
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8784567, upper bound: 24054.8806849
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -12930.6435547, 13584.5292969, -13129.0000000, 13794.9394531, -26725.5781250, 26713.5273438
1: -1376.1004639, 994.3010254, -1398.6281738, 1009.3098755, -2385.4104004, 2392.9291992
2: -2250.5549316, 2575.9577637, -2288.5126953, 2618.3264160, -4868.8808594, 4864.4707031
3: -2607.9326172, 1620.4578857, -2656.5407715, 1647.2707520, -4255.2031250, 4276.9975586
4: -1969.6529541, 2070.1787109, -2005.6097412, 2102.2109375, -4071.8635254, 4075.7885742

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8782631, upper bound: 24054.8805979
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8782631, upper bound: 24054.8805979
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -13556.4375000, 14354.1757812, -13532.1601562, 14255.6933594, -27812.1308594, 27886.3359375
1: -1450.1052246, 1042.8200684, -1443.9970703, 1040.4139404, -2490.5190430, 2486.8171387
2: -2359.0209961, 2717.7607422, -2358.5354004, 2704.5717773, -5063.5917969, 5076.2958984
3: -2731.5778809, 1706.1901855, -2736.6247559, 1699.7496338, -4431.3266602, 4442.8144531
4: -2061.8776855, 2186.2861328, -2065.9589844, 2172.3605957, -4234.2377930, 4252.2446289

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8784176, upper bound: 24054.8806479
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8784176, upper bound: 24054.8806479
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -13307.3593750, 13918.3876953, -12472.0908203, 13055.4951172, -26362.8515625, 26390.4785156
1: -1413.4774170, 1022.9713745, -1324.8291016, 959.4407959, -2372.9182129, 2347.8005371
2: -2320.4814453, 2643.0769043, -2171.1157227, 2481.1069336, -4801.5878906, 4814.1923828
3: -2694.3903809, 1665.4481201, -2520.0539551, 1560.8719482, -4255.2622070, 4185.5014648
4: -2035.4886475, 2121.2873535, -1903.9719238, 1990.8504639, -4026.3391113, 4025.2592773

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8802074, upper bound: 24054.8747057
time: 0.74 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8773543, upper bound: 24054.8743454
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -13630.3164062, 14359.6484375, -12472.0908203, 13055.4951172, -26685.8066406, 26831.7382812
1: -1454.3894043, 1047.9506836, -1324.8291016, 959.4407959, -2413.8300781, 2372.7797852
2: -2374.8720703, 2723.8681641, -2171.1157227, 2481.1069336, -4855.9790039, 4894.9838867
3: -2755.0297852, 1712.1046143, -2520.0539551, 1560.8719482, -4315.9018555, 4232.1577148
4: -2079.8920898, 2188.0883789, -1903.9719238, 1990.8504639, -4070.7426758, 4092.0598145

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8802074, upper bound: 24054.8749795
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8773543, upper bound: 24054.8747649
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -13307.3593750, 13918.3876953, -13132.0644531, 13877.3300781, -27184.6855469, 27050.4531250
1: -1413.4774170, 1022.9713745, -1403.7093506, 1010.1965942, -2423.6740723, 2426.6801758
2: -2320.4814453, 2643.0769043, -2286.3735352, 2632.4738770, -4952.9550781, 4929.4501953
3: -2694.3903809, 1665.4481201, -2653.1586914, 1652.0883789, -4346.4785156, 4318.6064453
4: -2035.4886475, 2121.2873535, -2002.5141602, 2114.1879883, -4149.6767578, 4123.8017578

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8796703, upper bound: 24054.8747375
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8771834, upper bound: 24054.8742466
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -13630.3164062, 14359.6484375, -13132.0644531, 13877.3300781, -27507.6406250, 27491.7128906
1: -1454.3894043, 1047.9506836, -1403.7093506, 1010.1965942, -2464.5859375, 2451.6599121
2: -2374.8720703, 2723.8681641, -2286.3735352, 2632.4738770, -5007.3457031, 5010.2416992
3: -2755.0297852, 1712.1046143, -2653.1586914, 1652.0883789, -4407.1181641, 4365.2631836
4: -2079.8920898, 2188.0883789, -2002.5141602, 2114.1879883, -4194.0800781, 4190.6025391

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8796703, upper bound: 24054.8750131
time: 0.75 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8771834, upper bound: 24054.8747787
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13379.7675781, 14091.6962891, -13293.1386719, 13997.1679688, -27376.9335938, 27384.8359375
1: -1427.4641113, 1029.4134521, -1418.6293945, 1021.6860352, -2449.1501465, 2448.0429688
2: -2331.5791016, 2673.2089844, -2317.1933594, 2657.4870605, -4989.0664062, 4990.4013672
3: -2703.5146484, 1680.8045654, -2689.7580566, 1669.9271240, -4373.4409180, 4370.5625000
4: -2041.1140137, 2146.9482422, -2029.9649658, 2133.6103516, -4174.7231445, 4176.9130859

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -13401.2724609, 14117.3222656, -13229.3095703, 13922.2041016, -27323.4726562, 27346.6328125
1: -1430.0307617, 1031.5002441, -1411.0484619, 1017.1754150, -2447.2058105, 2442.5488281
2: -2334.3488770, 2677.2775879, -2305.2060547, 2642.8059082, -4977.1547852, 4982.4829102
3: -2706.0461426, 1683.9815674, -2674.5632324, 1661.2543945, -4367.3007812, 4358.5444336
4: -2043.3508301, 2150.1789551, -2019.1561279, 2122.0251465, -4165.3759766, 4169.3349609

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8281530, upper bound: 24054.8281530
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8281530, upper bound: 24054.8432956
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -12848.9599609, 13442.1582031, -12913.0996094, 13562.0625000, -26411.0214844, 26355.2558594
1: -1365.3238525, 987.8851929, -1375.3590088, 992.8787231, -2358.2026367, 2363.2441406
2: -2240.3044434, 2553.1811523, -2250.1772461, 2574.6032715, -4814.9077148, 4803.3579102
3: -2602.3171387, 1608.6602783, -2612.2568359, 1620.0814209, -4222.3984375, 4220.9160156
4: -1965.4321289, 2048.6594238, -1972.1182861, 2066.7473145, -4032.1791992, 4020.7778320

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8692047, upper bound: 24054.8696147
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8692047, upper bound: 24054.8704866
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -13165.8251953, 13878.0019531, -13383.5390625, 14100.0830078, -27265.9082031, 27261.5410156
1: -1405.5627441, 1012.3012085, -1428.2814941, 1029.0416260, -2434.6044922, 2440.5825195
2: -2293.8891602, 2632.8549805, -2331.8327637, 2675.0070801, -4968.8959961, 4964.6875000
3: -2662.0959473, 1654.6337891, -2705.5544434, 1681.4239502, -4343.5195312, 4360.1884766
4: -2009.2031250, 2114.5288086, -2042.3613281, 2148.3947754, -4157.5976562, 4156.8891602

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8701763, upper bound: 24054.8717260
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8701763, upper bound: 24054.8729142
time: 0.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.47 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8584882, upper bound: 24054.8590940
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8607713, upper bound: 24054.8606382
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8705114, upper bound: 24054.8746975
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8748910, upper bound: 24054.8750780
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8578320, upper bound: 24054.8616961
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8458022, upper bound: 24054.8459303
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8730745, upper bound: 24054.8769676
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8774128, upper bound: 24054.8774128
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8776881, upper bound: 24054.8803528
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8776881, upper bound: 24054.8803528
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8784567, upper bound: 24054.8806849
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8784567, upper bound: 24054.8806849
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8782631, upper bound: 24054.8805979
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8782631, upper bound: 24054.8805979
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8784176, upper bound: 24054.8806479
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8784176, upper bound: 24054.8806479
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8802074, upper bound: 24054.8747057
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8773543, upper bound: 24054.8743454
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8802074, upper bound: 24054.8749795
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8773543, upper bound: 24054.8747649
NS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8796703, upper bound: 24054.8747375
NS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8771834, upper bound: 24054.8742466
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8796703, upper bound: 24054.8750131
NS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8771834, upper bound: 24054.8747787
NS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
NS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8281530, upper bound: 24054.8281530
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8281530, upper bound: 24054.8432956
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8692047, upper bound: 24054.8696147
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8692047, upper bound: 24054.8704866
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8701763, upper bound: 24054.8717260
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.47
Output dim: 0, lower bound: -24054.8701763, upper bound: 24054.8729142

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11918.7578125, 12489.2177734, -12082.5214844, 12714.0419922, -24632.8007812, 24571.7382812
1: -1267.6005859, 916.5861816, -1288.0332031, 928.5393677, -2196.1396484, 2204.6193848
2: -2076.4169922, 2374.2045898, -2104.8039551, 2416.1625977, -4492.5795898, 4479.0078125
3: -2412.3801270, 1493.4997559, -2445.2465820, 1516.4528809, -3928.8330078, 3938.7463379
4: -1821.4478760, 1904.3048096, -1845.6228027, 1939.6699219, -3761.1171875, 3749.9274902

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8520814, upper bound: 24054.8480296
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8521218, upper bound: 24054.8479687
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12626.2666016, 13347.4755859, -12642.4902344, 13365.7949219, -25992.0625000, 25989.9648438
1: -1350.4019775, 971.1697998, -1351.8690186, 971.5082397, -2321.9101562, 2323.0388184
2: -2199.4821777, 2533.0732422, -2201.9853516, 2536.7583008, -4736.2402344, 4735.0585938
3: -2553.6940918, 1589.7070312, -2556.4262695, 1590.7169189, -4144.4111328, 4146.1333008
4: -1926.7407227, 2033.3834229, -1928.9143066, 2037.6862793, -3964.4270020, 3962.2973633

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8560509, upper bound: 24054.8512819
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8526287, upper bound: 24054.8503838
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -11992.2031250, 12590.4062500, -11781.5175781, 12296.8408203, -24289.0429688, 24371.9199219
1: -1277.0511475, 922.5465088, -1250.1506348, 906.1028442, -2183.1540527, 2172.6970215
2: -2090.8562012, 2394.2968750, -2054.5659180, 2340.5527344, -4431.4091797, 4448.8627930
3: -2429.1318359, 1504.3000488, -2387.5068359, 1472.5754395, -3901.7072754, 3891.8068848
4: -1833.9467773, 1920.3876953, -1804.1070557, 1876.9453125, -3710.8918457, 3724.4931641

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8705094, upper bound: 24054.8738172
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8705094, upper bound: 24054.8746975
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -12508.9111328, 13188.7441406, -12260.0175781, 12893.3818359, -25402.2929688, 25448.7617188
1: -1335.7993164, 962.2077026, -1306.8206787, 943.2170410, -2279.0163574, 2269.0283203
2: -2180.8649902, 2505.5280762, -2138.0700684, 2452.5395508, -4633.4042969, 4643.5981445
3: -2533.1044922, 1572.5677490, -2485.0246582, 1538.3713379, -4071.4758301, 4057.5917969
4: -1911.8029785, 2010.6451416, -1875.8804932, 1967.7862549, -3879.5886230, 3886.5256348

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8748910, upper bound: 24054.8742963
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8748910, upper bound: 24054.8750780
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -12429.5136719, 13110.5722656, -12372.4726562, 13024.0292969, -25453.5429688, 25483.0449219
1: -1327.8737793, 955.2079468, -1320.5599365, 951.8706055, -2279.7441406, 2275.7678223
2: -2165.7211914, 2491.7473145, -2156.5231934, 2477.6135254, -4643.3349609, 4648.2705078
3: -2515.6145020, 1562.0046387, -2506.4113770, 1553.5449219, -4069.1589355, 4068.4160156
4: -1898.7388916, 2000.7791748, -1891.8199463, 1987.8250732, -3886.5637207, 3892.5981445

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8458022, upper bound: 24054.8459303
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8458022, upper bound: 24054.8459303
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -12672.3017578, 13411.7412109, -13270.6171875, 14028.2724609, -26700.5742188, 26682.3593750
1: -1355.8251953, 973.7415771, -1418.1691895, 1020.3902588, -2376.2153320, 2391.9104004
2: -2206.6169434, 2544.4829102, -2311.4555664, 2659.3703613, -4865.9858398, 4855.9384766
3: -2560.6718750, 1595.0457764, -2679.9873047, 1668.9205322, -4229.5922852, 4275.0332031
4: -1932.0712891, 2044.7630615, -2023.6148682, 2137.3151855, -4069.3862305, 4068.3774414

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8458022, upper bound: 24054.8459303
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8458022, upper bound: 24054.8459303
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -11953.1406250, 12522.7998047, -12082.7333984, 12623.6250000, -24576.7636719, 24605.5273438
1: -1270.6099854, 919.6210327, -1282.0953369, 929.3436279, -2199.9528809, 2201.7158203
2: -2083.4699707, 2383.0095215, -2104.9440918, 2401.2263184, -4484.6962891, 4487.9536133
3: -2419.8935547, 1496.6680908, -2444.1586914, 1510.6695557, -3930.5629883, 3940.8266602
4: -1827.8791504, 1911.8010254, -1846.9056396, 1926.3187256, -3754.1977539, 3758.7060547

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8725297, upper bound: 24054.8725297
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8725297, upper bound: 24054.8769676
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -12422.2919922, 13073.4501953, -12644.7285156, 13335.4179688, -25757.7109375, 25718.1777344
1: -1324.4514160, 955.6524048, -1349.9945068, 972.6901245, -2297.1416016, 2305.6469727
2: -2165.2907715, 2485.1379395, -2203.2358398, 2532.5375977, -4697.8281250, 4688.3730469
3: -2514.2993164, 1559.0327148, -2558.0322266, 1588.8985596, -4103.1977539, 4117.0649414
4: -1898.3940430, 1994.8886719, -1930.9320068, 2033.4117432, -3931.8056641, 3925.8205566

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8769676, upper bound: 24054.8730745
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8769676, upper bound: 24054.8774128
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -12275.8974609, 12852.7968750, -13133.9794922, 13734.5566406, -26010.4531250, 25986.7773438
1: -1304.3310547, 944.3629761, -1394.9830322, 1009.5057373, -2313.8366699, 2339.3459473
2: -2137.4736328, 2442.9379883, -2290.9533691, 2608.4792480, -4745.9521484, 4733.8906250
3: -2482.0854492, 1536.5905762, -2660.9450684, 1643.5782471, -4125.6635742, 4197.5356445
4: -1874.8510742, 1960.0054932, -2010.0198975, 2093.5175781, -3968.3686523, 3970.0251465

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8741131, upper bound: 24054.8794394
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8772723, upper bound: 24054.8798754
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8661788, upper bound: 24054.8756706
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -12275.8974609, 12852.7968750, -13489.4951172, 14215.0976562, -26490.9960938, 26342.2929688
1: -1304.3310547, 944.3629761, -1439.6510010, 1037.0994873, -2341.4304199, 2384.0139160
2: -2137.4736328, 2442.9379883, -2350.8142090, 2696.4018555, -4833.8745117, 4793.7519531
3: -2482.0854492, 1536.5905762, -2727.7048340, 1694.6638184, -4176.7480469, 4264.2954102
4: -1874.8510742, 1960.0054932, -2058.9343262, 2166.1176758, -4040.9687500, 4018.9394531

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8741131, upper bound: 24054.8794394
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8772723, upper bound: 24054.8798754
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8661788, upper bound: 24054.8756706
time: 1.62 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -12972.9052734, 13719.3583984, -13133.9794922, 13734.5566406, -26707.4609375, 26853.3339844
1: -1387.4932861, 997.9098511, -1394.9830322, 1009.5057373, -2396.9987793, 2392.8928223
2: -2258.8027344, 2602.2397461, -2290.9533691, 2608.4792480, -4867.2807617, 4893.1928711
3: -2621.5263672, 1632.8524170, -2660.9450684, 1643.5782471, -4265.1044922, 4293.7973633
4: -1978.2132568, 2090.0517578, -2010.0198975, 2093.5175781, -4071.7309570, 4100.0712891

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8747347, upper bound: 24054.8795599
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8633541, upper bound: 24054.8643352
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8629425, upper bound: 24054.8642948
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -12972.9052734, 13719.3583984, -13489.4951172, 14215.0976562, -27188.0039062, 27208.8535156
1: -1387.4932861, 997.9098511, -1439.6510010, 1037.0994873, -2424.5925293, 2437.5607910
2: -2258.8027344, 2602.2397461, -2350.8142090, 2696.4018555, -4955.2026367, 4953.0537109
3: -2621.5263672, 1632.8524170, -2727.7048340, 1694.6638184, -4316.1894531, 4360.5571289
4: -1978.2132568, 2090.0517578, -2058.9343262, 2166.1176758, -4144.3300781, 4148.9853516

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8747347, upper bound: 24054.8795599
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8633541, upper bound: 24054.8643352
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8629425, upper bound: 24054.8650432
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -12930.6435547, 13584.5292969, -13160.1406250, 13770.8349609, -26701.4785156, 26744.6699219
1: -1376.1004639, 994.3010254, -1398.3680420, 1011.5742188, -2387.6748047, 2392.6689453
2: -2250.5549316, 2575.9577637, -2295.3701172, 2615.0397949, -4865.5942383, 4871.3281250
3: -2607.9326172, 1620.4578857, -2665.9177246, 1647.4692383, -4255.4013672, 4286.3754883
4: -1969.6529541, 2070.1787109, -2013.6761475, 2098.7043457, -4068.3574219, 4083.8547363

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8521077, upper bound: 24054.8516032
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8759831, upper bound: 24054.8804762
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -12930.6435547, 13584.5292969, -13439.8935547, 14159.4677734, -27090.1093750, 27024.4179688
1: -1376.1004639, 994.3010254, -1434.2047119, 1033.3304443, -2409.4309082, 2428.5058594
2: -2250.5549316, 2575.9577637, -2342.4345703, 2686.4238281, -4936.9775391, 4918.3920898
3: -2607.9326172, 1620.4578857, -2718.2841797, 1688.2207031, -4296.1533203, 4338.7421875
4: -1969.6529541, 2070.1787109, -2052.0019531, 2157.7766113, -4127.4291992, 4122.1806641

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8515178, upper bound: 24054.8528261
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8706943, upper bound: 24054.8709405
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -13556.4375000, 14354.1757812, -13160.1406250, 13770.8349609, -27327.2734375, 27514.3164062
1: -1450.1052246, 1042.8200684, -1398.3680420, 1011.5742188, -2461.6794434, 2441.1879883
2: -2359.0209961, 2717.7607422, -2295.3701172, 2615.0397949, -4974.0605469, 5013.1308594
3: -2731.5778809, 1706.1901855, -2665.9177246, 1647.4692383, -4379.0458984, 4372.1074219
4: -2061.8776855, 2186.2861328, -2013.6761475, 2098.7043457, -4160.5820312, 4199.9619141

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8747367, upper bound: 24054.8796703
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8742466, upper bound: 24054.8772368
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -13556.4375000, 14354.1757812, -13439.8935547, 14159.4677734, -27715.9062500, 27794.0683594
1: -1450.1052246, 1042.8200684, -1434.2047119, 1033.3304443, -2483.4355469, 2477.0249023
2: -2359.0209961, 2717.7607422, -2342.4345703, 2686.4238281, -5045.4438477, 5060.1943359
3: -2731.5778809, 1706.1901855, -2718.2841797, 1688.2207031, -4419.7983398, 4424.4741211
4: -2061.8776855, 2186.2861328, -2052.0019531, 2157.7766113, -4219.6542969, 4238.2871094

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8747367, upper bound: 24054.8796703
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8742466, upper bound: 24054.8772368
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13157.4101562, 13761.1992188, -12290.4765625, 12869.6074219, -26027.0175781, 26051.6757812
1: -1397.6160889, 1011.0119629, -1305.8197021, 945.4436646, -2343.0598145, 2316.8315430
2: -2294.5400391, 2613.3325195, -2139.4785156, 2445.8476562, -4740.3867188, 4752.8105469
3: -2664.8310547, 1646.7248535, -2483.7006836, 1538.4375000, -4203.2685547, 4130.4257812
4: -2012.9846191, 2097.3894043, -1876.2922363, 1962.5000000, -3975.4846191, 3973.6816406

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8693950, upper bound: 24054.8482814
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8802949, upper bound: 24054.8738565
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13133.3085938, 13737.4941406, -12918.6230469, 13488.1201172, -26621.4277344, 26656.1152344
1: -1394.8831787, 1009.3571777, -1370.3371582, 992.7127075, -2387.5959473, 2379.6943359
2: -2289.5566406, 2607.7746582, -2249.8566895, 2562.7868652, -4852.3432617, 4857.6313477
3: -2657.9160156, 1643.7835693, -2612.0366211, 1614.5273438, -4272.4423828, 4255.8193359
4: -2007.3215332, 2093.2954102, -1973.1503906, 2056.7138672, -4064.0346680, 4066.4458008

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8668048, upper bound: 24054.8482017
time: 0.81 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8773713, upper bound: 24054.8735442
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13502.2089844, 14228.0546875, -12290.4765625, 12869.6074219, -26371.8164062, 26518.5312500
1: -1440.9241943, 1038.0367432, -1305.8197021, 945.4436646, -2386.3679199, 2343.8564453
2: -2352.5512695, 2698.7463379, -2139.4785156, 2445.8476562, -4798.3974609, 4838.2236328
3: -2729.3291016, 1696.2402344, -2483.7006836, 1538.4375000, -4267.7666016, 4179.9409180
4: -2060.3024902, 2168.0183105, -1876.2922363, 1962.5000000, -4022.8024902, 4044.3103027

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8682606, upper bound: 24054.8642731
time: 0.95 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8595090, upper bound: 24054.8423047
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8790170, upper bound: 24054.8740246
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8802074, upper bound: 24054.8748663
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -13418.4541016, 14135.8964844, -12918.6230469, 13488.1201172, -26906.5703125, 27054.5175781
1: -1431.5723877, 1031.1228027, -1370.3371582, 992.7127075, -2424.2846680, 2401.4599609
2: -2337.5065918, 2680.4091797, -2249.8566895, 2562.7868652, -4900.2919922, 4930.2656250
3: -2711.1931152, 1685.5135498, -2612.0366211, 1614.5273438, -4325.7197266, 4297.5502930
4: -2046.2662354, 2153.4824219, -1973.1503906, 2056.7138672, -4102.9790039, 4126.6323242

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8770373, upper bound: 24054.8738880
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8771767, upper bound: 24054.8746221
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13157.4101562, 13761.1992188, -12979.2539062, 13721.6582031, -26879.0683594, 26740.4531250
1: -1397.6160889, 1011.0119629, -1387.6752930, 998.3715820, -2395.9877930, 2398.6872559
2: -2294.5400391, 2613.3325195, -2259.5629883, 2602.6875000, -4897.2275391, 4872.8955078
3: -2664.8310547, 1646.7248535, -2621.9499512, 1633.2534180, -4298.0844727, 4268.6748047
4: -2012.9846191, 2097.3894043, -1978.7875977, 2090.4519043, -4103.4360352, 4076.1770020

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8199413, upper bound: 24054.7900113
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8795484, upper bound: 24054.8740792
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -13133.3085938, 13737.4941406, -13466.2099609, 14199.1416016, -27332.4492188, 27203.7031250
1: -1394.8831787, 1009.3571777, -1437.5007324, 1034.9819336, -2429.8652344, 2446.8571777
2: -2289.5566406, 2607.7746582, -2344.3933105, 2692.5891113, -4982.1455078, 4952.1679688
3: -2657.9160156, 1643.7835693, -2720.0827637, 1691.8671875, -4349.7832031, 4363.8647461
4: -2007.3215332, 2093.2954102, -2052.6416016, 2162.8027344, -4170.1240234, 4145.9370117

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8672695, upper bound: 24054.8648618
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8665782, upper bound: 24054.8648219
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -13502.2089844, 14228.0546875, -12979.2539062, 13721.6582031, -27223.8632812, 27207.3085938
1: -1440.9241943, 1038.0367432, -1387.6752930, 998.3715820, -2439.2958984, 2425.7119141
2: -2352.5512695, 2698.7463379, -2259.5629883, 2602.6875000, -4955.2377930, 4958.3090820
3: -2729.3291016, 1696.2402344, -2621.9499512, 1633.2534180, -4362.5825195, 4318.1904297
4: -2060.3024902, 2168.0183105, -1978.7875977, 2090.4519043, -4150.7539062, 4146.8056641

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8669847, upper bound: 24054.8651828
time: 0.87 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8637523, upper bound: 24054.8594565
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B2_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8633078, upper bound: 24054.8587202
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -13418.4541016, 14135.8964844, -13466.2099609, 14199.1416016, -27617.5957031, 27602.1054688
1: -1431.5723877, 1031.1228027, -1437.5007324, 1034.9819336, -2466.5539551, 2468.6228027
2: -2337.5065918, 2680.4091797, -2344.3933105, 2692.5891113, -5030.0947266, 5024.8027344
3: -2711.1931152, 1685.5135498, -2720.0827637, 1691.8671875, -4403.0605469, 4405.5957031
4: -2046.2662354, 2153.4824219, -2052.6416016, 2162.8027344, -4209.0688477, 4206.1240234

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8653127, upper bound: 24054.8651727
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8666856, upper bound: 24054.8652614
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -13379.7675781, 14091.6962891, -13320.7626953, 14006.8027344, -27386.5703125, 27412.4589844
1: -1427.4641113, 1029.4134521, -1419.8381348, 1025.4283447, -2452.8925781, 2449.2514648
2: -2331.5791016, 2673.2089844, -2319.5839844, 2660.0642090, -4991.6425781, 4992.7915039
3: -2703.5146484, 1680.8045654, -2691.0756836, 1671.7774658, -4375.2915039, 4371.8803711
4: -2041.1140137, 2146.9482422, -2032.0549316, 2135.4516602, -4176.5644531, 4179.0029297

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -13379.7675781, 14091.6962891, -12910.3037109, 13589.0341797, -26968.8007812, 27002.0000000
1: -1427.4641113, 1029.4134521, -1377.6464844, 992.5071411, -2419.9709473, 2407.0600586
2: -2331.5791016, 2673.2089844, -2250.0593262, 2580.5705566, -4912.1494141, 4923.2675781
3: -2703.5146484, 1680.8045654, -2612.7043457, 1621.8911133, -4325.4052734, 4293.5087891
4: -2041.1140137, 2146.9482422, -1971.7091064, 2071.2409668, -4112.3540039, 4118.6572266

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.7884219, upper bound: 24054.7884219
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -13401.2724609, 14117.3222656, -13238.4658203, 13941.8271484, -27343.0957031, 27355.7890625
1: -1430.0307617, 1031.5002441, -1412.5754395, 1019.2173462, -2449.2473145, 2444.0756836
2: -2334.3488770, 2677.2775879, -2305.6735840, 2645.0048828, -4979.3535156, 4982.9506836
3: -2706.0461426, 1683.9815674, -2673.1455078, 1663.4357910, -4369.4819336, 4357.1269531
4: -2043.3508301, 2150.1789551, -2018.4904785, 2123.6950684, -4167.0458984, 4168.6684570

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8256074, upper bound: 24054.8256074
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8266156, upper bound: 24054.8266156
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -13401.2724609, 14117.3222656, -13120.2558594, 13819.8203125, -27221.0937500, 27237.5781250
1: -1430.0307617, 1031.5002441, -1400.3261719, 1008.8643188, -2438.8942871, 2431.8264160
2: -2334.3488770, 2677.2775879, -2285.7294922, 2623.1230469, -4957.4716797, 4963.0053711
3: -2706.0461426, 1683.9815674, -2652.5065918, 1648.3321533, -4354.3779297, 4336.4882812
4: -2043.3508301, 2150.1789551, -2002.1207275, 2106.1586914, -4149.5087891, 4152.2988281

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8256074, upper bound: 24054.8267700
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8266156, upper bound: 24054.8396802
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -12848.9599609, 13442.1582031, -12976.0078125, 13570.5478516, -26419.5058594, 26418.1660156
1: -1365.3238525, 987.8851929, -1378.4681396, 997.6238403, -2362.9477539, 2366.3532715
2: -2240.3044434, 2553.1811523, -2262.5383301, 2577.6340332, -4817.9384766, 4815.7197266
3: -2602.3171387, 1608.6602783, -2627.9384766, 1624.2039795, -4226.5200195, 4236.5986328
4: -1965.4321289, 2048.6594238, -1985.0031738, 2068.2856445, -4033.7177734, 4033.6625977

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8497928, upper bound: 24054.8498035
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8516146, upper bound: 24054.8516088
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -12848.9599609, 13442.1582031, -13293.6289062, 14006.5849609, -26855.5410156, 26735.7851562
1: -1365.3238525, 987.8851929, -1418.7546387, 1022.1390381, -2387.4628906, 2406.6398926
2: -2240.3044434, 2553.1811523, -2316.1557617, 2657.3642578, -4897.6689453, 4869.3369141
3: -2602.3171387, 1608.6602783, -2687.7099609, 1670.2205811, -4272.5375977, 4296.3696289
4: -1965.4321289, 2048.6594238, -2028.7703857, 2134.2282715, -4099.6601562, 4077.4296875

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8497928, upper bound: 24054.8542650
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -24054.8516146, upper bound: 24054.8557579
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -13165.8251953, 13878.0019531, -12976.0078125, 13570.5478516, -26736.3730469, 26854.0097656
1: -1405.5627441, 1012.3012085, -1378.4681396, 997.6238403, -2403.1862793, 2390.7692871
2: -2293.8891602, 2632.8549805, -2262.5383301, 2577.6340332, -4871.5224609, 4895.3935547
3: -2662.0959473, 1654.6337891, -2627.9384766, 1624.2039795, -4286.2988281, 4282.5722656
4: -2009.2031250, 2114.5288086, -1985.0031738, 2068.2856445, -4077.4887695, 4099.5322266

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39

Time for candidate selection: 0.23 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.88 + 415.14 = 420.01 seconds
