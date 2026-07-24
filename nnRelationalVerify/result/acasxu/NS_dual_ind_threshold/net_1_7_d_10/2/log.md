## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 8.0073e-05


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0203888, -0.0202527, -0.0203888, -0.0202527, -0.0001361, 0.0001361)
1: (-0.0190220, -0.0186537, -0.0190220, -0.0186537, -0.0003682, 0.0003682)
2: (-0.0191668, -0.0187477, -0.0191668, -0.0187477, -0.0004192, 0.0004192)
3: (-0.0184252, -0.0173187, -0.0184252, -0.0173187, -0.0011065, 0.0011065)
4: (-0.0184166, -0.0172694, -0.0184166, -0.0172694, -0.0011472, 0.0011472)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.93 + 0.56 = 1.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.35
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.35
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0203750, -0.0202553, -0.0203888, -0.0202527, -0.0001223, 0.0001335
1: -0.0190149, -0.0186912, -0.0190220, -0.0186537, -0.0003612, 0.0003308
2: -0.0191588, -0.0187903, -0.0191668, -0.0187477, -0.0004112, 0.0003765
3: -0.0184040, -0.0174313, -0.0184252, -0.0173187, -0.0010854, 0.0009939
4: -0.0183947, -0.0173861, -0.0184166, -0.0172694, -0.0011253, 0.0010305

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.13 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.13 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0203933, -0.0202585, -0.0203888, -0.0202527, -0.0001406, 0.0001304
1: -0.0190065, -0.0186415, -0.0190220, -0.0186537, -0.0003527, 0.0003804
2: -0.0191492, -0.0187338, -0.0191668, -0.0187477, -0.0004015, 0.0004331
3: -0.0183786, -0.0172820, -0.0184252, -0.0173187, -0.0010599, 0.0011431
4: -0.0183683, -0.0172314, -0.0184166, -0.0172694, -0.0010989, 0.0011852

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.15 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.23 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.23
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203750, -0.0202553, -0.0203750, -0.0202553, -0.0001197, 0.0001197
1: -0.0190149, -0.0186912, -0.0190149, -0.0186912, -0.0003237, 0.0003237
2: -0.0191588, -0.0187903, -0.0191588, -0.0187903, -0.0003685, 0.0003685
3: -0.0184040, -0.0174313, -0.0184040, -0.0174313, -0.0009728, 0.0009728
4: -0.0183947, -0.0173861, -0.0183947, -0.0173861, -0.0010086, 0.0010086

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000858
time: 0.13 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.14 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203750, -0.0202553, -0.0203933, -0.0202585, -0.0001165, 0.0001380
1: -0.0190149, -0.0186912, -0.0190065, -0.0186415, -0.0003734, 0.0003153
2: -0.0191588, -0.0187903, -0.0191492, -0.0187338, -0.0004250, 0.0003589
3: -0.0184040, -0.0174313, -0.0183786, -0.0172820, -0.0011220, 0.0009473
4: -0.0183947, -0.0173861, -0.0183683, -0.0172314, -0.0011633, 0.0009822

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000858
time: 0.14 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
time: 0.13 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203933, -0.0202585, -0.0203750, -0.0202553, -0.0001380, 0.0001165
1: -0.0190065, -0.0186415, -0.0190149, -0.0186912, -0.0003153, 0.0003734
2: -0.0191492, -0.0187338, -0.0191588, -0.0187903, -0.0003589, 0.0004250
3: -0.0183786, -0.0172820, -0.0184040, -0.0174313, -0.0009473, 0.0011220
4: -0.0183683, -0.0172314, -0.0183947, -0.0173861, -0.0009822, 0.0011633

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000848, upper bound: 0.0000788
time: 0.14 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
time: 0.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203933, -0.0202585, -0.0203933, -0.0202585, -0.0001349, 0.0001349
1: -0.0190065, -0.0186415, -0.0190065, -0.0186415, -0.0003649, 0.0003649
2: -0.0191492, -0.0187338, -0.0191492, -0.0187338, -0.0004154, 0.0004154
3: -0.0183786, -0.0172820, -0.0183786, -0.0172820, -0.0010965, 0.0010965
4: -0.0183683, -0.0172314, -0.0183683, -0.0172314, -0.0011369, 0.0011369

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000848, upper bound: 0.0000788
time: 0.14 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
time: 0.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.66 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000858
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000858
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000848, upper bound: 0.0000788
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000848, upper bound: 0.0000788
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203750, -0.0202553, -0.0001142, 0.0001174
1: -0.0190088, -0.0187061, -0.0190149, -0.0186912, -0.0003176, 0.0003089
2: -0.0191519, -0.0188072, -0.0191588, -0.0187903, -0.0003616, 0.0003516
3: -0.0183857, -0.0174759, -0.0184040, -0.0174313, -0.0009544, 0.0009281
4: -0.0183756, -0.0174324, -0.0183947, -0.0173861, -0.0009895, 0.0009623

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
time: 0.14 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203750, -0.0202553, -0.0001102, 0.0001077
1: -0.0189825, -0.0187167, -0.0190149, -0.0186912, -0.0002913, 0.0002982
2: -0.0191219, -0.0188193, -0.0191588, -0.0187903, -0.0003316, 0.0003395
3: -0.0183066, -0.0175079, -0.0184040, -0.0174313, -0.0008753, 0.0008961
4: -0.0182936, -0.0174655, -0.0183947, -0.0173861, -0.0009075, 0.0009291

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000861
time: 0.14 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000861
time: 0.14 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203933, -0.0202585, -0.0001110, 0.0001358
1: -0.0190088, -0.0187061, -0.0190065, -0.0186415, -0.0003673, 0.0003004
2: -0.0191519, -0.0188072, -0.0191492, -0.0187338, -0.0004181, 0.0003419
3: -0.0183857, -0.0174759, -0.0183786, -0.0172820, -0.0011036, 0.0009027
4: -0.0183756, -0.0174324, -0.0183683, -0.0172314, -0.0011443, 0.0009359

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000841
time: 0.14 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000843
time: 0.14 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203933, -0.0202585, -0.0001071, 0.0001260
1: -0.0189825, -0.0187167, -0.0190065, -0.0186415, -0.0003410, 0.0002898
2: -0.0191219, -0.0188193, -0.0191492, -0.0187338, -0.0003881, 0.0003298
3: -0.0183066, -0.0175079, -0.0183786, -0.0172820, -0.0010245, 0.0008707
4: -0.0182936, -0.0174655, -0.0183683, -0.0172314, -0.0010622, 0.0009027

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000848
time: 0.14 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
time: 0.14 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0203927, -0.0202639, -0.0203750, -0.0202553, -0.0001374, 0.0001111
1: -0.0189917, -0.0186432, -0.0190149, -0.0186912, -0.0003005, 0.0003717
2: -0.0191324, -0.0187357, -0.0191588, -0.0187903, -0.0003421, 0.0004231
3: -0.0183343, -0.0172871, -0.0184040, -0.0174313, -0.0009031, 0.0011169
4: -0.0183224, -0.0172366, -0.0183947, -0.0173861, -0.0009363, 0.0011580

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000788
time: 0.14 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000788
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0203909, -0.0202592, -0.0203750, -0.0202553, -0.0001356, 0.0001158
1: -0.0190045, -0.0186481, -0.0190149, -0.0186912, -0.0003133, 0.0003669
2: -0.0191470, -0.0187412, -0.0191588, -0.0187903, -0.0003566, 0.0004176
3: -0.0183727, -0.0173016, -0.0184040, -0.0174313, -0.0009414, 0.0011024
4: -0.0183622, -0.0172517, -0.0183947, -0.0173861, -0.0009761, 0.0011430

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
time: 0.15 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
time: 0.14 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0203927, -0.0202639, -0.0203933, -0.0202585, -0.0001343, 0.0001295
1: -0.0189917, -0.0186432, -0.0190065, -0.0186415, -0.0003502, 0.0003632
2: -0.0191324, -0.0187357, -0.0191492, -0.0187338, -0.0003986, 0.0004135
3: -0.0183343, -0.0172871, -0.0183786, -0.0172820, -0.0010523, 0.0010915
4: -0.0183224, -0.0172366, -0.0183683, -0.0172314, -0.0010911, 0.0011316

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000788
time: 0.14 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000788
time: 0.14 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0203909, -0.0202592, -0.0203933, -0.0202585, -0.0001325, 0.0001342
1: -0.0190045, -0.0186481, -0.0190065, -0.0186415, -0.0003630, 0.0003584
2: -0.0191470, -0.0187412, -0.0191492, -0.0187338, -0.0004132, 0.0004080
3: -0.0183727, -0.0173016, -0.0183786, -0.0172820, -0.0010906, 0.0010769
4: -0.0183622, -0.0172517, -0.0183683, -0.0172314, -0.0011308, 0.0011166

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000848
time: 0.14 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000848
time: 0.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.25 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000861
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000861
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000841
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000843
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000848
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000788
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000788
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000788
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000788
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000848
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000848

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203695, -0.0202576, -0.0001119, 0.0001119
1: -0.0190088, -0.0187061, -0.0190088, -0.0187061, -0.0003028, 0.0003028
2: -0.0191519, -0.0188072, -0.0191519, -0.0188072, -0.0003446, 0.0003446
3: -0.0183857, -0.0174759, -0.0183857, -0.0174759, -0.0009098, 0.0009098
4: -0.0183756, -0.0174324, -0.0183756, -0.0174324, -0.0009433, 0.0009433

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000843
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203656, -0.0202673, -0.0001022, 0.0001080
1: -0.0190088, -0.0187061, -0.0189825, -0.0187167, -0.0002921, 0.0002764
2: -0.0191519, -0.0188072, -0.0191219, -0.0188193, -0.0003325, 0.0003147
3: -0.0183857, -0.0174759, -0.0183066, -0.0175079, -0.0008778, 0.0008306
4: -0.0183756, -0.0174324, -0.0182936, -0.0174655, -0.0009101, 0.0008612

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000843
time: 0.14 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203695, -0.0202576, -0.0001080, 0.0001022
1: -0.0189825, -0.0187167, -0.0190088, -0.0187061, -0.0002764, 0.0002921
2: -0.0191219, -0.0188193, -0.0191519, -0.0188072, -0.0003147, 0.0003325
3: -0.0183066, -0.0175079, -0.0183857, -0.0174759, -0.0008306, 0.0008778
4: -0.0182936, -0.0174655, -0.0183756, -0.0174324, -0.0008612, 0.0009101

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
time: 0.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203656, -0.0202673, -0.0000982, 0.0000982
1: -0.0189825, -0.0187167, -0.0189825, -0.0187167, -0.0002658, 0.0002658
2: -0.0191219, -0.0188193, -0.0191219, -0.0188193, -0.0003026, 0.0003026
3: -0.0183066, -0.0175079, -0.0183066, -0.0175079, -0.0007987, 0.0007987
4: -0.0182936, -0.0174655, -0.0182936, -0.0174655, -0.0008281, 0.0008281

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
time: 0.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203927, -0.0202639, -0.0001056, 0.0001351
1: -0.0190088, -0.0187061, -0.0189917, -0.0186432, -0.0003656, 0.0002857
2: -0.0191519, -0.0188072, -0.0191324, -0.0187357, -0.0004162, 0.0003252
3: -0.0183857, -0.0174759, -0.0183343, -0.0172871, -0.0010986, 0.0008584
4: -0.0183756, -0.0174324, -0.0183224, -0.0172366, -0.0011390, 0.0008900

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000838
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000841
time: 0.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203909, -0.0202592, -0.0001103, 0.0001333
1: -0.0190088, -0.0187061, -0.0190045, -0.0186481, -0.0003608, 0.0002984
2: -0.0191519, -0.0188072, -0.0191470, -0.0187412, -0.0004107, 0.0003397
3: -0.0183857, -0.0174759, -0.0183727, -0.0173016, -0.0010840, 0.0008968
4: -0.0183756, -0.0174324, -0.0183622, -0.0172517, -0.0011239, 0.0009298

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000838
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000843
time: 0.15 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203927, -0.0202639, -0.0001017, 0.0001254
1: -0.0189825, -0.0187167, -0.0189917, -0.0186432, -0.0003393, 0.0002750
2: -0.0191219, -0.0188193, -0.0191324, -0.0187357, -0.0003862, 0.0003131
3: -0.0183066, -0.0175079, -0.0183343, -0.0172871, -0.0010195, 0.0008265
4: -0.0182936, -0.0174655, -0.0183224, -0.0172366, -0.0010570, 0.0008569

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000841
time: 0.14 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000788, upper bound: 0.0000848
time: 0.14 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203909, -0.0202592, -0.0001064, 0.0001236
1: -0.0189825, -0.0187167, -0.0190045, -0.0186481, -0.0003344, 0.0002878
2: -0.0191219, -0.0188193, -0.0191470, -0.0187412, -0.0003807, 0.0003276
3: -0.0183066, -0.0175079, -0.0183727, -0.0173016, -0.0010049, 0.0008648
4: -0.0182936, -0.0174655, -0.0183622, -0.0172517, -0.0010419, 0.0008966

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000841
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
time: 0.14 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203927, -0.0202639, -0.0203741, -0.0202602, -0.0001325, 0.0001102
1: -0.0189917, -0.0186432, -0.0190017, -0.0186936, -0.0002981, 0.0003585
2: -0.0191324, -0.0187357, -0.0191438, -0.0187931, -0.0003393, 0.0004081
3: -0.0183343, -0.0172871, -0.0183643, -0.0174386, -0.0008958, 0.0010772
4: -0.0183224, -0.0172366, -0.0183535, -0.0173937, -0.0009287, 0.0011169

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203927, -0.0202639, -0.0203726, -0.0202560, -0.0001367, 0.0001087
1: -0.0189917, -0.0186432, -0.0190131, -0.0186978, -0.0002940, 0.0003698
2: -0.0191324, -0.0187357, -0.0191567, -0.0187978, -0.0003346, 0.0004210
3: -0.0183343, -0.0172871, -0.0183984, -0.0174510, -0.0008834, 0.0011113
4: -0.0183224, -0.0172366, -0.0183888, -0.0174065, -0.0009159, 0.0011522

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203909, -0.0202592, -0.0203741, -0.0202602, -0.0001307, 0.0001149
1: -0.0190045, -0.0186481, -0.0190017, -0.0186936, -0.0003109, 0.0003537
2: -0.0191470, -0.0187412, -0.0191438, -0.0187931, -0.0003539, 0.0004026
3: -0.0183727, -0.0173016, -0.0183643, -0.0174386, -0.0009341, 0.0010627
4: -0.0183622, -0.0172517, -0.0183535, -0.0173937, -0.0009685, 0.0011018

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000833, upper bound: 0.0000770
time: 0.14 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000845
time: 0.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203909, -0.0202592, -0.0203726, -0.0202560, -0.0001349, 0.0001134
1: -0.0190045, -0.0186481, -0.0190131, -0.0186978, -0.0003067, 0.0003650
2: -0.0191470, -0.0187412, -0.0191567, -0.0187978, -0.0003492, 0.0004155
3: -0.0183727, -0.0173016, -0.0183984, -0.0174510, -0.0009217, 0.0010968
4: -0.0183622, -0.0172517, -0.0183888, -0.0174065, -0.0009556, 0.0011371

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000833, upper bound: 0.0000770
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000845
time: 0.15 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203909, -0.0202592, -0.0203927, -0.0202639, -0.0001270, 0.0001335
1: -0.0190045, -0.0186481, -0.0189917, -0.0186432, -0.0003613, 0.0003437
2: -0.0191470, -0.0187412, -0.0191324, -0.0187357, -0.0004112, 0.0003912
3: -0.0183727, -0.0173016, -0.0183343, -0.0172871, -0.0010856, 0.0010327
4: -0.0183622, -0.0172517, -0.0183224, -0.0172366, -0.0011255, 0.0010707

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203909, -0.0202592, -0.0203909, -0.0202592, -0.0001318, 0.0001318
1: -0.0190045, -0.0186481, -0.0190045, -0.0186481, -0.0003564, 0.0003564
2: -0.0191470, -0.0187412, -0.0191470, -0.0187412, -0.0004057, 0.0004057
3: -0.0183727, -0.0173016, -0.0183727, -0.0173016, -0.0010710, 0.0010710
4: -0.0183622, -0.0172517, -0.0183622, -0.0172517, -0.0011105, 0.0011105

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.49 + 39.19 = 40.68 seconds
