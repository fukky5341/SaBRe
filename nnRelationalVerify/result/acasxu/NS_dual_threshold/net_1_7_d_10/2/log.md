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
execution time: IAR + RelationalAnalysis = 0.96 + 0.56 = 1.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

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
time: 0.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.38 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0203750, -0.0202553, -0.0203888, -0.0202527, -0.0001223, 0.0001335
1: -0.0190149, -0.0186912, -0.0190220, -0.0186537, -0.0003612, 0.0003308
2: -0.0191588, -0.0187903, -0.0191668, -0.0187477, -0.0004112, 0.0003765
3: -0.0184040, -0.0174313, -0.0184252, -0.0173187, -0.0010854, 0.0009939
4: -0.0183947, -0.0173861, -0.0184166, -0.0172694, -0.0011253, 0.0010305

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000840
time: 0.13 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

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

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760
time: 0.13 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000760
time: 0.13 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 1.26
Output dim: 0, lower bound: -0.0000760, upper bound: 0.0000760

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203750, -0.0202553, -0.0203750, -0.0202553, -0.0001197, 0.0001197
1: -0.0190149, -0.0186912, -0.0190149, -0.0186912, -0.0003237, 0.0003237
2: -0.0191588, -0.0187903, -0.0191588, -0.0187903, -0.0003685, 0.0003685
3: -0.0184040, -0.0174313, -0.0184040, -0.0174313, -0.0009728, 0.0009728
4: -0.0183947, -0.0173861, -0.0183947, -0.0173861, -0.0010086, 0.0010086

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000858
time: 0.15 seconds

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

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

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
time: 0.14 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203933, -0.0202585, -0.0203883, -0.0202579, -0.0001355, 0.0001298
1: -0.0190065, -0.0186415, -0.0190080, -0.0186552, -0.0003513, 0.0003664
2: -0.0191492, -0.0187338, -0.0191509, -0.0187493, -0.0003998, 0.0004171
3: -0.0183786, -0.0172820, -0.0183831, -0.0173231, -0.0010555, 0.0011011
4: -0.0183683, -0.0172314, -0.0183730, -0.0172739, -0.0010943, 0.0011417

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760
time: 0.14 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760
time: 0.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.32 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.32
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000858
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.32
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.32
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000858
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.32
Output dim: 0, lower bound: -0.0000861, upper bound: 0.0000861
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 1.32
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 1.32
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203750, -0.0202553, -0.0001142, 0.0001174
1: -0.0190088, -0.0187061, -0.0190149, -0.0186912, -0.0003176, 0.0003089
2: -0.0191519, -0.0188072, -0.0191588, -0.0187903, -0.0003616, 0.0003516
3: -0.0183857, -0.0174759, -0.0184040, -0.0174313, -0.0009544, 0.0009281
4: -0.0183756, -0.0174324, -0.0183947, -0.0173861, -0.0009895, 0.0009623

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

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

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.08 seconds

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

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000840
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000843
time: 0.15 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203933, -0.0202585, -0.0001071, 0.0001260
1: -0.0189825, -0.0187167, -0.0190065, -0.0186415, -0.0003410, 0.0002898
2: -0.0191219, -0.0188193, -0.0191492, -0.0187338, -0.0003881, 0.0003298
3: -0.0183066, -0.0175079, -0.0183786, -0.0172820, -0.0010245, 0.0008707
4: -0.0182936, -0.0174655, -0.0183683, -0.0172314, -0.0010622, 0.0009027

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000841
time: 0.14 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0203933, -0.0202585, -0.0203745, -0.0202605, -0.0001329, 0.0001160
1: -0.0190065, -0.0186415, -0.0190010, -0.0186927, -0.0003138, 0.0003594
2: -0.0191492, -0.0187338, -0.0191429, -0.0187920, -0.0003572, 0.0004092
3: -0.0183786, -0.0172820, -0.0183621, -0.0174356, -0.0009429, 0.0010801
4: -0.0183683, -0.0172314, -0.0183512, -0.0173906, -0.0009777, 0.0011198

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000756
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000742
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0203933, -0.0202585, -0.0203929, -0.0202635, -0.0001299, 0.0001344
1: -0.0190065, -0.0186415, -0.0189929, -0.0186428, -0.0003637, 0.0003513
2: -0.0191492, -0.0187338, -0.0191337, -0.0187352, -0.0004140, 0.0003999
3: -0.0183786, -0.0172820, -0.0183377, -0.0172857, -0.0010929, 0.0010557
4: -0.0183683, -0.0172314, -0.0183259, -0.0172352, -0.0011331, 0.0010946

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000749
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760
time: 0.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.35 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000858
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000861
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000861
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000840
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000843
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000841
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000849, upper bound: 0.0000849
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000756
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000742
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000749
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.35
Output dim: 0, lower bound: -0.0000840, upper bound: 0.0000760

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203695, -0.0202576, -0.0203695, -0.0202576, -0.0001119, 0.0001119
1: -0.0190088, -0.0187061, -0.0190088, -0.0187061, -0.0003028, 0.0003028
2: -0.0191519, -0.0188072, -0.0191519, -0.0188072, -0.0003446, 0.0003446
3: -0.0183857, -0.0174759, -0.0183857, -0.0174759, -0.0009098, 0.0009098
4: -0.0183756, -0.0174324, -0.0183756, -0.0174324, -0.0009433, 0.0009433

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
time: 0.16 seconds

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

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000858, upper bound: 0.0000762
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

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
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203695, -0.0202576, -0.0001080, 0.0001022
1: -0.0189825, -0.0187167, -0.0190088, -0.0187061, -0.0002764, 0.0002921
2: -0.0191219, -0.0188193, -0.0191519, -0.0188072, -0.0003147, 0.0003325
3: -0.0183066, -0.0175079, -0.0183857, -0.0174759, -0.0008306, 0.0008778
4: -0.0182936, -0.0174655, -0.0183756, -0.0174324, -0.0008612, 0.0009101

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203656, -0.0202673, -0.0000982, 0.0000982
1: -0.0189825, -0.0187167, -0.0189825, -0.0187167, -0.0002658, 0.0002658
2: -0.0191219, -0.0188193, -0.0191219, -0.0188193, -0.0003026, 0.0003026
3: -0.0183066, -0.0175079, -0.0183066, -0.0175079, -0.0007987, 0.0007987
4: -0.0182936, -0.0174655, -0.0182936, -0.0174655, -0.0008281, 0.0008281

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
time: 0.15 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203933, -0.0202585, -0.0001057, 0.0001310
1: -0.0189959, -0.0187206, -0.0190065, -0.0186415, -0.0003543, 0.0002859
2: -0.0191371, -0.0188237, -0.0191492, -0.0187338, -0.0004033, 0.0003254
3: -0.0183467, -0.0175195, -0.0183786, -0.0172820, -0.0010647, 0.0008591
4: -0.0183353, -0.0174776, -0.0183683, -0.0172314, -0.0011039, 0.0008907

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000833
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000838
time: 0.16 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203933, -0.0202585, -0.0001081, 0.0001351
1: -0.0190071, -0.0187140, -0.0190065, -0.0186415, -0.0003656, 0.0002924
2: -0.0191499, -0.0188163, -0.0191492, -0.0187338, -0.0004161, 0.0003329
3: -0.0183805, -0.0174998, -0.0183786, -0.0172820, -0.0010984, 0.0008788
4: -0.0183702, -0.0174572, -0.0183683, -0.0172314, -0.0011389, 0.0009111

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000839
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000840
time: 0.15 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203933, -0.0202585, -0.0001060, 0.0001216
1: -0.0189706, -0.0187196, -0.0190065, -0.0186415, -0.0003291, 0.0002869
2: -0.0191084, -0.0188226, -0.0191492, -0.0187338, -0.0003746, 0.0003266
3: -0.0182709, -0.0175165, -0.0183786, -0.0172820, -0.0009888, 0.0008621
4: -0.0182566, -0.0174745, -0.0183683, -0.0172314, -0.0010252, 0.0008938

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000825
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000838
time: 0.15 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203933, -0.0202585, -0.0001049, 0.0001256
1: -0.0189813, -0.0187226, -0.0190065, -0.0186415, -0.0003397, 0.0002838
2: -0.0191205, -0.0188261, -0.0191492, -0.0187338, -0.0003867, 0.0003231
3: -0.0183029, -0.0175257, -0.0183786, -0.0172820, -0.0010209, 0.0008529
4: -0.0182898, -0.0174840, -0.0183683, -0.0172314, -0.0010584, 0.0008843

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000845
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000845
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0203791, -0.0202669, -0.0203742, -0.0202607, -0.0001184, 0.0001072
1: -0.0189835, -0.0186801, -0.0190004, -0.0186934, -0.0002901, 0.0003203
2: -0.0191231, -0.0187777, -0.0191423, -0.0187928, -0.0003302, 0.0003646
3: -0.0183096, -0.0173979, -0.0183604, -0.0174379, -0.0008717, 0.0009625
4: -0.0182967, -0.0173515, -0.0183495, -0.0173930, -0.0009038, 0.0009980

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000730
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000756
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0203933, -0.0202592, -0.0203745, -0.0202605, -0.0001328, 0.0001153
1: -0.0190046, -0.0186418, -0.0190010, -0.0186927, -0.0003119, 0.0003592
2: -0.0191470, -0.0187341, -0.0191429, -0.0187920, -0.0003551, 0.0004089
3: -0.0183729, -0.0172828, -0.0183621, -0.0174356, -0.0009373, 0.0010793
4: -0.0183624, -0.0172322, -0.0183512, -0.0173906, -0.0009718, 0.0011190

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000835, upper bound: 0.0000742
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000739
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000742
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0203929, -0.0202635, -0.0203929, -0.0202635, -0.0001294, 0.0001294
1: -0.0189929, -0.0186428, -0.0189929, -0.0186428, -0.0003501, 0.0003501
2: -0.0191337, -0.0187352, -0.0191337, -0.0187352, -0.0003985, 0.0003985
3: -0.0183377, -0.0172857, -0.0183377, -0.0172857, -0.0010520, 0.0010520
4: -0.0183259, -0.0172352, -0.0183259, -0.0172352, -0.0010908, 0.0010908

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000730
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000739
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0204914, -0.0202556, -0.0203929, -0.0202635, -0.0002279, 0.0001373
1: -0.0190141, -0.0183764, -0.0189929, -0.0186428, -0.0003713, 0.0006164
2: -0.0191579, -0.0184320, -0.0191337, -0.0187352, -0.0004227, 0.0007017
3: -0.0184015, -0.0164854, -0.0183377, -0.0172857, -0.0011158, 0.0018524
4: -0.0183920, -0.0164054, -0.0183259, -0.0172352, -0.0011569, 0.0019206

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_B2_A2_A1
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
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000756
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000742
time: 0.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.38 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000843
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000843
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000833
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000838
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000839
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000840
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000825
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000838
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000845
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000845, upper bound: 0.0000845
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000730
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000756
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000739
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000742
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000730
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000739
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000756
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.38
Output dim: 0, lower bound: -0.0000837, upper bound: 0.0000742

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203695, -0.0202576, -0.0001065, 0.0001071
1: -0.0189959, -0.0187206, -0.0190088, -0.0187061, -0.0002898, 0.0002883
2: -0.0191371, -0.0188237, -0.0191519, -0.0188072, -0.0003299, 0.0003281
3: -0.0183467, -0.0175195, -0.0183857, -0.0174759, -0.0008708, 0.0008662
4: -0.0183353, -0.0174776, -0.0183756, -0.0174324, -0.0009029, 0.0008980

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
time: 0.16 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203695, -0.0202576, -0.0001090, 0.0001113
1: -0.0190071, -0.0187140, -0.0190088, -0.0187061, -0.0003010, 0.0002948
2: -0.0191499, -0.0188163, -0.0191519, -0.0188072, -0.0003427, 0.0003356
3: -0.0183805, -0.0174998, -0.0183857, -0.0174759, -0.0009046, 0.0008859
4: -0.0183702, -0.0174572, -0.0183756, -0.0174324, -0.0009379, 0.0009185

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000843
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000843
time: 0.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203656, -0.0202673, -0.0000968, 0.0001032
1: -0.0189959, -0.0187206, -0.0189825, -0.0187167, -0.0002792, 0.0002619
2: -0.0191371, -0.0188237, -0.0191219, -0.0188193, -0.0003178, 0.0002982
3: -0.0183467, -0.0175195, -0.0183066, -0.0175079, -0.0008388, 0.0007870
4: -0.0183353, -0.0174776, -0.0182936, -0.0174655, -0.0008697, 0.0008160

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000838
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000838
time: 0.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203656, -0.0202673, -0.0000992, 0.0001073
1: -0.0190071, -0.0187140, -0.0189825, -0.0187167, -0.0002904, 0.0002685
2: -0.0191499, -0.0188163, -0.0191219, -0.0188193, -0.0003306, 0.0003056
3: -0.0183805, -0.0174998, -0.0183066, -0.0175079, -0.0008726, 0.0008067
4: -0.0183702, -0.0174572, -0.0182936, -0.0174655, -0.0009047, 0.0008364

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000843
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000843
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203641, -0.0202624, -0.0001032, 0.0000968
1: -0.0189825, -0.0187167, -0.0189959, -0.0187206, -0.0002619, 0.0002792
2: -0.0191219, -0.0188193, -0.0191371, -0.0188237, -0.0002982, 0.0003178
3: -0.0183066, -0.0175079, -0.0183467, -0.0175195, -0.0007870, 0.0008388
4: -0.0182936, -0.0174655, -0.0183353, -0.0174776, -0.0008160, 0.0008697

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000841
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000849
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0203656, -0.0202673, -0.0203666, -0.0202582, -0.0001073, 0.0000992
1: -0.0189825, -0.0187167, -0.0190071, -0.0187140, -0.0002685, 0.0002904
2: -0.0191219, -0.0188193, -0.0191499, -0.0188163, -0.0003056, 0.0003306
3: -0.0183066, -0.0175079, -0.0183805, -0.0174998, -0.0008067, 0.0008726
4: -0.0182936, -0.0174655, -0.0183702, -0.0174572, -0.0008364, 0.0009047

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
time: 0.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203656, -0.0202673, -0.0000972, 0.0000939
1: -0.0189706, -0.0187196, -0.0189825, -0.0187167, -0.0002539, 0.0002629
2: -0.0191084, -0.0188226, -0.0191219, -0.0188193, -0.0002890, 0.0002993
3: -0.0182709, -0.0175165, -0.0183066, -0.0175079, -0.0007630, 0.0007900
4: -0.0182566, -0.0174745, -0.0182936, -0.0174655, -0.0007911, 0.0008191

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000841
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000841
time: 0.16 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203656, -0.0202673, -0.0000961, 0.0000978
1: -0.0189813, -0.0187226, -0.0189825, -0.0187167, -0.0002646, 0.0002599
2: -0.0191205, -0.0188261, -0.0191219, -0.0188193, -0.0003012, 0.0002958
3: -0.0183029, -0.0175257, -0.0183066, -0.0175079, -0.0007950, 0.0007809
4: -0.0182898, -0.0174840, -0.0182936, -0.0174655, -0.0008243, 0.0008096

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
time: 0.16 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203639, -0.0202626, -0.0203791, -0.0202669, -0.0000969, 0.0001165
1: -0.0189953, -0.0187213, -0.0189835, -0.0186801, -0.0003152, 0.0002622
2: -0.0191365, -0.0188246, -0.0191231, -0.0187777, -0.0003588, 0.0002985
3: -0.0183451, -0.0175217, -0.0183096, -0.0173979, -0.0009472, 0.0007878
4: -0.0183336, -0.0174799, -0.0182967, -0.0173515, -0.0009821, 0.0008168

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203933, -0.0202592, -0.0001050, 0.0001309
1: -0.0189959, -0.0187206, -0.0190046, -0.0186418, -0.0003541, 0.0002840
2: -0.0191371, -0.0188237, -0.0191470, -0.0187341, -0.0004030, 0.0003233
3: -0.0183467, -0.0175195, -0.0183729, -0.0172828, -0.0010639, 0.0008534
4: -0.0183353, -0.0174776, -0.0183624, -0.0172322, -0.0011031, 0.0008848

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
time: 0.14 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
time: 0.15 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203662, -0.0202584, -0.0203791, -0.0202669, -0.0000992, 0.0001207
1: -0.0190066, -0.0187151, -0.0189835, -0.0186801, -0.0003265, 0.0002684
2: -0.0191493, -0.0188175, -0.0191231, -0.0187777, -0.0003716, 0.0003055
3: -0.0183789, -0.0175030, -0.0183096, -0.0173979, -0.0009810, 0.0008066
4: -0.0183686, -0.0174605, -0.0182967, -0.0173515, -0.0010171, 0.0008363

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000770, upper bound: 0.0000837
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203933, -0.0202592, -0.0001074, 0.0001350
1: -0.0190071, -0.0187140, -0.0190046, -0.0186418, -0.0003653, 0.0002905
2: -0.0191499, -0.0188163, -0.0191470, -0.0187341, -0.0004158, 0.0003307
3: -0.0183805, -0.0174998, -0.0183729, -0.0172828, -0.0010977, 0.0008731
4: -0.0183702, -0.0174572, -0.0183624, -0.0172322, -0.0011381, 0.0009052

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000839
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000840
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203643, -0.0202721, -0.0203791, -0.0202669, -0.0000973, 0.0001070
1: -0.0189695, -0.0187202, -0.0189835, -0.0186801, -0.0002894, 0.0002633
2: -0.0191071, -0.0188234, -0.0191231, -0.0187777, -0.0003295, 0.0002997
3: -0.0182676, -0.0175185, -0.0183096, -0.0173979, -0.0008697, 0.0007911
4: -0.0182532, -0.0174765, -0.0182967, -0.0173515, -0.0009017, 0.0008202

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203933, -0.0202592, -0.0001053, 0.0001215
1: -0.0189706, -0.0187196, -0.0190046, -0.0186418, -0.0003288, 0.0002850
2: -0.0191084, -0.0188226, -0.0191470, -0.0187341, -0.0003743, 0.0003244
3: -0.0182709, -0.0175165, -0.0183729, -0.0172828, -0.0009881, 0.0008564
4: -0.0182566, -0.0174745, -0.0183624, -0.0172322, -0.0010244, 0.0008879

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
time: 0.17 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203631, -0.0202680, -0.0203791, -0.0202669, -0.0000962, 0.0001111
1: -0.0189807, -0.0187234, -0.0189835, -0.0186801, -0.0003006, 0.0002601
2: -0.0191199, -0.0188269, -0.0191231, -0.0187777, -0.0003422, 0.0002961
3: -0.0183012, -0.0175279, -0.0183096, -0.0173979, -0.0009032, 0.0007817
4: -0.0182880, -0.0174863, -0.0182967, -0.0173515, -0.0009365, 0.0008105

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203933, -0.0202592, -0.0001042, 0.0001255
1: -0.0189813, -0.0187226, -0.0190046, -0.0186418, -0.0003395, 0.0002819
2: -0.0191205, -0.0188261, -0.0191470, -0.0187341, -0.0003864, 0.0003209
3: -0.0183029, -0.0175257, -0.0183729, -0.0172828, -0.0010201, 0.0008472
4: -0.0182898, -0.0174840, -0.0183624, -0.0172322, -0.0010576, 0.0008784

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000845
time: 0.14 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000845
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0203786, -0.0202721, -0.0203742, -0.0202607, -0.0001179, 0.0001020
1: -0.0189694, -0.0186815, -0.0190004, -0.0186934, -0.0002760, 0.0003190
2: -0.0191070, -0.0187792, -0.0191423, -0.0187928, -0.0003142, 0.0003631
3: -0.0182673, -0.0174020, -0.0183604, -0.0174379, -0.0008294, 0.0009585
4: -0.0182529, -0.0173557, -0.0183495, -0.0173930, -0.0008600, 0.0009937

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0204978, -0.0202664, -0.0203742, -0.0202607, -0.0002371, 0.0001078
1: -0.0189850, -0.0183589, -0.0190004, -0.0186934, -0.0002915, 0.0006415
2: -0.0191247, -0.0184120, -0.0191423, -0.0187928, -0.0003319, 0.0007303
3: -0.0183139, -0.0164327, -0.0183604, -0.0174379, -0.0008760, 0.0019277
4: -0.0183012, -0.0163508, -0.0183495, -0.0173930, -0.0009083, 0.0019987

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0203928, -0.0202642, -0.0203745, -0.0202605, -0.0001323, 0.0001103
1: -0.0189910, -0.0186430, -0.0190010, -0.0186927, -0.0002983, 0.0003580
2: -0.0191316, -0.0187355, -0.0191429, -0.0187920, -0.0003396, 0.0004075
3: -0.0183320, -0.0172865, -0.0183621, -0.0174356, -0.0008964, 0.0010756
4: -0.0183200, -0.0172360, -0.0183512, -0.0173906, -0.0009294, 0.0011152

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0204911, -0.0202565, -0.0203745, -0.0202605, -0.0002307, 0.0001179
1: -0.0190117, -0.0183770, -0.0190010, -0.0186927, -0.0003190, 0.0006240
2: -0.0191551, -0.0184326, -0.0191429, -0.0187920, -0.0003631, 0.0007103
3: -0.0183942, -0.0164870, -0.0183621, -0.0174356, -0.0009586, 0.0018751
4: -0.0183845, -0.0164071, -0.0183512, -0.0173906, -0.0009939, 0.0019441

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0203786, -0.0202721, -0.0203926, -0.0202637, -0.0001149, 0.0001205
1: -0.0189694, -0.0186815, -0.0189923, -0.0186436, -0.0003259, 0.0003108
2: -0.0191070, -0.0187792, -0.0191331, -0.0187361, -0.0003709, 0.0003538
3: -0.0182673, -0.0174020, -0.0183360, -0.0172881, -0.0009792, 0.0009340
4: -0.0182529, -0.0173557, -0.0183241, -0.0172377, -0.0010152, 0.0009684

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000730
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000730
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0203928, -0.0202642, -0.0203929, -0.0202635, -0.0001293, 0.0001287
1: -0.0189910, -0.0186430, -0.0189929, -0.0186428, -0.0003482, 0.0003499
2: -0.0191316, -0.0187355, -0.0191337, -0.0187352, -0.0003964, 0.0003982
3: -0.0183320, -0.0172865, -0.0183377, -0.0172857, -0.0010463, 0.0010513
4: -0.0183200, -0.0172360, -0.0183259, -0.0172352, -0.0010849, 0.0010900

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000759
time: 0.15 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000759
time: 0.18 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0204987, -0.0202660, -0.0203926, -0.0202637, -0.0002350, 0.0001266
1: -0.0189860, -0.0183564, -0.0189923, -0.0186436, -0.0003424, 0.0006359
2: -0.0191259, -0.0184092, -0.0191331, -0.0187361, -0.0003898, 0.0007238
3: -0.0183170, -0.0164253, -0.0183360, -0.0172881, -0.0010289, 0.0019107
4: -0.0183045, -0.0163431, -0.0183241, -0.0172377, -0.0010668, 0.0019810

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742
time: 0.14 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0204911, -0.0202565, -0.0203929, -0.0202635, -0.0002277, 0.0001364
1: -0.0190117, -0.0183770, -0.0189929, -0.0186428, -0.0003689, 0.0006159
2: -0.0191551, -0.0184326, -0.0191337, -0.0187352, -0.0004199, 0.0007011
3: -0.0183942, -0.0164870, -0.0183377, -0.0172857, -0.0011085, 0.0018507
4: -0.0183845, -0.0164071, -0.0183259, -0.0172352, -0.0011493, 0.0019189

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742
time: 0.18 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742
time: 0.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.41 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000838
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000843
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000843
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000838
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000838
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000843
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000843
NS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000841
NS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000849
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000841
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000843, upper bound: 0.0000849
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000841
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000841
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000841, upper bound: 0.0000849
NS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
NS_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
NS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000839
NS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000840
NS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
NS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000838
NS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000845
NS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000785, upper bound: 0.0000845
NS_A2_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000730
NS_A2_B1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000730
NS_A2_B1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000759
NS_A2_B1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000759
NS_A2_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742
NS_A2_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742
NS_A2_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742
NS_A2_B1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 1.41
Output dim: 0, lower bound: -0.0000730, upper bound: 0.0000742

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203641, -0.0202624, -0.0001018, 0.0001018
1: -0.0189959, -0.0187206, -0.0189959, -0.0187206, -0.0002753, 0.0002753
2: -0.0191371, -0.0188237, -0.0191371, -0.0188237, -0.0003134, 0.0003134
3: -0.0183467, -0.0175195, -0.0183467, -0.0175195, -0.0008272, 0.0008272
4: -0.0183353, -0.0174776, -0.0183353, -0.0174776, -0.0008577, 0.0008577

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000833, upper bound: 0.0000743
time: 0.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000838, upper bound: 0.0000841
time: 0.16 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203666, -0.0202582, -0.0001059, 0.0001042
1: -0.0189959, -0.0187206, -0.0190071, -0.0187140, -0.0002818, 0.0002865
2: -0.0191371, -0.0188237, -0.0191499, -0.0188163, -0.0003208, 0.0003262
3: -0.0183467, -0.0175195, -0.0183805, -0.0174998, -0.0008469, 0.0008610
4: -0.0183353, -0.0174776, -0.0183702, -0.0174572, -0.0008781, 0.0008927

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203641, -0.0202624, -0.0001042, 0.0001059
1: -0.0190071, -0.0187140, -0.0189959, -0.0187206, -0.0002865, 0.0002818
2: -0.0191499, -0.0188163, -0.0191371, -0.0188237, -0.0003262, 0.0003208
3: -0.0183805, -0.0174998, -0.0183467, -0.0175195, -0.0008610, 0.0008469
4: -0.0183702, -0.0174572, -0.0183353, -0.0174776, -0.0008927, 0.0008781

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203666, -0.0202582, -0.0001083, 0.0001083
1: -0.0190071, -0.0187140, -0.0190071, -0.0187140, -0.0002931, 0.0002931
2: -0.0191499, -0.0188163, -0.0191499, -0.0188163, -0.0003336, 0.0003336
3: -0.0183805, -0.0174998, -0.0183805, -0.0174998, -0.0008807, 0.0008807
4: -0.0183702, -0.0174572, -0.0183702, -0.0174572, -0.0009131, 0.0009131

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203645, -0.0202717, -0.0000924, 0.0001021
1: -0.0189959, -0.0187206, -0.0189706, -0.0187196, -0.0002763, 0.0002500
2: -0.0191371, -0.0188237, -0.0191084, -0.0188226, -0.0003145, 0.0002846
3: -0.0183467, -0.0175195, -0.0182709, -0.0175165, -0.0008302, 0.0007513
4: -0.0183353, -0.0174776, -0.0182566, -0.0174745, -0.0008608, 0.0007790

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203634, -0.0202678, -0.0000964, 0.0001010
1: -0.0189959, -0.0187206, -0.0189813, -0.0187226, -0.0002732, 0.0002607
2: -0.0191371, -0.0188237, -0.0191205, -0.0188261, -0.0003110, 0.0002968
3: -0.0183467, -0.0175195, -0.0183029, -0.0175257, -0.0008210, 0.0007834
4: -0.0183353, -0.0174776, -0.0182898, -0.0174840, -0.0008513, 0.0008122

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203645, -0.0202717, -0.0000948, 0.0001063
1: -0.0190071, -0.0187140, -0.0189706, -0.0187196, -0.0002875, 0.0002566
2: -0.0191499, -0.0188163, -0.0191084, -0.0188226, -0.0003273, 0.0002921
3: -0.0183805, -0.0174998, -0.0182709, -0.0175165, -0.0008640, 0.0007711
4: -0.0183702, -0.0174572, -0.0182566, -0.0174745, -0.0008958, 0.0007994

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203634, -0.0202678, -0.0000988, 0.0001052
1: -0.0190071, -0.0187140, -0.0189813, -0.0187226, -0.0002845, 0.0002673
2: -0.0191499, -0.0188163, -0.0191205, -0.0188261, -0.0003238, 0.0003042
3: -0.0183805, -0.0174998, -0.0183029, -0.0175257, -0.0008548, 0.0008031
4: -0.0183702, -0.0174572, -0.0182898, -0.0174840, -0.0008863, 0.0008326

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203641, -0.0202624, -0.0001021, 0.0000924
1: -0.0189706, -0.0187196, -0.0189959, -0.0187206, -0.0002500, 0.0002763
2: -0.0191084, -0.0188226, -0.0191371, -0.0188237, -0.0002846, 0.0003145
3: -0.0182709, -0.0175165, -0.0183467, -0.0175195, -0.0007513, 0.0008302
4: -0.0182566, -0.0174745, -0.0183353, -0.0174776, -0.0007790, 0.0008608

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203641, -0.0202624, -0.0001010, 0.0000964
1: -0.0189813, -0.0187226, -0.0189959, -0.0187206, -0.0002607, 0.0002732
2: -0.0191205, -0.0188261, -0.0191371, -0.0188237, -0.0002968, 0.0003110
3: -0.0183029, -0.0175257, -0.0183467, -0.0175195, -0.0007834, 0.0008210
4: -0.0182898, -0.0174840, -0.0183353, -0.0174776, -0.0008122, 0.0008513

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203666, -0.0202582, -0.0001063, 0.0000948
1: -0.0189706, -0.0187196, -0.0190071, -0.0187140, -0.0002566, 0.0002875
2: -0.0191084, -0.0188226, -0.0191499, -0.0188163, -0.0002921, 0.0003273
3: -0.0182709, -0.0175165, -0.0183805, -0.0174998, -0.0007711, 0.0008640
4: -0.0182566, -0.0174745, -0.0183702, -0.0174572, -0.0007994, 0.0008958

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203666, -0.0202582, -0.0001052, 0.0000988
1: -0.0189813, -0.0187226, -0.0190071, -0.0187140, -0.0002673, 0.0002845
2: -0.0191205, -0.0188261, -0.0191499, -0.0188163, -0.0003042, 0.0003238
3: -0.0183029, -0.0175257, -0.0183805, -0.0174998, -0.0008031, 0.0008548
4: -0.0182898, -0.0174840, -0.0183702, -0.0174572, -0.0008326, 0.0008863

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203645, -0.0202717, -0.0000928, 0.0000928
1: -0.0189706, -0.0187196, -0.0189706, -0.0187196, -0.0002510, 0.0002510
2: -0.0191084, -0.0188226, -0.0191084, -0.0188226, -0.0002858, 0.0002858
3: -0.0182709, -0.0175165, -0.0182709, -0.0175165, -0.0007543, 0.0007543
4: -0.0182566, -0.0174745, -0.0182566, -0.0174745, -0.0007821, 0.0007821

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203634, -0.0202678, -0.0000967, 0.0000917
1: -0.0189706, -0.0187196, -0.0189813, -0.0187226, -0.0002480, 0.0002617
2: -0.0191084, -0.0188226, -0.0191205, -0.0188261, -0.0002823, 0.0002979
3: -0.0182709, -0.0175165, -0.0183029, -0.0175257, -0.0007452, 0.0007864
4: -0.0182566, -0.0174745, -0.0182898, -0.0174840, -0.0007726, 0.0008153

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203645, -0.0202717, -0.0000917, 0.0000967
1: -0.0189813, -0.0187226, -0.0189706, -0.0187196, -0.0002617, 0.0002480
2: -0.0191205, -0.0188261, -0.0191084, -0.0188226, -0.0002979, 0.0002823
3: -0.0183029, -0.0175257, -0.0182709, -0.0175165, -0.0007864, 0.0007452
4: -0.0182898, -0.0174840, -0.0182566, -0.0174745, -0.0008153, 0.0007726

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203634, -0.0202678, -0.0000956, 0.0000956
1: -0.0189813, -0.0187226, -0.0189813, -0.0187226, -0.0002586, 0.0002586
2: -0.0191205, -0.0188261, -0.0191205, -0.0188261, -0.0002944, 0.0002944
3: -0.0183029, -0.0175257, -0.0183029, -0.0175257, -0.0007772, 0.0007772
4: -0.0182898, -0.0174840, -0.0182898, -0.0174840, -0.0008058, 0.0008058

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203926, -0.0202646, -0.0000995, 0.0001302
1: -0.0189959, -0.0187206, -0.0189897, -0.0186436, -0.0003523, 0.0002691
2: -0.0191371, -0.0188237, -0.0191301, -0.0187361, -0.0004010, 0.0003064
3: -0.0183467, -0.0175195, -0.0183282, -0.0172881, -0.0010587, 0.0008087
4: -0.0183353, -0.0174776, -0.0183161, -0.0172376, -0.0010976, 0.0008385

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0203641, -0.0202624, -0.0203908, -0.0202599, -0.0001043, 0.0001285
1: -0.0189959, -0.0187206, -0.0190026, -0.0186483, -0.0003475, 0.0002820
2: -0.0191371, -0.0188237, -0.0191448, -0.0187415, -0.0003956, 0.0003211
3: -0.0183467, -0.0175195, -0.0183670, -0.0173024, -0.0010443, 0.0008475
4: -0.0183353, -0.0174776, -0.0183563, -0.0172525, -0.0010828, 0.0008787

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203926, -0.0202646, -0.0001019, 0.0001344
1: -0.0190071, -0.0187140, -0.0189897, -0.0186436, -0.0003635, 0.0002757
2: -0.0191499, -0.0188163, -0.0191301, -0.0187361, -0.0004138, 0.0003138
3: -0.0183805, -0.0174998, -0.0183282, -0.0172881, -0.0010924, 0.0008284
4: -0.0183702, -0.0174572, -0.0183161, -0.0172376, -0.0011326, 0.0008589

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0203666, -0.0202582, -0.0203908, -0.0202599, -0.0001067, 0.0001326
1: -0.0190071, -0.0187140, -0.0190026, -0.0186483, -0.0003588, 0.0002886
2: -0.0191499, -0.0188163, -0.0191448, -0.0187415, -0.0004084, 0.0003285
3: -0.0183805, -0.0174998, -0.0183670, -0.0173024, -0.0010781, 0.0008672
4: -0.0183702, -0.0174572, -0.0183563, -0.0172525, -0.0011178, 0.0008991

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203926, -0.0202646, -0.0000999, 0.0001209
1: -0.0189706, -0.0187196, -0.0189897, -0.0186436, -0.0003271, 0.0002701
2: -0.0191084, -0.0188226, -0.0191301, -0.0187361, -0.0003723, 0.0003075
3: -0.0182709, -0.0175165, -0.0183282, -0.0172881, -0.0009828, 0.0008117
4: -0.0182566, -0.0174745, -0.0183161, -0.0172376, -0.0010190, 0.0008416

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0203645, -0.0202717, -0.0203908, -0.0202599, -0.0001046, 0.0001191
1: -0.0189706, -0.0187196, -0.0190026, -0.0186483, -0.0003223, 0.0002830
2: -0.0191084, -0.0188226, -0.0191448, -0.0187415, -0.0003669, 0.0003222
3: -0.0182709, -0.0175165, -0.0183670, -0.0173024, -0.0009685, 0.0008505
4: -0.0182566, -0.0174745, -0.0183563, -0.0172525, -0.0010041, 0.0008818

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203926, -0.0202646, -0.0000987, 0.0001248
1: -0.0189813, -0.0187226, -0.0189897, -0.0186436, -0.0003377, 0.0002671
2: -0.0191205, -0.0188261, -0.0191301, -0.0187361, -0.0003844, 0.0003040
3: -0.0183029, -0.0175257, -0.0183282, -0.0172881, -0.0010148, 0.0008026
4: -0.0182898, -0.0174840, -0.0183161, -0.0172376, -0.0010522, 0.0008321

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## BFS NS instance: NS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0203634, -0.0202678, -0.0203908, -0.0202599, -0.0001035, 0.0001231
1: -0.0189813, -0.0187226, -0.0190026, -0.0186483, -0.0003330, 0.0002800
2: -0.0191205, -0.0188261, -0.0191448, -0.0187415, -0.0003790, 0.0003187
3: -0.0183029, -0.0175257, -0.0183670, -0.0173024, -0.0010005, 0.0008413
4: -0.0182898, -0.0174840, -0.0183563, -0.0172525, -0.0010373, 0.0008723

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.52 + 107.18 = 108.70 seconds
