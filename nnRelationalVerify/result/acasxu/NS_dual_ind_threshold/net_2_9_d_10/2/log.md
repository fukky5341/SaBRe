## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.00067574


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0200870, 0.0212446, 0.0200870, 0.0212446, -0.0011577, 0.0011577)
1: (-0.0206165, -0.0201471, -0.0206165, -0.0201471, -0.0004694, 0.0004694)
2: (0.0188940, 0.0192454, 0.0188940, 0.0192454, -0.0003515, 0.0003515)
3: (-0.0173310, -0.0166863, -0.0173310, -0.0166863, -0.0006447, 0.0006447)
4: (0.0197300, 0.0202528, 0.0197300, 0.0202528, -0.0005228, 0.0005228)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.11 + 0.65 = 3.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0010395, upper bound: 0.0010396

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009600, upper bound: 0.0009583
time: 0.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009655, upper bound: 0.0009656
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.84 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0009600, upper bound: 0.0009583
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0009655, upper bound: 0.0009656

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0202457, 0.0213740, 0.0200970, 0.0212351, -0.0009894, 0.0012770
1: -0.0206503, -0.0202121, -0.0206153, -0.0201551, -0.0004952, 0.0004032
2: 0.0188515, 0.0192165, 0.0188947, 0.0192419, -0.0003903, 0.0003218
3: -0.0174366, -0.0167617, -0.0173303, -0.0166933, -0.0007432, 0.0005686
4: 0.0196567, 0.0202016, 0.0197311, 0.0202473, -0.0005906, 0.0004705

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009583
time: 0.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009584, upper bound: 0.0009584
time: 0.30 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0201430, 0.0212365, 0.0200870, 0.0212446, -0.0011017, 0.0011496
1: -0.0206140, -0.0201493, -0.0206165, -0.0201471, -0.0004669, 0.0004672
2: 0.0188950, 0.0192316, 0.0188940, 0.0192454, -0.0003504, 0.0003377
3: -0.0173298, -0.0167202, -0.0173310, -0.0166863, -0.0006435, 0.0006108
4: 0.0197315, 0.0202282, 0.0197300, 0.0202528, -0.0005212, 0.0004982

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009593
time: 0.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009656
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.69 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009583
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -0.0009584, upper bound: 0.0009584
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009593
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009656

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202457, 0.0213740, 0.0202457, 0.0213740, -0.0011282, 0.0011282
1: -0.0206503, -0.0202121, -0.0206503, -0.0202121, -0.0004382, 0.0004382
2: 0.0188515, 0.0192165, 0.0188515, 0.0192165, -0.0003650, 0.0003650
3: -0.0174366, -0.0167617, -0.0174366, -0.0167617, -0.0006749, 0.0006749
4: 0.0196567, 0.0202016, 0.0196567, 0.0202016, -0.0005449, 0.0005449

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007349
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.30 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202457, 0.0213740, 0.0201430, 0.0212365, -0.0009908, 0.0012310
1: -0.0206503, -0.0202121, -0.0206140, -0.0201493, -0.0005010, 0.0004020
2: 0.0188515, 0.0192165, 0.0188950, 0.0192316, -0.0003801, 0.0003215
3: -0.0174366, -0.0167617, -0.0173298, -0.0167202, -0.0007163, 0.0005681
4: 0.0196567, 0.0202016, 0.0197315, 0.0202282, -0.0005715, 0.0004700

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007350
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.30 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0201430, 0.0212365, 0.0202457, 0.0213740, -0.0012310, 0.0009908
1: -0.0206140, -0.0201493, -0.0206503, -0.0202121, -0.0004020, 0.0005010
2: 0.0188950, 0.0192316, 0.0188515, 0.0192165, -0.0003215, 0.0003801
3: -0.0173298, -0.0167202, -0.0174366, -0.0167617, -0.0005681, 0.0007163
4: 0.0197315, 0.0202282, 0.0196567, 0.0202016, -0.0004700, 0.0005715

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008513, upper bound: 0.0007623
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009258
time: 0.30 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0201430, 0.0212365, 0.0201430, 0.0212365, -0.0010936, 0.0010936
1: -0.0206140, -0.0201493, -0.0206140, -0.0201493, -0.0004647, 0.0004647
2: 0.0188950, 0.0192316, 0.0188950, 0.0192316, -0.0003366, 0.0003366
3: -0.0173298, -0.0167202, -0.0173298, -0.0167202, -0.0006096, 0.0006096
4: 0.0197315, 0.0202282, 0.0197315, 0.0202282, -0.0004966, 0.0004966

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008511, upper bound: 0.0008672
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009250, upper bound: 0.0009258
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.72 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007349
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007350
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0008513, upper bound: 0.0007623
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0009248, upper bound: 0.0009258
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0008511, upper bound: 0.0008672
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.72
Output dim: 0, lower bound: -0.0009250, upper bound: 0.0009258

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202457, 0.0213740, -0.0011211, 0.0011053
1: -0.0206424, -0.0202158, -0.0206503, -0.0202121, -0.0004303, 0.0004345
2: 0.0188597, 0.0192138, 0.0188515, 0.0192165, -0.0003568, 0.0003623
3: -0.0174254, -0.0167668, -0.0174366, -0.0167617, -0.0006637, 0.0006697
4: 0.0196682, 0.0201977, 0.0196567, 0.0202016, -0.0005334, 0.0005410

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0202457, 0.0213740, -0.0011266, 0.0010106
1: -0.0205985, -0.0201775, -0.0206503, -0.0202121, -0.0003864, 0.0004728
2: 0.0188998, 0.0192525, 0.0188515, 0.0192165, -0.0003167, 0.0004010
3: -0.0173663, -0.0167003, -0.0174366, -0.0167617, -0.0006046, 0.0007362
4: 0.0197258, 0.0202510, 0.0196567, 0.0202016, -0.0004758, 0.0005943

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008859
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008859
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0201430, 0.0212365, -0.0009837, 0.0012081
1: -0.0206424, -0.0202158, -0.0206140, -0.0201493, -0.0004930, 0.0003982
2: 0.0188597, 0.0192138, 0.0188950, 0.0192316, -0.0003719, 0.0003188
3: -0.0174254, -0.0167668, -0.0173298, -0.0167202, -0.0007052, 0.0005630
4: 0.0196682, 0.0201977, 0.0197315, 0.0202282, -0.0005600, 0.0004662

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0201430, 0.0212365, -0.0009892, 0.0011134
1: -0.0205985, -0.0201775, -0.0206140, -0.0201493, -0.0004491, 0.0004365
2: 0.0188998, 0.0192525, 0.0188950, 0.0192316, -0.0003318, 0.0003575
3: -0.0173663, -0.0167003, -0.0173298, -0.0167202, -0.0006460, 0.0006295
4: 0.0197258, 0.0202510, 0.0197315, 0.0202282, -0.0005024, 0.0005194

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202457, 0.0213740, -0.0011628, 0.0009586
1: -0.0205810, -0.0201717, -0.0206503, -0.0202121, -0.0003689, 0.0004786
2: 0.0189097, 0.0192050, 0.0188515, 0.0192165, -0.0003067, 0.0003535
3: -0.0173487, -0.0167472, -0.0174366, -0.0167617, -0.0005870, 0.0006893
4: 0.0197520, 0.0201970, 0.0196567, 0.0202016, -0.0004496, 0.0005403

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008303, upper bound: 0.0007463
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007868, upper bound: 0.0007316
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008482, upper bound: 0.0007574
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0202457, 0.0213740, -0.0011520, 0.0009513
1: -0.0206054, -0.0201697, -0.0206503, -0.0202121, -0.0003933, 0.0004806
2: 0.0189083, 0.0191981, 0.0188515, 0.0192165, -0.0003082, 0.0003466
3: -0.0173123, -0.0167838, -0.0174366, -0.0167617, -0.0005506, 0.0006528
4: 0.0197496, 0.0201790, 0.0196567, 0.0202016, -0.0004519, 0.0005223

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008571, upper bound: 0.0008666
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006971, upper bound: 0.0009037
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006888, upper bound: 0.0008685
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0201430, 0.0212365, -0.0010254, 0.0010614
1: -0.0205810, -0.0201717, -0.0206140, -0.0201493, -0.0004316, 0.0004423
2: 0.0189097, 0.0192050, 0.0188950, 0.0192316, -0.0003219, 0.0003100
3: -0.0173487, -0.0167472, -0.0173298, -0.0167202, -0.0006285, 0.0005826
4: 0.0197520, 0.0201970, 0.0197315, 0.0202282, -0.0004762, 0.0004654

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0007928
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0008671
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0201430, 0.0212365, -0.0010146, 0.0010541
1: -0.0206054, -0.0201697, -0.0206140, -0.0201493, -0.0004560, 0.0004444
2: 0.0189083, 0.0191981, 0.0188950, 0.0192316, -0.0003233, 0.0003031
3: -0.0173123, -0.0167838, -0.0173298, -0.0167202, -0.0005921, 0.0005460
4: 0.0197496, 0.0201790, 0.0197315, 0.0202282, -0.0004785, 0.0004474

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0007927
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0009258
time: 0.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.76 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008859
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008859
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0007868, upper bound: 0.0007316
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008482, upper bound: 0.0007574
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0006971, upper bound: 0.0009037
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0006888, upper bound: 0.0008685
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0007928
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0008671
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0007927
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.76
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0009258

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202529, 0.0213510, -0.0010982, 0.0010982
1: -0.0206424, -0.0202158, -0.0206424, -0.0202158, -0.0004266, 0.0004266
2: 0.0188597, 0.0192138, 0.0188597, 0.0192138, -0.0003541, 0.0003541
3: -0.0174254, -0.0167668, -0.0174254, -0.0167668, -0.0006586, 0.0006586
4: 0.0196682, 0.0201977, 0.0196682, 0.0201977, -0.0005295, 0.0005295

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009428, upper bound: 0.0007164
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202474, 0.0212563, -0.0010035, 0.0011036
1: -0.0206424, -0.0202158, -0.0205985, -0.0201775, -0.0004649, 0.0003826
2: 0.0188597, 0.0192138, 0.0188998, 0.0192525, -0.0003928, 0.0003141
3: -0.0174254, -0.0167668, -0.0173663, -0.0167003, -0.0007251, 0.0005994
4: 0.0196682, 0.0201977, 0.0197258, 0.0202510, -0.0005827, 0.0004719

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009428, upper bound: 0.0007165
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0202529, 0.0213510, -0.0011036, 0.0010035
1: -0.0205985, -0.0201775, -0.0206424, -0.0202158, -0.0003826, 0.0004649
2: 0.0188998, 0.0192525, 0.0188597, 0.0192138, -0.0003141, 0.0003928
3: -0.0173663, -0.0167003, -0.0174254, -0.0167668, -0.0005994, 0.0007251
4: 0.0197258, 0.0202510, 0.0196682, 0.0201977, -0.0004719, 0.0005827

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0006815
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008783, upper bound: 0.0008783
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0202474, 0.0212563, -0.0010090, 0.0010090
1: -0.0205985, -0.0201775, -0.0205985, -0.0201775, -0.0004209, 0.0004209
2: 0.0188998, 0.0192525, 0.0188998, 0.0192525, -0.0003527, 0.0003527
3: -0.0173663, -0.0167003, -0.0173663, -0.0167003, -0.0006659, 0.0006659
4: 0.0197258, 0.0202510, 0.0197258, 0.0202510, -0.0005252, 0.0005252

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0006815
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0008783
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0201501, 0.0212124, -0.0009595, 0.0012009
1: -0.0206424, -0.0202158, -0.0206052, -0.0201521, -0.0004903, 0.0003893
2: 0.0188597, 0.0192138, 0.0189029, 0.0192290, -0.0003693, 0.0003109
3: -0.0174254, -0.0167668, -0.0173191, -0.0167232, -0.0007022, 0.0005523
4: 0.0196682, 0.0201977, 0.0197425, 0.0202244, -0.0005561, 0.0004552

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0201518, 0.0211516, -0.0008987, 0.0011992
1: -0.0206424, -0.0202158, -0.0205904, -0.0201290, -0.0005134, 0.0003746
2: 0.0188597, 0.0192138, 0.0189421, 0.0192552, -0.0003955, 0.0002717
3: -0.0174254, -0.0167668, -0.0172646, -0.0166722, -0.0007532, 0.0004978
4: 0.0196682, 0.0201977, 0.0197968, 0.0202637, -0.0005955, 0.0004009

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0201501, 0.0212124, -0.0009650, 0.0011063
1: -0.0205985, -0.0201775, -0.0206052, -0.0201521, -0.0004463, 0.0004276
2: 0.0188998, 0.0192525, 0.0189029, 0.0192290, -0.0003292, 0.0003496
3: -0.0173663, -0.0167003, -0.0173191, -0.0167232, -0.0006431, 0.0006188
4: 0.0197258, 0.0202510, 0.0197425, 0.0202244, -0.0004985, 0.0005084

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0201518, 0.0211516, -0.0009042, 0.0011045
1: -0.0205985, -0.0201775, -0.0205904, -0.0201290, -0.0004695, 0.0004129
2: 0.0188998, 0.0192525, 0.0189421, 0.0192552, -0.0003554, 0.0003104
3: -0.0173663, -0.0167003, -0.0172646, -0.0166722, -0.0006940, 0.0005643
4: 0.0197258, 0.0202510, 0.0197968, 0.0202637, -0.0005379, 0.0004542

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202494, 0.0213860, -0.0011749, 0.0009550
1: -0.0205810, -0.0201717, -0.0206486, -0.0202504, -0.0003305, 0.0004769
2: 0.0189097, 0.0192050, 0.0188510, 0.0191879, -0.0002782, 0.0003540
3: -0.0173487, -0.0167472, -0.0174429, -0.0168151, -0.0005336, 0.0006957
4: 0.0197520, 0.0201970, 0.0196540, 0.0201614, -0.0004095, 0.0005429

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007860, upper bound: 0.0005733
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007860, upper bound: 0.0007317
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202578, 0.0213665, -0.0011553, 0.0009466
1: -0.0205810, -0.0201717, -0.0206488, -0.0202141, -0.0003669, 0.0004771
2: 0.0189097, 0.0192050, 0.0188527, 0.0192146, -0.0003048, 0.0003523
3: -0.0173487, -0.0167472, -0.0174354, -0.0167646, -0.0005841, 0.0006882
4: 0.0197520, 0.0201970, 0.0196583, 0.0201991, -0.0004472, 0.0005387

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008105, upper bound: 0.0005735
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008105, upper bound: 0.0005733
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0202529, 0.0213510, -0.0011290, 0.0009442
1: -0.0206054, -0.0201697, -0.0206424, -0.0202158, -0.0003895, 0.0004727
2: 0.0189083, 0.0191981, 0.0188597, 0.0192138, -0.0003055, 0.0003384
3: -0.0173123, -0.0167838, -0.0174254, -0.0167668, -0.0005454, 0.0006417
4: 0.0197496, 0.0201790, 0.0196682, 0.0201977, -0.0004481, 0.0005108

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006887, upper bound: 0.0008684
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006887, upper bound: 0.0008684
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0202474, 0.0212563, -0.0010343, 0.0009497
1: -0.0206054, -0.0201697, -0.0205985, -0.0201775, -0.0004278, 0.0004288
2: 0.0189083, 0.0191981, 0.0188998, 0.0192525, -0.0003442, 0.0002983
3: -0.0173123, -0.0167838, -0.0173663, -0.0167003, -0.0006119, 0.0005825
4: 0.0197496, 0.0201790, 0.0197258, 0.0202510, -0.0005013, 0.0004532

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006889, upper bound: 0.0008685
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006889, upper bound: 0.0008685
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202111, 0.0212044, -0.0009932, 0.0009932
1: -0.0205810, -0.0201717, -0.0205810, -0.0201717, -0.0004093, 0.0004093
2: 0.0189097, 0.0192050, 0.0189097, 0.0192050, -0.0002953, 0.0002953
3: -0.0173487, -0.0167472, -0.0173487, -0.0167472, -0.0006015, 0.0006015
4: 0.0197520, 0.0201970, 0.0197520, 0.0201970, -0.0004450, 0.0004450

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007825, upper bound: 0.0007193
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008609, upper bound: 0.0008413
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202220, 0.0211970, -0.0009859, 0.0009824
1: -0.0205810, -0.0201717, -0.0206054, -0.0201697, -0.0004113, 0.0004337
2: 0.0189097, 0.0192050, 0.0189083, 0.0191981, -0.0002884, 0.0002967
3: -0.0173487, -0.0167472, -0.0173123, -0.0167838, -0.0005649, 0.0005651
4: 0.0197520, 0.0201970, 0.0197496, 0.0201790, -0.0004270, 0.0004473

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007825, upper bound: 0.0007194
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008609, upper bound: 0.0008626
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0202111, 0.0212044, -0.0009824, 0.0009859
1: -0.0206054, -0.0201697, -0.0205810, -0.0201717, -0.0004337, 0.0004113
2: 0.0189083, 0.0191981, 0.0189097, 0.0192050, -0.0002967, 0.0002884
3: -0.0173123, -0.0167838, -0.0173487, -0.0167472, -0.0005651, 0.0005649
4: 0.0197496, 0.0201790, 0.0197520, 0.0201970, -0.0004473, 0.0004270

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0006172
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008590, upper bound: 0.0007873
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0202220, 0.0211970, -0.0009751, 0.0009751
1: -0.0206054, -0.0201697, -0.0206054, -0.0201697, -0.0004357, 0.0004357
2: 0.0189083, 0.0191981, 0.0189083, 0.0191981, -0.0002898, 0.0002898
3: -0.0173123, -0.0167838, -0.0173123, -0.0167838, -0.0005285, 0.0005285
4: 0.0197496, 0.0201790, 0.0197496, 0.0201790, -0.0004293, 0.0004293

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0006522
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008590, upper bound: 0.0008072
time: 0.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.00 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0009428, upper bound: 0.0007164
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0009428, upper bound: 0.0007165
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0006815
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008783, upper bound: 0.0008783
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0006815
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0008783
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0007860, upper bound: 0.0005733
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0007860, upper bound: 0.0007317
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008105, upper bound: 0.0005735
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008105, upper bound: 0.0005733
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0006887, upper bound: 0.0008684
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0006887, upper bound: 0.0008684
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0006889, upper bound: 0.0008685
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0006889, upper bound: 0.0008685
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0007825, upper bound: 0.0007193
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008609, upper bound: 0.0008413
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0007825, upper bound: 0.0007194
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008609, upper bound: 0.0008626
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0006172
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008590, upper bound: 0.0007873
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0006522
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.00
Output dim: 0, lower bound: -0.0008590, upper bound: 0.0008072

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202529, 0.0213510, -0.0010946, 0.0011119
1: -0.0206408, -0.0202541, -0.0206424, -0.0202158, -0.0004250, 0.0003883
2: 0.0188591, 0.0191857, 0.0188597, 0.0192138, -0.0003547, 0.0003260
3: -0.0174319, -0.0168199, -0.0174254, -0.0167668, -0.0006651, 0.0006056
4: 0.0196653, 0.0201577, 0.0196682, 0.0201977, -0.0005324, 0.0004895

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0007303
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0007302
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202529, 0.0213510, -0.0010860, 0.0010907
1: -0.0206409, -0.0202179, -0.0206424, -0.0202158, -0.0004251, 0.0004245
2: 0.0188609, 0.0192120, 0.0188597, 0.0192138, -0.0003529, 0.0003522
3: -0.0174243, -0.0167697, -0.0174254, -0.0167668, -0.0006574, 0.0006558
4: 0.0196698, 0.0201953, 0.0196682, 0.0201977, -0.0005279, 0.0005271

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009007
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009006
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202474, 0.0212563, -0.0009999, 0.0011174
1: -0.0206408, -0.0202541, -0.0205985, -0.0201775, -0.0004633, 0.0003444
2: 0.0188591, 0.0191857, 0.0188998, 0.0192525, -0.0003934, 0.0002860
3: -0.0174319, -0.0168199, -0.0173663, -0.0167003, -0.0007316, 0.0005464
4: 0.0196653, 0.0201577, 0.0197258, 0.0202510, -0.0005856, 0.0004319

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0007164
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0007163
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202474, 0.0212563, -0.0009913, 0.0010962
1: -0.0206409, -0.0202179, -0.0205985, -0.0201775, -0.0004634, 0.0003806
2: 0.0188609, 0.0192120, 0.0188998, 0.0192525, -0.0003916, 0.0003122
3: -0.0174243, -0.0167697, -0.0173663, -0.0167003, -0.0007239, 0.0005966
4: 0.0196698, 0.0201953, 0.0197258, 0.0202510, -0.0005812, 0.0004695

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008867
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008867
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202529, 0.0213510, -0.0011002, 0.0010320
1: -0.0205989, -0.0202154, -0.0206424, -0.0202158, -0.0003830, 0.0004270
2: 0.0188975, 0.0192236, 0.0188597, 0.0192138, -0.0003164, 0.0003639
3: -0.0173748, -0.0167530, -0.0174254, -0.0167668, -0.0006080, 0.0006725
4: 0.0197205, 0.0202114, 0.0196682, 0.0201977, -0.0004772, 0.0005432

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0006955
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0006954
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202529, 0.0213510, -0.0010907, 0.0009983
1: -0.0205974, -0.0201791, -0.0206424, -0.0202158, -0.0003816, 0.0004633
2: 0.0189011, 0.0192514, 0.0188597, 0.0192138, -0.0003127, 0.0003916
3: -0.0173648, -0.0167024, -0.0174254, -0.0167668, -0.0005980, 0.0007230
4: 0.0197275, 0.0202494, 0.0196682, 0.0201977, -0.0004702, 0.0005812

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0008923
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0008922
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202474, 0.0212563, -0.0010055, 0.0010375
1: -0.0205989, -0.0202154, -0.0205985, -0.0201775, -0.0004213, 0.0003831
2: 0.0188975, 0.0192236, 0.0188998, 0.0192525, -0.0003551, 0.0003238
3: -0.0173748, -0.0167530, -0.0173663, -0.0167003, -0.0006745, 0.0006133
4: 0.0197205, 0.0202114, 0.0197258, 0.0202510, -0.0005305, 0.0004856

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006816
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202474, 0.0212563, -0.0009960, 0.0010038
1: -0.0205974, -0.0201791, -0.0205985, -0.0201775, -0.0004199, 0.0004194
2: 0.0189011, 0.0192514, 0.0188998, 0.0192525, -0.0003514, 0.0003516
3: -0.0173648, -0.0167024, -0.0173663, -0.0167003, -0.0006645, 0.0006639
4: 0.0197275, 0.0202494, 0.0197258, 0.0202510, -0.0005235, 0.0005236

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008784
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008783
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202172, 0.0211979, 0.0202494, 0.0213860, -0.0011688, 0.0009486
1: -0.0205787, -0.0201691, -0.0206486, -0.0202504, -0.0003282, 0.0004795
2: 0.0189100, 0.0192084, 0.0188510, 0.0191879, -0.0002779, 0.0003573
3: -0.0173496, -0.0167430, -0.0174429, -0.0168151, -0.0005345, 0.0007000
4: 0.0197526, 0.0202006, 0.0196540, 0.0201614, -0.0004088, 0.0005466

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202175, 0.0211856, 0.0202494, 0.0213860, -0.0011685, 0.0009362
1: -0.0205747, -0.0201729, -0.0206486, -0.0202504, -0.0003242, 0.0004757
2: 0.0189116, 0.0192040, 0.0188510, 0.0191879, -0.0002763, 0.0003529
3: -0.0173449, -0.0167489, -0.0174429, -0.0168151, -0.0005298, 0.0006940
4: 0.0197576, 0.0201955, 0.0196540, 0.0201614, -0.0004038, 0.0005414

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202172, 0.0211979, 0.0202578, 0.0213665, -0.0011493, 0.0009401
1: -0.0205787, -0.0201691, -0.0206488, -0.0202141, -0.0003645, 0.0004797
2: 0.0189100, 0.0192084, 0.0188527, 0.0192146, -0.0003046, 0.0003557
3: -0.0173496, -0.0167430, -0.0174354, -0.0167646, -0.0005850, 0.0006924
4: 0.0197526, 0.0202006, 0.0196583, 0.0201991, -0.0004466, 0.0005423

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202175, 0.0211856, 0.0202578, 0.0213665, -0.0011490, 0.0009278
1: -0.0205747, -0.0201729, -0.0206488, -0.0202141, -0.0003606, 0.0004759
2: 0.0189116, 0.0192040, 0.0188527, 0.0192146, -0.0003030, 0.0003513
3: -0.0173449, -0.0167489, -0.0174354, -0.0167646, -0.0005803, 0.0006865
4: 0.0197576, 0.0201955, 0.0196583, 0.0201991, -0.0004415, 0.0005372

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202287, 0.0211764, 0.0202529, 0.0213510, -0.0011223, 0.0009236
1: -0.0205972, -0.0201713, -0.0206424, -0.0202158, -0.0003813, 0.0004711
2: 0.0189161, 0.0191959, 0.0188597, 0.0192138, -0.0002977, 0.0003362
3: -0.0173016, -0.0167878, -0.0174254, -0.0167668, -0.0005348, 0.0006376
4: 0.0197605, 0.0201758, 0.0196682, 0.0201977, -0.0004372, 0.0005076

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202395, 0.0210979, 0.0202529, 0.0213510, -0.0011115, 0.0008451
1: -0.0205799, -0.0201569, -0.0206424, -0.0202158, -0.0003641, 0.0004855
2: 0.0189606, 0.0192180, 0.0188597, 0.0192138, -0.0002533, 0.0003582
3: -0.0172516, -0.0167349, -0.0174254, -0.0167668, -0.0004847, 0.0006905
4: 0.0198225, 0.0202102, 0.0196682, 0.0201977, -0.0003752, 0.0005420

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202287, 0.0211764, 0.0202474, 0.0212563, -0.0010276, 0.0009290
1: -0.0205972, -0.0201713, -0.0205985, -0.0201775, -0.0004196, 0.0004272
2: 0.0189161, 0.0191959, 0.0188998, 0.0192525, -0.0003364, 0.0002961
3: -0.0173016, -0.0167878, -0.0173663, -0.0167003, -0.0006013, 0.0005785
4: 0.0197605, 0.0201758, 0.0197258, 0.0202510, -0.0004905, 0.0004500

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202395, 0.0210979, 0.0202474, 0.0212563, -0.0010168, 0.0008505
1: -0.0205799, -0.0201569, -0.0205985, -0.0201775, -0.0004024, 0.0004416
2: 0.0189606, 0.0192180, 0.0188998, 0.0192525, -0.0002919, 0.0003182
3: -0.0172516, -0.0167349, -0.0173663, -0.0167003, -0.0005512, 0.0006313
4: 0.0198225, 0.0202102, 0.0197258, 0.0202510, -0.0004285, 0.0004844

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202646, 0.0211898, 0.0202111, 0.0212044, -0.0009398, 0.0009787
1: -0.0205192, -0.0201732, -0.0205810, -0.0201717, -0.0003475, 0.0004078
2: 0.0189286, 0.0192075, 0.0189097, 0.0192050, -0.0002764, 0.0002977
3: -0.0173234, -0.0167475, -0.0173487, -0.0167472, -0.0005761, 0.0006012
4: 0.0197802, 0.0201957, 0.0197520, 0.0201970, -0.0004168, 0.0004437

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007133, upper bound: 0.0005796
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007134, upper bound: 0.0007739
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202178, 0.0212029, 0.0202111, 0.0212044, -0.0009866, 0.0009918
1: -0.0205805, -0.0201719, -0.0205810, -0.0201717, -0.0004088, 0.0004091
2: 0.0189114, 0.0192044, 0.0189097, 0.0192050, -0.0002936, 0.0002946
3: -0.0173485, -0.0167476, -0.0173487, -0.0167472, -0.0006013, 0.0006011
4: 0.0197522, 0.0201962, 0.0197520, 0.0201970, -0.0004447, 0.0004443

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007815, upper bound: 0.0005801
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007816, upper bound: 0.0008413
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202646, 0.0211898, 0.0202220, 0.0211970, -0.0009325, 0.0009678
1: -0.0205192, -0.0201732, -0.0206054, -0.0201697, -0.0003495, 0.0004322
2: 0.0189286, 0.0192075, 0.0189083, 0.0191981, -0.0002694, 0.0002992
3: -0.0173234, -0.0167475, -0.0173123, -0.0167838, -0.0005396, 0.0005648
4: 0.0197802, 0.0201957, 0.0197496, 0.0201790, -0.0003988, 0.0004461

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006757, upper bound: 0.0006858
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007805, upper bound: 0.0007128
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202178, 0.0212029, 0.0202220, 0.0211970, -0.0009792, 0.0009809
1: -0.0205805, -0.0201719, -0.0206054, -0.0201697, -0.0004109, 0.0004335
2: 0.0189114, 0.0192044, 0.0189083, 0.0191981, -0.0002866, 0.0002961
3: -0.0173485, -0.0167476, -0.0173123, -0.0167838, -0.0005647, 0.0005647
4: 0.0197522, 0.0201962, 0.0197496, 0.0201790, -0.0004267, 0.0004466

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007934, upper bound: 0.0008525
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008592, upper bound: 0.0008562
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202111, 0.0212044, -0.0009733, 0.0010154
1: -0.0206041, -0.0201669, -0.0205810, -0.0201717, -0.0004324, 0.0004141
2: 0.0189091, 0.0191864, 0.0189097, 0.0192050, -0.0002959, 0.0002766
3: -0.0173165, -0.0167978, -0.0173487, -0.0167472, -0.0005692, 0.0005509
4: 0.0197484, 0.0201631, 0.0197520, 0.0201970, -0.0004486, 0.0004112

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202371, 0.0211815, 0.0202111, 0.0212044, -0.0009673, 0.0009703
1: -0.0205994, -0.0201716, -0.0205810, -0.0201717, -0.0004277, 0.0004094
2: 0.0189103, 0.0191949, 0.0189097, 0.0192050, -0.0002947, 0.0002851
3: -0.0173109, -0.0167887, -0.0173487, -0.0167472, -0.0005636, 0.0005600
4: 0.0197526, 0.0201746, 0.0197520, 0.0201970, -0.0004444, 0.0004227

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202220, 0.0211970, -0.0009659, 0.0010046
1: -0.0206041, -0.0201669, -0.0206054, -0.0201697, -0.0004344, 0.0004384
2: 0.0189091, 0.0191864, 0.0189083, 0.0191981, -0.0002890, 0.0002781
3: -0.0173165, -0.0167978, -0.0173123, -0.0167838, -0.0005327, 0.0005145
4: 0.0197484, 0.0201631, 0.0197496, 0.0201790, -0.0004306, 0.0004135

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0006521
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0006523
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202371, 0.0211815, 0.0202220, 0.0211970, -0.0009599, 0.0009595
1: -0.0205994, -0.0201716, -0.0206054, -0.0201697, -0.0004298, 0.0004338
2: 0.0189103, 0.0191949, 0.0189083, 0.0191981, -0.0002878, 0.0002866
3: -0.0173109, -0.0167887, -0.0173123, -0.0167838, -0.0005271, 0.0005236
4: 0.0197526, 0.0201746, 0.0197496, 0.0201790, -0.0004264, 0.0004250

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0007961
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0008072
time: 0.32 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.82 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0007303
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0007302
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009007
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009006
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0007164
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0007163
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008867
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008867
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0006955
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0006954
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0008923
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007163, upper bound: 0.0008922
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006816
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008784
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008783
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007133, upper bound: 0.0005796
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007134, upper bound: 0.0007739
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007815, upper bound: 0.0005801
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007816, upper bound: 0.0008413
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006757, upper bound: 0.0006858
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007805, upper bound: 0.0007128
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0007934, upper bound: 0.0008525
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0008592, upper bound: 0.0008562
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0006521
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0006523
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0007961
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.82
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0008072

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202564, 0.0213648, -0.0011084, 0.0011084
1: -0.0206408, -0.0202541, -0.0206408, -0.0202541, -0.0003867, 0.0003867
2: 0.0188591, 0.0191857, 0.0188591, 0.0191857, -0.0003266, 0.0003266
3: -0.0174319, -0.0168199, -0.0174319, -0.0168199, -0.0006121, 0.0006121
4: 0.0196653, 0.0201577, 0.0196653, 0.0201577, -0.0004924, 0.0004924

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202651, 0.0213436, -0.0010872, 0.0010998
1: -0.0206408, -0.0202541, -0.0206409, -0.0202179, -0.0004229, 0.0003869
2: 0.0188591, 0.0191857, 0.0188609, 0.0192120, -0.0003529, 0.0003248
3: -0.0174319, -0.0168199, -0.0174243, -0.0167697, -0.0006623, 0.0006044
4: 0.0196653, 0.0201577, 0.0196698, 0.0201953, -0.0005300, 0.0004879

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202564, 0.0213648, -0.0010998, 0.0010872
1: -0.0206409, -0.0202179, -0.0206408, -0.0202541, -0.0003869, 0.0004229
2: 0.0188609, 0.0192120, 0.0188591, 0.0191857, -0.0003248, 0.0003529
3: -0.0174243, -0.0167697, -0.0174319, -0.0168199, -0.0006044, 0.0006623
4: 0.0196698, 0.0201953, 0.0196653, 0.0201577, -0.0004879, 0.0005300

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202651, 0.0213436, -0.0010786, 0.0010786
1: -0.0206409, -0.0202179, -0.0206409, -0.0202179, -0.0004230, 0.0004230
2: 0.0188609, 0.0192120, 0.0188609, 0.0192120, -0.0003510, 0.0003510
3: -0.0174243, -0.0167697, -0.0174243, -0.0167697, -0.0006546, 0.0006546
4: 0.0196698, 0.0201953, 0.0196698, 0.0201953, -0.0005255, 0.0005255

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202508, 0.0212848, -0.0010284, 0.0011140
1: -0.0206408, -0.0202541, -0.0205989, -0.0202154, -0.0004254, 0.0003448
2: 0.0188591, 0.0191857, 0.0188975, 0.0192236, -0.0003645, 0.0002883
3: -0.0174319, -0.0168199, -0.0173748, -0.0167530, -0.0006790, 0.0005550
4: 0.0196653, 0.0201577, 0.0197205, 0.0202114, -0.0005461, 0.0004372

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202603, 0.0212512, -0.0009948, 0.0011045
1: -0.0206408, -0.0202541, -0.0205974, -0.0201791, -0.0004617, 0.0003434
2: 0.0188591, 0.0191857, 0.0189011, 0.0192514, -0.0003923, 0.0002846
3: -0.0174319, -0.0168199, -0.0173648, -0.0167024, -0.0007295, 0.0005449
4: 0.0196653, 0.0201577, 0.0197275, 0.0202494, -0.0005841, 0.0004302

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202508, 0.0212848, -0.0010198, 0.0010928
1: -0.0206409, -0.0202179, -0.0205989, -0.0202154, -0.0004255, 0.0003810
2: 0.0188609, 0.0192120, 0.0188975, 0.0192236, -0.0003627, 0.0003145
3: -0.0174243, -0.0167697, -0.0173748, -0.0167530, -0.0006713, 0.0006052
4: 0.0196698, 0.0201953, 0.0197205, 0.0202114, -0.0005416, 0.0004749

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202603, 0.0212512, -0.0009861, 0.0010833
1: -0.0206409, -0.0202179, -0.0205974, -0.0201791, -0.0004618, 0.0003795
2: 0.0188609, 0.0192120, 0.0189011, 0.0192514, -0.0003904, 0.0003108
3: -0.0174243, -0.0167697, -0.0173648, -0.0167024, -0.0007219, 0.0005951
4: 0.0196698, 0.0201953, 0.0197275, 0.0202494, -0.0005796, 0.0004678

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202564, 0.0213648, -0.0011140, 0.0010284
1: -0.0205989, -0.0202154, -0.0206408, -0.0202541, -0.0003448, 0.0004254
2: 0.0188975, 0.0192236, 0.0188591, 0.0191857, -0.0002883, 0.0003645
3: -0.0173748, -0.0167530, -0.0174319, -0.0168199, -0.0005550, 0.0006790
4: 0.0197205, 0.0202114, 0.0196653, 0.0201577, -0.0004372, 0.0005461

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202651, 0.0213436, -0.0010928, 0.0010198
1: -0.0205989, -0.0202154, -0.0206409, -0.0202179, -0.0003810, 0.0004255
2: 0.0188975, 0.0192236, 0.0188609, 0.0192120, -0.0003145, 0.0003627
3: -0.0173748, -0.0167530, -0.0174243, -0.0167697, -0.0006052, 0.0006713
4: 0.0197205, 0.0202114, 0.0196698, 0.0201953, -0.0004749, 0.0005416

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202564, 0.0213648, -0.0011045, 0.0009948
1: -0.0205974, -0.0201791, -0.0206408, -0.0202541, -0.0003434, 0.0004617
2: 0.0189011, 0.0192514, 0.0188591, 0.0191857, -0.0002846, 0.0003923
3: -0.0173648, -0.0167024, -0.0174319, -0.0168199, -0.0005449, 0.0007295
4: 0.0197275, 0.0202494, 0.0196653, 0.0201577, -0.0004302, 0.0005841

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202651, 0.0213436, -0.0010833, 0.0009861
1: -0.0205974, -0.0201791, -0.0206409, -0.0202179, -0.0003795, 0.0004618
2: 0.0189011, 0.0192514, 0.0188609, 0.0192120, -0.0003108, 0.0003904
3: -0.0173648, -0.0167024, -0.0174243, -0.0167697, -0.0005951, 0.0007219
4: 0.0197275, 0.0202494, 0.0196698, 0.0201953, -0.0004678, 0.0005796

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202508, 0.0212848, -0.0010341, 0.0010341
1: -0.0205989, -0.0202154, -0.0205989, -0.0202154, -0.0003835, 0.0003835
2: 0.0188975, 0.0192236, 0.0188975, 0.0192236, -0.0003262, 0.0003262
3: -0.0173748, -0.0167530, -0.0173748, -0.0167530, -0.0006219, 0.0006219
4: 0.0197205, 0.0202114, 0.0197205, 0.0202114, -0.0004909, 0.0004909

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202603, 0.0212512, -0.0010004, 0.0010245
1: -0.0205989, -0.0202154, -0.0205974, -0.0201791, -0.0004198, 0.0003821
2: 0.0188975, 0.0192236, 0.0189011, 0.0192514, -0.0003539, 0.0003225
3: -0.0173748, -0.0167530, -0.0173648, -0.0167024, -0.0006724, 0.0006118
4: 0.0197205, 0.0202114, 0.0197275, 0.0202494, -0.0005290, 0.0004839

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202508, 0.0212848, -0.0010245, 0.0010004
1: -0.0205974, -0.0201791, -0.0205989, -0.0202154, -0.0003821, 0.0004198
2: 0.0189011, 0.0192514, 0.0188975, 0.0192236, -0.0003225, 0.0003539
3: -0.0173648, -0.0167024, -0.0173748, -0.0167530, -0.0006118, 0.0006724
4: 0.0197275, 0.0202494, 0.0197205, 0.0202114, -0.0004839, 0.0005290

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202603, 0.0212512, -0.0009909, 0.0009909
1: -0.0205974, -0.0201791, -0.0205974, -0.0201791, -0.0004184, 0.0004184
2: 0.0189011, 0.0192514, 0.0189011, 0.0192514, -0.0003503, 0.0003503
3: -0.0173648, -0.0167024, -0.0173648, -0.0167024, -0.0006624, 0.0006624
4: 0.0197275, 0.0202494, 0.0197275, 0.0202494, -0.0005219, 0.0005219

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202646, 0.0211898, 0.0202646, 0.0211898, -0.0009252, 0.0009252
1: -0.0205192, -0.0201732, -0.0205192, -0.0201732, -0.0003460, 0.0003460
2: 0.0189286, 0.0192075, 0.0189286, 0.0192075, -0.0002788, 0.0002788
3: -0.0173234, -0.0167475, -0.0173234, -0.0167475, -0.0005759, 0.0005759
4: 0.0197802, 0.0201957, 0.0197802, 0.0201957, -0.0004155, 0.0004155

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202646, 0.0211898, 0.0202178, 0.0212029, -0.0009383, 0.0009720
1: -0.0205192, -0.0201732, -0.0205805, -0.0201719, -0.0003474, 0.0004073
2: 0.0189286, 0.0192075, 0.0189114, 0.0192044, -0.0002757, 0.0002960
3: -0.0173234, -0.0167475, -0.0173485, -0.0167476, -0.0005757, 0.0006010
4: 0.0197802, 0.0201957, 0.0197522, 0.0201962, -0.0004161, 0.0004435

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202178, 0.0212029, 0.0202646, 0.0211898, -0.0009720, 0.0009383
1: -0.0205805, -0.0201719, -0.0205192, -0.0201732, -0.0004073, 0.0003474
2: 0.0189114, 0.0192044, 0.0189286, 0.0192075, -0.0002960, 0.0002757
3: -0.0173485, -0.0167476, -0.0173234, -0.0167475, -0.0006010, 0.0005757
4: 0.0197522, 0.0201962, 0.0197802, 0.0201957, -0.0004435, 0.0004161

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007678, upper bound: 0.0004407
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007779, upper bound: 0.0005793
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202178, 0.0212029, 0.0202178, 0.0212029, -0.0009851, 0.0009851
1: -0.0205805, -0.0201719, -0.0205805, -0.0201719, -0.0004087, 0.0004087
2: 0.0189114, 0.0192044, 0.0189114, 0.0192044, -0.0002929, 0.0002929
3: -0.0173485, -0.0167476, -0.0173485, -0.0167476, -0.0006009, 0.0006009
4: 0.0197522, 0.0201962, 0.0197522, 0.0201962, -0.0004440, 0.0004440

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007678, upper bound: 0.0005998
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007779, upper bound: 0.0007948
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202646, 0.0211898, 0.0202311, 0.0212266, -0.0009620, 0.0009587
1: -0.0205192, -0.0201732, -0.0206041, -0.0201669, -0.0003523, 0.0004309
2: 0.0189286, 0.0192075, 0.0189091, 0.0191864, -0.0002577, 0.0002984
3: -0.0173234, -0.0167475, -0.0173165, -0.0167978, -0.0005255, 0.0005690
4: 0.0197802, 0.0201957, 0.0197484, 0.0201631, -0.0003830, 0.0004473

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202646, 0.0211898, 0.0202371, 0.0211815, -0.0009169, 0.0009527
1: -0.0205192, -0.0201732, -0.0205994, -0.0201716, -0.0003476, 0.0004262
2: 0.0189286, 0.0192075, 0.0189103, 0.0191949, -0.0002662, 0.0002972
3: -0.0173234, -0.0167475, -0.0173109, -0.0167887, -0.0005347, 0.0005634
4: 0.0197802, 0.0201957, 0.0197526, 0.0201746, -0.0003945, 0.0004431

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007318, upper bound: 0.0005108
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202178, 0.0212029, 0.0202311, 0.0212266, -0.0010088, 0.0009718
1: -0.0205805, -0.0201719, -0.0206041, -0.0201669, -0.0004136, 0.0004322
2: 0.0189114, 0.0192044, 0.0189091, 0.0191864, -0.0002749, 0.0002953
3: -0.0173485, -0.0167476, -0.0173165, -0.0167978, -0.0005507, 0.0005688
4: 0.0197522, 0.0201962, 0.0197484, 0.0201631, -0.0004109, 0.0004479

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007927, upper bound: 0.0006238
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007927, upper bound: 0.0006238
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202178, 0.0212029, 0.0202371, 0.0211815, -0.0009636, 0.0009658
1: -0.0205805, -0.0201719, -0.0205994, -0.0201716, -0.0004090, 0.0004276
2: 0.0189114, 0.0192044, 0.0189103, 0.0191949, -0.0002834, 0.0002941
3: -0.0173485, -0.0167476, -0.0173109, -0.0167887, -0.0005598, 0.0005632
4: 0.0197522, 0.0201962, 0.0197526, 0.0201746, -0.0004224, 0.0004437

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008396, upper bound: 0.0006238
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008396, upper bound: 0.0006238
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202311, 0.0212266, -0.0009955, 0.0009955
1: -0.0206041, -0.0201669, -0.0206041, -0.0201669, -0.0004371, 0.0004371
2: 0.0189091, 0.0191864, 0.0189091, 0.0191864, -0.0002772, 0.0002772
3: -0.0173165, -0.0167978, -0.0173165, -0.0167978, -0.0005187, 0.0005187
4: 0.0197484, 0.0201631, 0.0197484, 0.0201631, -0.0004148, 0.0004148

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202371, 0.0211815, -0.0009504, 0.0009895
1: -0.0206041, -0.0201669, -0.0205994, -0.0201716, -0.0004325, 0.0004325
2: 0.0189091, 0.0191864, 0.0189103, 0.0191949, -0.0002857, 0.0002761
3: -0.0173165, -0.0167978, -0.0173109, -0.0167887, -0.0005278, 0.0005130
4: 0.0197484, 0.0201631, 0.0197526, 0.0201746, -0.0004263, 0.0004106

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202371, 0.0211815, 0.0202311, 0.0212266, -0.0009895, 0.0009504
1: -0.0205994, -0.0201716, -0.0206041, -0.0201669, -0.0004325, 0.0004325
2: 0.0189103, 0.0191949, 0.0189091, 0.0191864, -0.0002761, 0.0002857
3: -0.0173109, -0.0167887, -0.0173165, -0.0167978, -0.0005130, 0.0005278
4: 0.0197526, 0.0201746, 0.0197484, 0.0201631, -0.0004106, 0.0004263

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202371, 0.0211815, 0.0202371, 0.0211815, -0.0009444, 0.0009444
1: -0.0205994, -0.0201716, -0.0205994, -0.0201716, -0.0004279, 0.0004279
2: 0.0189103, 0.0191949, 0.0189103, 0.0191949, -0.0002846, 0.0002846
3: -0.0173109, -0.0167887, -0.0173109, -0.0167887, -0.0005222, 0.0005222
4: 0.0197526, 0.0201746, 0.0197526, 0.0201746, -0.0004221, 0.0004221

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.76 + 309.56 = 313.33 seconds
