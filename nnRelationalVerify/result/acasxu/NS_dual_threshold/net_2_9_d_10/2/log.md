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
execution time: IAR + RelationalAnalysis = 3.02 + 0.66 = 3.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0010395, upper bound: 0.0010396

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009600, upper bound: 0.0009583
time: 0.29 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009655, upper bound: 0.0009656
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0009600, upper bound: 0.0009583
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0009655, upper bound: 0.0009656

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0202457, 0.0213740, 0.0200970, 0.0212351, -0.0009894, 0.0012770
1: -0.0206503, -0.0202121, -0.0206153, -0.0201551, -0.0004952, 0.0004032
2: 0.0188515, 0.0192165, 0.0188947, 0.0192419, -0.0003903, 0.0003218
3: -0.0174366, -0.0167617, -0.0173303, -0.0166933, -0.0007432, 0.0005686
4: 0.0196567, 0.0202016, 0.0197311, 0.0202473, -0.0005906, 0.0004705

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009583
time: 0.29 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009584, upper bound: 0.0009584
time: 0.29 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0201430, 0.0212365, 0.0200870, 0.0212446, -0.0011017, 0.0011496
1: -0.0206140, -0.0201493, -0.0206165, -0.0201471, -0.0004669, 0.0004672
2: 0.0188950, 0.0192316, 0.0188940, 0.0192454, -0.0003504, 0.0003377
3: -0.0173298, -0.0167202, -0.0173310, -0.0166863, -0.0006435, 0.0006108
4: 0.0197315, 0.0202282, 0.0197300, 0.0202528, -0.0005212, 0.0004982

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009593
time: 0.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009656
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.62 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009583
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -0.0009584, upper bound: 0.0009584
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009593
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 0, lower bound: -0.0009583, upper bound: 0.0009656

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202457, 0.0213740, 0.0202457, 0.0213740, -0.0011282, 0.0011282
1: -0.0206503, -0.0202121, -0.0206503, -0.0202121, -0.0004382, 0.0004382
2: 0.0188515, 0.0192165, 0.0188515, 0.0192165, -0.0003650, 0.0003650
3: -0.0174366, -0.0167617, -0.0174366, -0.0167617, -0.0006749, 0.0006749
4: 0.0196567, 0.0202016, 0.0196567, 0.0202016, -0.0005449, 0.0005449

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007349
time: 0.29 seconds

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

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007350
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0201430, 0.0212365, 0.0202457, 0.0213740, -0.0012310, 0.0009908
1: -0.0206140, -0.0201493, -0.0206503, -0.0202121, -0.0004020, 0.0005010
2: 0.0188950, 0.0192316, 0.0188515, 0.0192165, -0.0003215, 0.0003801
3: -0.0173298, -0.0167202, -0.0174366, -0.0167617, -0.0005681, 0.0007163
4: 0.0197315, 0.0202282, 0.0196567, 0.0202016, -0.0004700, 0.0005715

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008108, upper bound: 0.0006174
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009567, upper bound: 0.0009577
time: 0.30 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0201430, 0.0212365, 0.0201430, 0.0212365, -0.0010936, 0.0010936
1: -0.0206140, -0.0201493, -0.0206140, -0.0201493, -0.0004647, 0.0004647
2: 0.0188950, 0.0192316, 0.0188950, 0.0192316, -0.0003366, 0.0003366
3: -0.0173298, -0.0167202, -0.0173298, -0.0167202, -0.0006096, 0.0006096
4: 0.0197315, 0.0202282, 0.0197315, 0.0202282, -0.0004966, 0.0004966

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008511, upper bound: 0.0008672
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009250, upper bound: 0.0009258
time: 0.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.63 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007349
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0009305, upper bound: 0.0007350
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0008108, upper bound: 0.0006174
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0009567, upper bound: 0.0009577
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0008511, upper bound: 0.0008672
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -0.0009250, upper bound: 0.0009258

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202457, 0.0213740, -0.0011211, 0.0011053
1: -0.0206424, -0.0202158, -0.0206503, -0.0202121, -0.0004303, 0.0004345
2: 0.0188597, 0.0192138, 0.0188515, 0.0192165, -0.0003568, 0.0003623
3: -0.0174254, -0.0167668, -0.0174366, -0.0167617, -0.0006637, 0.0006697
4: 0.0196682, 0.0201977, 0.0196567, 0.0202016, -0.0005334, 0.0005410

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0202457, 0.0213740, -0.0011266, 0.0010106
1: -0.0205985, -0.0201775, -0.0206503, -0.0202121, -0.0003864, 0.0004728
2: 0.0188998, 0.0192525, 0.0188515, 0.0192165, -0.0003167, 0.0004010
3: -0.0173663, -0.0167003, -0.0174366, -0.0167617, -0.0006046, 0.0007362
4: 0.0197258, 0.0202510, 0.0196567, 0.0202016, -0.0004758, 0.0005943

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0006816
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008783, upper bound: 0.0008783
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0201430, 0.0212365, -0.0009837, 0.0012081
1: -0.0206424, -0.0202158, -0.0206140, -0.0201493, -0.0004930, 0.0003982
2: 0.0188597, 0.0192138, 0.0188950, 0.0192316, -0.0003719, 0.0003188
3: -0.0174254, -0.0167668, -0.0173298, -0.0167202, -0.0007052, 0.0005630
4: 0.0196682, 0.0201977, 0.0197315, 0.0202282, -0.0005600, 0.0004662

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202474, 0.0212563, 0.0201430, 0.0212365, -0.0009892, 0.0011134
1: -0.0205985, -0.0201775, -0.0206140, -0.0201493, -0.0004491, 0.0004365
2: 0.0188998, 0.0192525, 0.0188950, 0.0192316, -0.0003318, 0.0003575
3: -0.0173663, -0.0167003, -0.0173298, -0.0167202, -0.0006460, 0.0006295
4: 0.0197258, 0.0202510, 0.0197315, 0.0202282, -0.0005024, 0.0005194

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008627, upper bound: 0.0005262
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008626, upper bound: 0.0007231
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202571, 0.0211753, 0.0202457, 0.0213740, -0.0011169, 0.0009296
1: -0.0205465, -0.0201486, -0.0206503, -0.0202121, -0.0003344, 0.0005017
2: 0.0189159, 0.0192576, 0.0188515, 0.0192165, -0.0003006, 0.0004061
3: -0.0173134, -0.0166880, -0.0174366, -0.0167617, -0.0005517, 0.0007486
4: 0.0197606, 0.0202575, 0.0196567, 0.0202016, -0.0004409, 0.0006008

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0201464, 0.0212334, 0.0202457, 0.0213740, -0.0012275, 0.0009877
1: -0.0206135, -0.0201500, -0.0206503, -0.0202121, -0.0004014, 0.0005003
2: 0.0188968, 0.0192308, 0.0188515, 0.0192165, -0.0003197, 0.0003792
3: -0.0173277, -0.0167215, -0.0174366, -0.0167617, -0.0005660, 0.0007150
4: 0.0197336, 0.0202271, 0.0196567, 0.0202016, -0.0004680, 0.0005704

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007261, upper bound: 0.0009290
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007176, upper bound: 0.0008672
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0201430, 0.0212365, -0.0010254, 0.0010614
1: -0.0205810, -0.0201717, -0.0206140, -0.0201493, -0.0004316, 0.0004423
2: 0.0189097, 0.0192050, 0.0188950, 0.0192316, -0.0003219, 0.0003100
3: -0.0173487, -0.0167472, -0.0173298, -0.0167202, -0.0006285, 0.0005826
4: 0.0197520, 0.0201970, 0.0197315, 0.0202282, -0.0004762, 0.0004654

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0007928
time: 0.31 seconds

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

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0007927
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0009258
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.65 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008857, upper bound: 0.0008858
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008782, upper bound: 0.0006816
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008783, upper bound: 0.0008783
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008700, upper bound: 0.0007266
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008627, upper bound: 0.0005262
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008626, upper bound: 0.0007231
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0007261, upper bound: 0.0009290
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0007176, upper bound: 0.0008672
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0007928
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008569, upper bound: 0.0008671
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0007927
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.65
Output dim: 0, lower bound: -0.0008653, upper bound: 0.0009258

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202529, 0.0213510, -0.0010982, 0.0010982
1: -0.0206424, -0.0202158, -0.0206424, -0.0202158, -0.0004266, 0.0004266
2: 0.0188597, 0.0192138, 0.0188597, 0.0192138, -0.0003541, 0.0003541
3: -0.0174254, -0.0167668, -0.0174254, -0.0167668, -0.0006586, 0.0006586
4: 0.0196682, 0.0201977, 0.0196682, 0.0201977, -0.0005295, 0.0005295

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009428, upper bound: 0.0007164
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202474, 0.0212563, -0.0010035, 0.0011036
1: -0.0206424, -0.0202158, -0.0205985, -0.0201775, -0.0004649, 0.0003826
2: 0.0188597, 0.0192138, 0.0188998, 0.0192525, -0.0003928, 0.0003141
3: -0.0174254, -0.0167668, -0.0173663, -0.0167003, -0.0007251, 0.0005994
4: 0.0196682, 0.0201977, 0.0197258, 0.0202510, -0.0005827, 0.0004719

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008868
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202457, 0.0213740, -0.0011232, 0.0010391
1: -0.0205989, -0.0202154, -0.0206503, -0.0202121, -0.0003868, 0.0004349
2: 0.0188975, 0.0192236, 0.0188515, 0.0192165, -0.0003190, 0.0003721
3: -0.0173748, -0.0167530, -0.0174366, -0.0167617, -0.0006131, 0.0006836
4: 0.0197205, 0.0202114, 0.0196567, 0.0202016, -0.0004811, 0.0005547

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006814
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202457, 0.0213740, -0.0011137, 0.0010055
1: -0.0205974, -0.0201791, -0.0206503, -0.0202121, -0.0003854, 0.0004712
2: 0.0189011, 0.0192514, 0.0188515, 0.0192165, -0.0003154, 0.0003999
3: -0.0173648, -0.0167024, -0.0174366, -0.0167617, -0.0006031, 0.0007342
4: 0.0197275, 0.0202494, 0.0196567, 0.0202016, -0.0004741, 0.0005927

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008783
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008783
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0201501, 0.0212124, -0.0009595, 0.0012009
1: -0.0206424, -0.0202158, -0.0206052, -0.0201521, -0.0004903, 0.0003893
2: 0.0188597, 0.0192138, 0.0189029, 0.0192290, -0.0003693, 0.0003109
3: -0.0174254, -0.0167668, -0.0173191, -0.0167232, -0.0007022, 0.0005523
4: 0.0196682, 0.0201977, 0.0197425, 0.0202244, -0.0005561, 0.0004552

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009271, upper bound: 0.0005611
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008765, upper bound: 0.0007314
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0201518, 0.0211516, -0.0008987, 0.0011992
1: -0.0206424, -0.0202158, -0.0205904, -0.0201290, -0.0005134, 0.0003746
2: 0.0188597, 0.0192138, 0.0189421, 0.0192552, -0.0003955, 0.0002717
3: -0.0174254, -0.0167668, -0.0172646, -0.0166722, -0.0007532, 0.0004978
4: 0.0196682, 0.0201977, 0.0197968, 0.0202637, -0.0005955, 0.0004009

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0201430, 0.0212365, -0.0009858, 0.0011419
1: -0.0205989, -0.0202154, -0.0206140, -0.0201493, -0.0004495, 0.0003986
2: 0.0188975, 0.0192236, 0.0188950, 0.0192316, -0.0003342, 0.0003286
3: -0.0173748, -0.0167530, -0.0173298, -0.0167202, -0.0006546, 0.0005768
4: 0.0197205, 0.0202114, 0.0197315, 0.0202282, -0.0005077, 0.0004798

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008626, upper bound: 0.0005262
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008626, upper bound: 0.0005262
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0201430, 0.0212365, -0.0009762, 0.0011082
1: -0.0205974, -0.0201791, -0.0206140, -0.0201493, -0.0004481, 0.0004349
2: 0.0189011, 0.0192514, 0.0188950, 0.0192316, -0.0003305, 0.0003564
3: -0.0173648, -0.0167024, -0.0173298, -0.0167202, -0.0006446, 0.0006274
4: 0.0197275, 0.0202494, 0.0197315, 0.0202282, -0.0005007, 0.0005179

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008627, upper bound: 0.0007231
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008627, upper bound: 0.0007230
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0201464, 0.0212334, 0.0202529, 0.0213510, -0.0012046, 0.0009806
1: -0.0206135, -0.0201500, -0.0206424, -0.0202158, -0.0003976, 0.0004924
2: 0.0188968, 0.0192308, 0.0188597, 0.0192138, -0.0003171, 0.0003710
3: -0.0173277, -0.0167215, -0.0174254, -0.0167668, -0.0005609, 0.0007039
4: 0.0197336, 0.0202271, 0.0196682, 0.0201977, -0.0004641, 0.0005589

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007176, upper bound: 0.0008672
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007176, upper bound: 0.0008671
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0201464, 0.0212334, 0.0202474, 0.0212563, -0.0011099, 0.0009861
1: -0.0206135, -0.0201500, -0.0205985, -0.0201775, -0.0004359, 0.0004484
2: 0.0188968, 0.0192308, 0.0188998, 0.0192525, -0.0003557, 0.0003310
3: -0.0173277, -0.0167215, -0.0173663, -0.0167003, -0.0006274, 0.0006447
4: 0.0197336, 0.0202271, 0.0197258, 0.0202510, -0.0005173, 0.0005013

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005163, upper bound: 0.0008598
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007131, upper bound: 0.0008596
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202111, 0.0212044, -0.0009932, 0.0009932
1: -0.0205810, -0.0201717, -0.0205810, -0.0201717, -0.0004093, 0.0004093
2: 0.0189097, 0.0192050, 0.0189097, 0.0192050, -0.0002953, 0.0002953
3: -0.0173487, -0.0167472, -0.0173487, -0.0167472, -0.0006015, 0.0006015
4: 0.0197520, 0.0201970, 0.0197520, 0.0201970, -0.0004450, 0.0004450

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008442, upper bound: 0.0006263
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008651, upper bound: 0.0008397
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202220, 0.0211970, -0.0009859, 0.0009824
1: -0.0205810, -0.0201717, -0.0206054, -0.0201697, -0.0004113, 0.0004337
2: 0.0189097, 0.0192050, 0.0189083, 0.0191981, -0.0002884, 0.0002967
3: -0.0173487, -0.0167472, -0.0173123, -0.0167838, -0.0005649, 0.0005651
4: 0.0197520, 0.0201970, 0.0197496, 0.0201790, -0.0004270, 0.0004473

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007289, upper bound: 0.0008570
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008651, upper bound: 0.0008608
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0202111, 0.0212044, -0.0009824, 0.0009859
1: -0.0206054, -0.0201697, -0.0205810, -0.0201717, -0.0004337, 0.0004113
2: 0.0189083, 0.0191981, 0.0189097, 0.0192050, -0.0002967, 0.0002884
3: -0.0173123, -0.0167838, -0.0173487, -0.0167472, -0.0005651, 0.0005649
4: 0.0197496, 0.0201790, 0.0197520, 0.0201970, -0.0004473, 0.0004270

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0006172
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008590, upper bound: 0.0007873
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202220, 0.0211970, 0.0202220, 0.0211970, -0.0009751, 0.0009751
1: -0.0206054, -0.0201697, -0.0206054, -0.0201697, -0.0004357, 0.0004357
2: 0.0189083, 0.0191981, 0.0189083, 0.0191981, -0.0002898, 0.0002898
3: -0.0173123, -0.0167838, -0.0173123, -0.0167838, -0.0005285, 0.0005285
4: 0.0197496, 0.0201790, 0.0197496, 0.0201790, -0.0004293, 0.0004293

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

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
- Time for NS candidates: 4.89 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0009428, upper bound: 0.0007164
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008868
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006814
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008783
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0006815, upper bound: 0.0008783
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0009271, upper bound: 0.0005611
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008765, upper bound: 0.0007314
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008626, upper bound: 0.0005262
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008626, upper bound: 0.0005262
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008627, upper bound: 0.0007231
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008627, upper bound: 0.0007230
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0007176, upper bound: 0.0008672
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0007176, upper bound: 0.0008671
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0005163, upper bound: 0.0008598
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0007131, upper bound: 0.0008596
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008442, upper bound: 0.0006263
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008651, upper bound: 0.0008397
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0007289, upper bound: 0.0008570
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008651, upper bound: 0.0008608
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0006172
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008590, upper bound: 0.0007873
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008547, upper bound: 0.0006522
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.89
Output dim: 0, lower bound: -0.0008590, upper bound: 0.0008072

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202529, 0.0213510, -0.0010946, 0.0011119
1: -0.0206408, -0.0202541, -0.0206424, -0.0202158, -0.0004250, 0.0003883
2: 0.0188591, 0.0191857, 0.0188597, 0.0192138, -0.0003547, 0.0003260
3: -0.0174319, -0.0168199, -0.0174254, -0.0167668, -0.0006651, 0.0006056
4: 0.0196653, 0.0201577, 0.0196682, 0.0201977, -0.0005324, 0.0004895

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

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
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202529, 0.0213510, -0.0010860, 0.0010907
1: -0.0206409, -0.0202179, -0.0206424, -0.0202158, -0.0004251, 0.0004245
2: 0.0188609, 0.0192120, 0.0188597, 0.0192138, -0.0003529, 0.0003522
3: -0.0174243, -0.0167697, -0.0174254, -0.0167668, -0.0006574, 0.0006558
4: 0.0196698, 0.0201953, 0.0196682, 0.0201977, -0.0005279, 0.0005271

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009007
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009006
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202508, 0.0212848, -0.0010320, 0.0011002
1: -0.0206424, -0.0202158, -0.0205989, -0.0202154, -0.0004270, 0.0003830
2: 0.0188597, 0.0192138, 0.0188975, 0.0192236, -0.0003639, 0.0003164
3: -0.0174254, -0.0167668, -0.0173748, -0.0167530, -0.0006725, 0.0006080
4: 0.0196682, 0.0201977, 0.0197205, 0.0202114, -0.0005432, 0.0004772

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0007163
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008867
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0202529, 0.0213510, 0.0202603, 0.0212512, -0.0009983, 0.0010907
1: -0.0206424, -0.0202158, -0.0205974, -0.0201791, -0.0004633, 0.0003816
2: 0.0188597, 0.0192138, 0.0189011, 0.0192514, -0.0003916, 0.0003127
3: -0.0174254, -0.0167668, -0.0173648, -0.0167024, -0.0007230, 0.0005980
4: 0.0196682, 0.0201977, 0.0197275, 0.0202494, -0.0005812, 0.0004702

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0007164
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202494, 0.0213860, -0.0011353, 0.0010355
1: -0.0205989, -0.0202154, -0.0206486, -0.0202504, -0.0003484, 0.0004332
2: 0.0188975, 0.0192236, 0.0188510, 0.0191879, -0.0002905, 0.0003726
3: -0.0173748, -0.0167530, -0.0174429, -0.0168151, -0.0005597, 0.0006900
4: 0.0197205, 0.0202114, 0.0196540, 0.0201614, -0.0004410, 0.0005573

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006816
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006816
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202578, 0.0213665, -0.0011157, 0.0010270
1: -0.0205989, -0.0202154, -0.0206488, -0.0202141, -0.0003847, 0.0004334
2: 0.0188975, 0.0192236, 0.0188527, 0.0192146, -0.0003171, 0.0003709
3: -0.0173748, -0.0167530, -0.0174354, -0.0167646, -0.0006103, 0.0006824
4: 0.0197205, 0.0202114, 0.0196583, 0.0201991, -0.0004787, 0.0005531

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202494, 0.0213860, -0.0011257, 0.0010018
1: -0.0205974, -0.0201791, -0.0206486, -0.0202504, -0.0003470, 0.0004695
2: 0.0189011, 0.0192514, 0.0188510, 0.0191879, -0.0002868, 0.0004003
3: -0.0173648, -0.0167024, -0.0174429, -0.0168151, -0.0005497, 0.0007405
4: 0.0197275, 0.0202494, 0.0196540, 0.0201614, -0.0004339, 0.0005954

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202578, 0.0213665, -0.0011062, 0.0009934
1: -0.0205974, -0.0201791, -0.0206488, -0.0202141, -0.0003833, 0.0004697
2: 0.0189011, 0.0192514, 0.0188527, 0.0192146, -0.0003135, 0.0003986
3: -0.0173648, -0.0167024, -0.0174354, -0.0167646, -0.0006002, 0.0007330
4: 0.0197275, 0.0202494, 0.0196583, 0.0201991, -0.0004717, 0.0005911

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0201501, 0.0212124, -0.0009560, 0.0012147
1: -0.0206408, -0.0202541, -0.0206052, -0.0201521, -0.0004887, 0.0003511
2: 0.0188591, 0.0191857, 0.0189029, 0.0192290, -0.0003699, 0.0002828
3: -0.0174319, -0.0168199, -0.0173191, -0.0167232, -0.0007087, 0.0004993
4: 0.0196653, 0.0201577, 0.0197425, 0.0202244, -0.0005590, 0.0004151

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0008462
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0008462
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0201501, 0.0212124, -0.0009473, 0.0011935
1: -0.0206409, -0.0202179, -0.0206052, -0.0201521, -0.0004888, 0.0003873
2: 0.0188609, 0.0192120, 0.0189029, 0.0192290, -0.0003681, 0.0003091
3: -0.0174243, -0.0167697, -0.0173191, -0.0167232, -0.0007011, 0.0005495
4: 0.0196698, 0.0201953, 0.0197425, 0.0202244, -0.0005546, 0.0004528

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007303, upper bound: 0.0009387
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007303, upper bound: 0.0009388
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0201501, 0.0212124, -0.0009616, 0.0011348
1: -0.0205989, -0.0202154, -0.0206052, -0.0201521, -0.0004468, 0.0003898
2: 0.0188975, 0.0192236, 0.0189029, 0.0192290, -0.0003316, 0.0003207
3: -0.0173748, -0.0167530, -0.0173191, -0.0167232, -0.0006516, 0.0005662
4: 0.0197205, 0.0202114, 0.0197425, 0.0202244, -0.0005039, 0.0004688

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0201518, 0.0211516, -0.0009008, 0.0011330
1: -0.0205989, -0.0202154, -0.0205904, -0.0201290, -0.0004699, 0.0003750
2: 0.0188975, 0.0192236, 0.0189421, 0.0192552, -0.0003577, 0.0002815
3: -0.0173748, -0.0167530, -0.0172646, -0.0166722, -0.0007026, 0.0005117
4: 0.0197205, 0.0202114, 0.0197968, 0.0202637, -0.0005433, 0.0004146

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0201501, 0.0212124, -0.0009520, 0.0011011
1: -0.0205974, -0.0201791, -0.0206052, -0.0201521, -0.0004453, 0.0004261
2: 0.0189011, 0.0192514, 0.0189029, 0.0192290, -0.0003279, 0.0003485
3: -0.0173648, -0.0167024, -0.0173191, -0.0167232, -0.0006416, 0.0006167
4: 0.0197275, 0.0202494, 0.0197425, 0.0202244, -0.0004969, 0.0005069

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0201518, 0.0211516, -0.0008913, 0.0010994
1: -0.0205974, -0.0201791, -0.0205904, -0.0201290, -0.0004685, 0.0004113
2: 0.0189011, 0.0192514, 0.0189421, 0.0192552, -0.0003541, 0.0003093
3: -0.0173648, -0.0167024, -0.0172646, -0.0166722, -0.0006926, 0.0005622
4: 0.0197275, 0.0202494, 0.0197968, 0.0202637, -0.0005362, 0.0004526

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0201535, 0.0212094, 0.0202529, 0.0213510, -0.0011975, 0.0009565
1: -0.0206047, -0.0201528, -0.0206424, -0.0202158, -0.0003888, 0.0004896
2: 0.0189047, 0.0192281, 0.0188597, 0.0192138, -0.0003092, 0.0003684
3: -0.0173170, -0.0167245, -0.0174254, -0.0167668, -0.0005502, 0.0007009
4: 0.0197446, 0.0202233, 0.0196682, 0.0201977, -0.0004531, 0.0005551

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005510, upper bound: 0.0009257
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007214, upper bound: 0.0008755
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0201545, 0.0211494, 0.0202529, 0.0213510, -0.0011966, 0.0008965
1: -0.0205900, -0.0201296, -0.0206424, -0.0202158, -0.0003742, 0.0005128
2: 0.0189436, 0.0192545, 0.0188597, 0.0192138, -0.0002702, 0.0003948
3: -0.0172644, -0.0166728, -0.0174254, -0.0167668, -0.0004976, 0.0007527
4: 0.0197985, 0.0202630, 0.0196682, 0.0201977, -0.0003992, 0.0005948

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0201464, 0.0212334, 0.0202508, 0.0212848, -0.0011384, 0.0009827
1: -0.0206135, -0.0201500, -0.0205989, -0.0202154, -0.0003981, 0.0004489
2: 0.0188968, 0.0192308, 0.0188975, 0.0192236, -0.0003268, 0.0003333
3: -0.0173277, -0.0167215, -0.0173748, -0.0167530, -0.0005747, 0.0006533
4: 0.0197336, 0.0202271, 0.0197205, 0.0202114, -0.0004778, 0.0005067

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005162, upper bound: 0.0008597
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005161, upper bound: 0.0008598
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0201464, 0.0212334, 0.0202603, 0.0212512, -0.0011048, 0.0009731
1: -0.0206135, -0.0201500, -0.0205974, -0.0201791, -0.0004344, 0.0004474
2: 0.0188968, 0.0192308, 0.0189011, 0.0192514, -0.0003546, 0.0003296
3: -0.0173277, -0.0167215, -0.0173648, -0.0167024, -0.0006253, 0.0006433
4: 0.0197336, 0.0202271, 0.0197275, 0.0202494, -0.0005158, 0.0004996

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007131, upper bound: 0.0008596
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007131, upper bound: 0.0008597
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202172, 0.0211979, 0.0202111, 0.0212044, -0.0009872, 0.0009868
1: -0.0205787, -0.0201691, -0.0205810, -0.0201717, -0.0004070, 0.0004119
2: 0.0189100, 0.0192084, 0.0189097, 0.0192050, -0.0002951, 0.0002986
3: -0.0173496, -0.0167430, -0.0173487, -0.0167472, -0.0006024, 0.0006057
4: 0.0197526, 0.0202006, 0.0197520, 0.0201970, -0.0004444, 0.0004487

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007222, upper bound: 0.0006262
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007222, upper bound: 0.0006262
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202175, 0.0211856, 0.0202111, 0.0212044, -0.0009869, 0.0009745
1: -0.0205747, -0.0201729, -0.0205810, -0.0201717, -0.0004030, 0.0004081
2: 0.0189116, 0.0192040, 0.0189097, 0.0192050, -0.0002934, 0.0002942
3: -0.0173449, -0.0167489, -0.0173487, -0.0167472, -0.0005977, 0.0005998
4: 0.0197576, 0.0201955, 0.0197520, 0.0201970, -0.0004393, 0.0004435

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007804, upper bound: 0.0007610
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008580, upper bound: 0.0008349
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202311, 0.0212266, -0.0010154, 0.0009733
1: -0.0205810, -0.0201717, -0.0206041, -0.0201669, -0.0004141, 0.0004324
2: 0.0189097, 0.0192050, 0.0189091, 0.0191864, -0.0002766, 0.0002959
3: -0.0173487, -0.0167472, -0.0173165, -0.0167978, -0.0005509, 0.0005692
4: 0.0197520, 0.0201970, 0.0197484, 0.0201631, -0.0004112, 0.0004486

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007985, upper bound: 0.0006283
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007985, upper bound: 0.0006284
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0202111, 0.0212044, 0.0202371, 0.0211815, -0.0009703, 0.0009673
1: -0.0205810, -0.0201717, -0.0205994, -0.0201716, -0.0004094, 0.0004277
2: 0.0189097, 0.0192050, 0.0189103, 0.0191949, -0.0002851, 0.0002947
3: -0.0173487, -0.0167472, -0.0173109, -0.0167887, -0.0005600, 0.0005636
4: 0.0197520, 0.0201970, 0.0197526, 0.0201746, -0.0004227, 0.0004444

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008441, upper bound: 0.0006284
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008441, upper bound: 0.0006283
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202111, 0.0212044, -0.0009733, 0.0010154
1: -0.0206041, -0.0201669, -0.0205810, -0.0201717, -0.0004324, 0.0004141
2: 0.0189091, 0.0191864, 0.0189097, 0.0192050, -0.0002959, 0.0002766
3: -0.0173165, -0.0167978, -0.0173487, -0.0167472, -0.0005692, 0.0005509
4: 0.0197484, 0.0201631, 0.0197520, 0.0201970, -0.0004486, 0.0004112

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202371, 0.0211815, 0.0202111, 0.0212044, -0.0009673, 0.0009703
1: -0.0205994, -0.0201716, -0.0205810, -0.0201717, -0.0004277, 0.0004094
2: 0.0189103, 0.0191949, 0.0189097, 0.0192050, -0.0002947, 0.0002851
3: -0.0173109, -0.0167887, -0.0173487, -0.0167472, -0.0005636, 0.0005600
4: 0.0197526, 0.0201746, 0.0197520, 0.0201970, -0.0004444, 0.0004227

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202220, 0.0211970, -0.0009659, 0.0010046
1: -0.0206041, -0.0201669, -0.0206054, -0.0201697, -0.0004344, 0.0004384
2: 0.0189091, 0.0191864, 0.0189083, 0.0191981, -0.0002890, 0.0002781
3: -0.0173165, -0.0167978, -0.0173123, -0.0167838, -0.0005327, 0.0005145
4: 0.0197484, 0.0201631, 0.0197496, 0.0201790, -0.0004306, 0.0004135

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

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

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

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
- Time for NS candidates: 3.76 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0007303
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0007302
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009007
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0009006
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0007163
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006953, upper bound: 0.0008867
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0007164
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0008921, upper bound: 0.0008867
NS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006816
NS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006816
NS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
NS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0006815
NS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
NS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
NS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
NS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006814, upper bound: 0.0008783
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0008462
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007302, upper bound: 0.0008462
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007303, upper bound: 0.0009387
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007303, upper bound: 0.0009388
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0005510, upper bound: 0.0009257
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007214, upper bound: 0.0008755
NS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0005162, upper bound: 0.0008597
NS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0005161, upper bound: 0.0008598
NS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007131, upper bound: 0.0008596
NS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007131, upper bound: 0.0008597
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007222, upper bound: 0.0006262
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007222, upper bound: 0.0006262
NS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007804, upper bound: 0.0007610
NS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0008580, upper bound: 0.0008349
NS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007985, upper bound: 0.0006283
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0007985, upper bound: 0.0006284
NS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0008441, upper bound: 0.0006284
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0008441, upper bound: 0.0006283
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006286, upper bound: 0.0006173
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0006521
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0006523
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0007961
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.76
Output dim: 0, lower bound: -0.0006795, upper bound: 0.0008072

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202564, 0.0213648, -0.0011084, 0.0011084
1: -0.0206408, -0.0202541, -0.0206408, -0.0202541, -0.0003867, 0.0003867
2: 0.0188591, 0.0191857, 0.0188591, 0.0191857, -0.0003266, 0.0003266
3: -0.0174319, -0.0168199, -0.0174319, -0.0168199, -0.0006121, 0.0006121
4: 0.0196653, 0.0201577, 0.0196653, 0.0201577, -0.0004924, 0.0004924

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202651, 0.0213436, -0.0010872, 0.0010998
1: -0.0206408, -0.0202541, -0.0206409, -0.0202179, -0.0004229, 0.0003869
2: 0.0188591, 0.0191857, 0.0188609, 0.0192120, -0.0003529, 0.0003248
3: -0.0174319, -0.0168199, -0.0174243, -0.0167697, -0.0006623, 0.0006044
4: 0.0196653, 0.0201577, 0.0196698, 0.0201953, -0.0005300, 0.0004879

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202564, 0.0213648, -0.0010998, 0.0010872
1: -0.0206409, -0.0202179, -0.0206408, -0.0202541, -0.0003869, 0.0004229
2: 0.0188609, 0.0192120, 0.0188591, 0.0191857, -0.0003248, 0.0003529
3: -0.0174243, -0.0167697, -0.0174319, -0.0168199, -0.0006044, 0.0006623
4: 0.0196698, 0.0201953, 0.0196653, 0.0201577, -0.0004879, 0.0005300

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202651, 0.0213436, -0.0010786, 0.0010786
1: -0.0206409, -0.0202179, -0.0206409, -0.0202179, -0.0004230, 0.0004230
2: 0.0188609, 0.0192120, 0.0188609, 0.0192120, -0.0003510, 0.0003510
3: -0.0174243, -0.0167697, -0.0174243, -0.0167697, -0.0006546, 0.0006546
4: 0.0196698, 0.0201953, 0.0196698, 0.0201953, -0.0005255, 0.0005255

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202508, 0.0212848, -0.0010284, 0.0011140
1: -0.0206408, -0.0202541, -0.0205989, -0.0202154, -0.0004254, 0.0003448
2: 0.0188591, 0.0191857, 0.0188975, 0.0192236, -0.0003645, 0.0002883
3: -0.0174319, -0.0168199, -0.0173748, -0.0167530, -0.0006790, 0.0005550
4: 0.0196653, 0.0201577, 0.0197205, 0.0202114, -0.0005461, 0.0004372

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202508, 0.0212848, -0.0010198, 0.0010928
1: -0.0206409, -0.0202179, -0.0205989, -0.0202154, -0.0004255, 0.0003810
2: 0.0188609, 0.0192120, 0.0188975, 0.0192236, -0.0003627, 0.0003145
3: -0.0174243, -0.0167697, -0.0173748, -0.0167530, -0.0006713, 0.0006052
4: 0.0196698, 0.0201953, 0.0197205, 0.0202114, -0.0005416, 0.0004749

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0202603, 0.0212512, -0.0009948, 0.0011045
1: -0.0206408, -0.0202541, -0.0205974, -0.0201791, -0.0004617, 0.0003434
2: 0.0188591, 0.0191857, 0.0189011, 0.0192514, -0.0003923, 0.0002846
3: -0.0174319, -0.0168199, -0.0173648, -0.0167024, -0.0007295, 0.0005449
4: 0.0196653, 0.0201577, 0.0197275, 0.0202494, -0.0005841, 0.0004302

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0202603, 0.0212512, -0.0009861, 0.0010833
1: -0.0206409, -0.0202179, -0.0205974, -0.0201791, -0.0004618, 0.0003795
2: 0.0188609, 0.0192120, 0.0189011, 0.0192514, -0.0003904, 0.0003108
3: -0.0174243, -0.0167697, -0.0173648, -0.0167024, -0.0007219, 0.0005951
4: 0.0196698, 0.0201953, 0.0197275, 0.0202494, -0.0005796, 0.0004678

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202564, 0.0213648, -0.0011140, 0.0010284
1: -0.0205989, -0.0202154, -0.0206408, -0.0202541, -0.0003448, 0.0004254
2: 0.0188975, 0.0192236, 0.0188591, 0.0191857, -0.0002883, 0.0003645
3: -0.0173748, -0.0167530, -0.0174319, -0.0168199, -0.0005550, 0.0006790
4: 0.0197205, 0.0202114, 0.0196653, 0.0201577, -0.0004372, 0.0005461

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202508, 0.0212848, -0.0010341, 0.0010341
1: -0.0205989, -0.0202154, -0.0205989, -0.0202154, -0.0003835, 0.0003835
2: 0.0188975, 0.0192236, 0.0188975, 0.0192236, -0.0003262, 0.0003262
3: -0.0173748, -0.0167530, -0.0173748, -0.0167530, -0.0006219, 0.0006219
4: 0.0197205, 0.0202114, 0.0197205, 0.0202114, -0.0004909, 0.0004909

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202651, 0.0213436, -0.0010928, 0.0010198
1: -0.0205989, -0.0202154, -0.0206409, -0.0202179, -0.0003810, 0.0004255
2: 0.0188975, 0.0192236, 0.0188609, 0.0192120, -0.0003145, 0.0003627
3: -0.0173748, -0.0167530, -0.0174243, -0.0167697, -0.0006052, 0.0006713
4: 0.0197205, 0.0202114, 0.0196698, 0.0201953, -0.0004749, 0.0005416

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0202508, 0.0212848, 0.0202603, 0.0212512, -0.0010004, 0.0010245
1: -0.0205989, -0.0202154, -0.0205974, -0.0201791, -0.0004198, 0.0003821
2: 0.0188975, 0.0192236, 0.0189011, 0.0192514, -0.0003539, 0.0003225
3: -0.0173748, -0.0167530, -0.0173648, -0.0167024, -0.0006724, 0.0006118
4: 0.0197205, 0.0202114, 0.0197275, 0.0202494, -0.0005290, 0.0004839

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202564, 0.0213648, -0.0011045, 0.0009948
1: -0.0205974, -0.0201791, -0.0206408, -0.0202541, -0.0003434, 0.0004617
2: 0.0189011, 0.0192514, 0.0188591, 0.0191857, -0.0002846, 0.0003923
3: -0.0173648, -0.0167024, -0.0174319, -0.0168199, -0.0005449, 0.0007295
4: 0.0197275, 0.0202494, 0.0196653, 0.0201577, -0.0004302, 0.0005841

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202508, 0.0212848, -0.0010245, 0.0010004
1: -0.0205974, -0.0201791, -0.0205989, -0.0202154, -0.0003821, 0.0004198
2: 0.0189011, 0.0192514, 0.0188975, 0.0192236, -0.0003225, 0.0003539
3: -0.0173648, -0.0167024, -0.0173748, -0.0167530, -0.0006118, 0.0006724
4: 0.0197275, 0.0202494, 0.0197205, 0.0202114, -0.0004839, 0.0005290

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202651, 0.0213436, -0.0010833, 0.0009861
1: -0.0205974, -0.0201791, -0.0206409, -0.0202179, -0.0003795, 0.0004618
2: 0.0189011, 0.0192514, 0.0188609, 0.0192120, -0.0003108, 0.0003904
3: -0.0173648, -0.0167024, -0.0174243, -0.0167697, -0.0005951, 0.0007219
4: 0.0197275, 0.0202494, 0.0196698, 0.0201953, -0.0004678, 0.0005796

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0202603, 0.0212512, 0.0202603, 0.0212512, -0.0009909, 0.0009909
1: -0.0205974, -0.0201791, -0.0205974, -0.0201791, -0.0004184, 0.0004184
2: 0.0189011, 0.0192514, 0.0189011, 0.0192514, -0.0003503, 0.0003503
3: -0.0173648, -0.0167024, -0.0173648, -0.0167024, -0.0006624, 0.0006624
4: 0.0197275, 0.0202494, 0.0197275, 0.0202494, -0.0005219, 0.0005219

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0201577, 0.0212396, -0.0009832, 0.0012071
1: -0.0206408, -0.0202541, -0.0206031, -0.0201518, -0.0004890, 0.0003490
2: 0.0188591, 0.0191857, 0.0189042, 0.0192219, -0.0003628, 0.0002815
3: -0.0174319, -0.0168199, -0.0173226, -0.0167278, -0.0007041, 0.0005027
4: 0.0196653, 0.0201577, 0.0197420, 0.0202145, -0.0005492, 0.0004156

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202564, 0.0213648, 0.0201653, 0.0211946, -0.0009382, 0.0011995
1: -0.0206408, -0.0202541, -0.0205990, -0.0201537, -0.0004871, 0.0003449
2: 0.0188591, 0.0191857, 0.0189050, 0.0192263, -0.0003672, 0.0002808
3: -0.0174319, -0.0168199, -0.0173175, -0.0167268, -0.0007051, 0.0004976
4: 0.0196653, 0.0201577, 0.0197457, 0.0202207, -0.0005554, 0.0004120

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0201577, 0.0212396, -0.0009745, 0.0011859
1: -0.0206409, -0.0202179, -0.0206031, -0.0201518, -0.0004891, 0.0003852
2: 0.0188609, 0.0192120, 0.0189042, 0.0192219, -0.0003609, 0.0003077
3: -0.0174243, -0.0167697, -0.0173226, -0.0167278, -0.0006964, 0.0005529
4: 0.0196698, 0.0201953, 0.0197420, 0.0202145, -0.0005447, 0.0004533

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202651, 0.0213436, 0.0201653, 0.0211946, -0.0009296, 0.0011783
1: -0.0206409, -0.0202179, -0.0205990, -0.0201537, -0.0004872, 0.0003810
2: 0.0188609, 0.0192120, 0.0189050, 0.0192263, -0.0003653, 0.0003070
3: -0.0174243, -0.0167697, -0.0173175, -0.0167268, -0.0006974, 0.0005479
4: 0.0196698, 0.0201953, 0.0197457, 0.0202207, -0.0005510, 0.0004496

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0201535, 0.0212094, 0.0202564, 0.0213648, -0.0012113, 0.0009530
1: -0.0206047, -0.0201528, -0.0206408, -0.0202541, -0.0003506, 0.0004880
2: 0.0189047, 0.0192281, 0.0188591, 0.0191857, -0.0002811, 0.0003690
3: -0.0173170, -0.0167245, -0.0174319, -0.0168199, -0.0004972, 0.0007074
4: 0.0197446, 0.0202233, 0.0196653, 0.0201577, -0.0004131, 0.0005580

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008407, upper bound: 0.0007293
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0008407, upper bound: 0.0009024
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0201535, 0.0212094, 0.0202651, 0.0213436, -0.0011901, 0.0009443
1: -0.0206047, -0.0201528, -0.0206409, -0.0202179, -0.0003868, 0.0004881
2: 0.0189047, 0.0192281, 0.0188609, 0.0192120, -0.0003073, 0.0003672
3: -0.0173170, -0.0167245, -0.0174243, -0.0167697, -0.0005474, 0.0006997
4: 0.0197446, 0.0202233, 0.0196698, 0.0201953, -0.0004507, 0.0005535

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009350, upper bound: 0.0007293
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0009350, upper bound: 0.0009026
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0201535, 0.0212094, 0.0202508, 0.0212848, -0.0011313, 0.0009586
1: -0.0206047, -0.0201528, -0.0205989, -0.0202154, -0.0003893, 0.0004461
2: 0.0189047, 0.0192281, 0.0188975, 0.0192236, -0.0003190, 0.0003307
3: -0.0173170, -0.0167245, -0.0173748, -0.0167530, -0.0005641, 0.0006503
4: 0.0197446, 0.0202233, 0.0197205, 0.0202114, -0.0004668, 0.0005028

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0201545, 0.0211494, 0.0202508, 0.0212848, -0.0011304, 0.0008986
1: -0.0205900, -0.0201296, -0.0205989, -0.0202154, -0.0003746, 0.0004692
2: 0.0189436, 0.0192545, 0.0188975, 0.0192236, -0.0002800, 0.0003571
3: -0.0172644, -0.0166728, -0.0173748, -0.0167530, -0.0005115, 0.0007021
4: 0.0197985, 0.0202630, 0.0197205, 0.0202114, -0.0004129, 0.0005426

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0201535, 0.0212094, 0.0202603, 0.0212512, -0.0010977, 0.0009491
1: -0.0206047, -0.0201528, -0.0205974, -0.0201791, -0.0004256, 0.0004446
2: 0.0189047, 0.0192281, 0.0189011, 0.0192514, -0.0003467, 0.0003270
3: -0.0173170, -0.0167245, -0.0173648, -0.0167024, -0.0006146, 0.0006403
4: 0.0197446, 0.0202233, 0.0197275, 0.0202494, -0.0005048, 0.0004958

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0201545, 0.0211494, 0.0202603, 0.0212512, -0.0010967, 0.0008891
1: -0.0205900, -0.0201296, -0.0205974, -0.0201791, -0.0004109, 0.0004678
2: 0.0189436, 0.0192545, 0.0189011, 0.0192514, -0.0003077, 0.0003534
3: -0.0172644, -0.0166728, -0.0173648, -0.0167024, -0.0005620, 0.0006920
4: 0.0197985, 0.0202630, 0.0197275, 0.0202494, -0.0004509, 0.0005355

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202172, 0.0211979, 0.0202172, 0.0211979, -0.0009807, 0.0009807
1: -0.0205787, -0.0201691, -0.0205787, -0.0201691, -0.0004095, 0.0004095
2: 0.0189100, 0.0192084, 0.0189100, 0.0192084, -0.0002984, 0.0002984
3: -0.0173496, -0.0167430, -0.0173496, -0.0167430, -0.0006066, 0.0006066
4: 0.0197526, 0.0202006, 0.0197526, 0.0202006, -0.0004481, 0.0004481

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202172, 0.0211979, 0.0202175, 0.0211856, -0.0009684, 0.0009804
1: -0.0205787, -0.0201691, -0.0205747, -0.0201729, -0.0004058, 0.0004056
2: 0.0189100, 0.0192084, 0.0189116, 0.0192040, -0.0002940, 0.0002968
3: -0.0173496, -0.0167430, -0.0173449, -0.0167489, -0.0006007, 0.0006020
4: 0.0197526, 0.0202006, 0.0197576, 0.0201955, -0.0004429, 0.0004430

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0202709, 0.0211888, 0.0202111, 0.0212044, -0.0009335, 0.0009777
1: -0.0205151, -0.0201744, -0.0205810, -0.0201717, -0.0003434, 0.0004066
2: 0.0189294, 0.0192061, 0.0189097, 0.0192050, -0.0002756, 0.0002964
3: -0.0173206, -0.0167492, -0.0173487, -0.0167472, -0.0005734, 0.0005995
4: 0.0197810, 0.0201938, 0.0197520, 0.0201970, -0.0004160, 0.0004419

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007103, upper bound: 0.0005759
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007103, upper bound: 0.0007610
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0202260, 0.0211842, 0.0202111, 0.0212044, -0.0009784, 0.0009730
1: -0.0205743, -0.0201731, -0.0205810, -0.0201717, -0.0004026, 0.0004079
2: 0.0189133, 0.0192033, 0.0189097, 0.0192050, -0.0002917, 0.0002936
3: -0.0173448, -0.0167493, -0.0173487, -0.0167472, -0.0005975, 0.0005994
4: 0.0197579, 0.0201948, 0.0197520, 0.0201970, -0.0004391, 0.0004428

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007779, upper bound: 0.0005793
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0007779, upper bound: 0.0008349
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0202172, 0.0211979, 0.0202311, 0.0212266, -0.0010094, 0.0009668
1: -0.0205787, -0.0201691, -0.0206041, -0.0201669, -0.0004117, 0.0004350
2: 0.0189100, 0.0192084, 0.0189091, 0.0191864, -0.0002764, 0.0002993
3: -0.0173496, -0.0167430, -0.0173165, -0.0167978, -0.0005518, 0.0005735
4: 0.0197526, 0.0202006, 0.0197484, 0.0201631, -0.0004106, 0.0004523

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0202175, 0.0211856, 0.0202311, 0.0212266, -0.0010091, 0.0009545
1: -0.0205747, -0.0201729, -0.0206041, -0.0201669, -0.0004078, 0.0004312
2: 0.0189116, 0.0192040, 0.0189091, 0.0191864, -0.0002748, 0.0002949
3: -0.0173449, -0.0167489, -0.0173165, -0.0167978, -0.0005471, 0.0005675
4: 0.0197576, 0.0201955, 0.0197484, 0.0201631, -0.0004055, 0.0004471

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0202172, 0.0211979, 0.0202371, 0.0211815, -0.0009642, 0.0009608
1: -0.0205787, -0.0201691, -0.0205994, -0.0201716, -0.0004071, 0.0004303
2: 0.0189100, 0.0192084, 0.0189103, 0.0191949, -0.0002849, 0.0002981
3: -0.0173496, -0.0167430, -0.0173109, -0.0167887, -0.0005609, 0.0005679
4: 0.0197526, 0.0202006, 0.0197526, 0.0201746, -0.0004221, 0.0004480

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0202175, 0.0211856, 0.0202371, 0.0211815, -0.0009640, 0.0009485
1: -0.0205747, -0.0201729, -0.0205994, -0.0201716, -0.0004031, 0.0004265
2: 0.0189116, 0.0192040, 0.0189103, 0.0191949, -0.0002833, 0.0002937
3: -0.0173449, -0.0167489, -0.0173109, -0.0167887, -0.0005563, 0.0005619
4: 0.0197576, 0.0201955, 0.0197526, 0.0201746, -0.0004170, 0.0004429

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202311, 0.0212266, -0.0009955, 0.0009955
1: -0.0206041, -0.0201669, -0.0206041, -0.0201669, -0.0004371, 0.0004371
2: 0.0189091, 0.0191864, 0.0189091, 0.0191864, -0.0002772, 0.0002772
3: -0.0173165, -0.0167978, -0.0173165, -0.0167978, -0.0005187, 0.0005187
4: 0.0197484, 0.0201631, 0.0197484, 0.0201631, -0.0004148, 0.0004148

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0202311, 0.0212266, 0.0202371, 0.0211815, -0.0009504, 0.0009895
1: -0.0206041, -0.0201669, -0.0205994, -0.0201716, -0.0004325, 0.0004325
2: 0.0189091, 0.0191864, 0.0189103, 0.0191949, -0.0002857, 0.0002761
3: -0.0173165, -0.0167978, -0.0173109, -0.0167887, -0.0005278, 0.0005130
4: 0.0197484, 0.0201631, 0.0197526, 0.0201746, -0.0004263, 0.0004106

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0202371, 0.0211815, 0.0202311, 0.0212266, -0.0009895, 0.0009504
1: -0.0205994, -0.0201716, -0.0206041, -0.0201669, -0.0004325, 0.0004325
2: 0.0189103, 0.0191949, 0.0189091, 0.0191864, -0.0002761, 0.0002857
3: -0.0173109, -0.0167887, -0.0173165, -0.0167978, -0.0005130, 0.0005278
4: 0.0197526, 0.0201746, 0.0197484, 0.0201631, -0.0004106, 0.0004263

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0202371, 0.0211815, 0.0202371, 0.0211815, -0.0009444, 0.0009444
1: -0.0205994, -0.0201716, -0.0205994, -0.0201716, -0.0004279, 0.0004279
2: 0.0189103, 0.0191949, 0.0189103, 0.0191949, -0.0002846, 0.0002846
3: -0.0173109, -0.0167887, -0.0173109, -0.0167887, -0.0005222, 0.0005222
4: 0.0197526, 0.0201746, 0.0197526, 0.0201746, -0.0004221, 0.0004221

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.68 + 341.86 = 345.53 seconds
