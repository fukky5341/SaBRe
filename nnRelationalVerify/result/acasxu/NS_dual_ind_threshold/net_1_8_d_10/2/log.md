## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.5895e-05


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0202691, -0.0202429, -0.0202691, -0.0202429, -0.0000261, 0.0000261)
1: (-0.0191910, -0.0185430, -0.0191910, -0.0185430, -0.0006480, 0.0006480)
2: (-0.0191720, -0.0184301, -0.0191720, -0.0184301, -0.0007418, 0.0007418)
3: (-0.0182742, -0.0171878, -0.0182742, -0.0171878, -0.0010863, 0.0010863)
4: (-0.0182938, -0.0173051, -0.0182938, -0.0173051, -0.0009887, 0.0009887)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.08 + 0.56 = 1.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0000187, upper bound: 0.0000186

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000175
time: 0.15 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.38 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.0000164, upper bound: 0.0000175
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.38
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0202683, -0.0202467, -0.0202691, -0.0202429, -0.0000254, 0.0000224
1: -0.0191382, -0.0185592, -0.0191910, -0.0185430, -0.0005952, 0.0006318
2: -0.0191202, -0.0184587, -0.0191720, -0.0184301, -0.0006900, 0.0007133
3: -0.0181853, -0.0172092, -0.0182742, -0.0171878, -0.0009975, 0.0010650
4: -0.0182031, -0.0173372, -0.0182938, -0.0173051, -0.0008980, 0.0009566

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.15 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.15 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0202697, -0.0202498, -0.0202691, -0.0202429, -0.0000267, 0.0000193
1: -0.0190973, -0.0185272, -0.0191910, -0.0185430, -0.0005543, 0.0006638
2: -0.0190817, -0.0184325, -0.0191720, -0.0184301, -0.0006516, 0.0007395
3: -0.0181015, -0.0171318, -0.0182742, -0.0171878, -0.0009137, 0.0011423
4: -0.0181199, -0.0172843, -0.0182938, -0.0173051, -0.0008148, 0.0010095

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.35 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.35
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.35
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.35
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.35
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202683, -0.0202467, -0.0202683, -0.0202467, -0.0000216, 0.0000216
1: -0.0191382, -0.0185592, -0.0191382, -0.0185592, -0.0005790, 0.0005790
2: -0.0191202, -0.0184587, -0.0191202, -0.0184587, -0.0006615, 0.0006615
3: -0.0181853, -0.0172092, -0.0181853, -0.0172092, -0.0009762, 0.0009762
4: -0.0182031, -0.0173372, -0.0182031, -0.0173372, -0.0008658, 0.0008658

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000173
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000175
time: 0.16 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202683, -0.0202467, -0.0202697, -0.0202498, -0.0000185, 0.0000230
1: -0.0191382, -0.0185592, -0.0190973, -0.0185272, -0.0006110, 0.0005381
2: -0.0191202, -0.0184587, -0.0190817, -0.0184325, -0.0006877, 0.0006230
3: -0.0181853, -0.0172092, -0.0181015, -0.0171318, -0.0010535, 0.0008924
4: -0.0182031, -0.0173372, -0.0181199, -0.0172843, -0.0009188, 0.0007826

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000174
time: 0.15 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000175
time: 0.16 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202697, -0.0202498, -0.0202683, -0.0202467, -0.0000230, 0.0000185
1: -0.0190973, -0.0185272, -0.0191382, -0.0185592, -0.0005381, 0.0006110
2: -0.0190817, -0.0184325, -0.0191202, -0.0184587, -0.0006230, 0.0006877
3: -0.0181015, -0.0171318, -0.0181853, -0.0172092, -0.0008924, 0.0010535
4: -0.0181199, -0.0172843, -0.0182031, -0.0173372, -0.0007826, 0.0009188

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000155
time: 0.16 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
time: 0.16 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202697, -0.0202498, -0.0202697, -0.0202498, -0.0000199, 0.0000199
1: -0.0190973, -0.0185272, -0.0190973, -0.0185272, -0.0005701, 0.0005701
2: -0.0190817, -0.0184325, -0.0190817, -0.0184325, -0.0006492, 0.0006492
3: -0.0181015, -0.0171318, -0.0181015, -0.0171318, -0.0009697, 0.0009697
4: -0.0181199, -0.0172843, -0.0181199, -0.0172843, -0.0008356, 0.0008356

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000155
time: 0.15 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000161
time: 0.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.82 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000173
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000175
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000174
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000175
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000155
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000162
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000155, upper bound: 0.0000155
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.82
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000161

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0202672, -0.0202488, -0.0202683, -0.0202467, -0.0000205, 0.0000195
1: -0.0191049, -0.0185793, -0.0191382, -0.0185592, -0.0005457, 0.0005589
2: -0.0190893, -0.0184904, -0.0191202, -0.0184587, -0.0006306, 0.0006298
3: -0.0181317, -0.0172386, -0.0181853, -0.0172092, -0.0009225, 0.0009467
4: -0.0181501, -0.0173805, -0.0182031, -0.0173372, -0.0008129, 0.0008226

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000179
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000179
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202680, -0.0202472, -0.0202683, -0.0202467, -0.0000213, 0.0000211
1: -0.0191220, -0.0185568, -0.0191382, -0.0185592, -0.0005628, 0.0005814
2: -0.0191069, -0.0184651, -0.0191202, -0.0184587, -0.0006482, 0.0006551
3: -0.0181744, -0.0171990, -0.0181853, -0.0172092, -0.0009653, 0.0009863
4: -0.0181943, -0.0173463, -0.0182031, -0.0173372, -0.0008571, 0.0008568

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000180
time: 0.16 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000180
time: 0.16 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202672, -0.0202488, -0.0202697, -0.0202498, -0.0000174, 0.0000209
1: -0.0191049, -0.0185793, -0.0190973, -0.0185272, -0.0005777, 0.0005180
2: -0.0190893, -0.0184904, -0.0190817, -0.0184325, -0.0006568, 0.0005913
3: -0.0181317, -0.0172386, -0.0181015, -0.0171318, -0.0009999, 0.0008630
4: -0.0181501, -0.0173805, -0.0181199, -0.0172843, -0.0008658, 0.0007394

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000155
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000173
time: 0.16 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202680, -0.0202472, -0.0202697, -0.0202498, -0.0000182, 0.0000225
1: -0.0191220, -0.0185568, -0.0190973, -0.0185272, -0.0005948, 0.0005405
2: -0.0191069, -0.0184651, -0.0190817, -0.0184325, -0.0006744, 0.0006166
3: -0.0181744, -0.0171990, -0.0181015, -0.0171318, -0.0010426, 0.0009026
4: -0.0181943, -0.0173463, -0.0181199, -0.0172843, -0.0009101, 0.0007736

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000159
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000174
time: 0.15 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0202696, -0.0202498, -0.0202683, -0.0202467, -0.0000229, 0.0000185
1: -0.0190971, -0.0185301, -0.0191382, -0.0185592, -0.0005379, 0.0006081
2: -0.0190816, -0.0184358, -0.0191202, -0.0184587, -0.0006229, 0.0006844
3: -0.0181009, -0.0171365, -0.0181853, -0.0172092, -0.0008917, 0.0010488
4: -0.0181192, -0.0172883, -0.0182031, -0.0173372, -0.0007820, 0.0009148

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000173, upper bound: 0.0000162
time: 0.15 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000175, upper bound: 0.0000162
time: 0.16 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0202696, -0.0202498, -0.0202697, -0.0202498, -0.0000198, 0.0000199
1: -0.0190971, -0.0185301, -0.0190973, -0.0185272, -0.0005699, 0.0005672
2: -0.0190816, -0.0184358, -0.0190817, -0.0184325, -0.0006491, 0.0006460
3: -0.0181009, -0.0171365, -0.0181015, -0.0171318, -0.0009691, 0.0009650
4: -0.0181192, -0.0172883, -0.0181199, -0.0172843, -0.0008350, 0.0008316

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0000147, upper bound: 0.0000149
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0000147, upper bound: 0.0000161
time: 0.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.39 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000179
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000179
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000180
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000180, upper bound: 0.0000180
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000155
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000163, upper bound: 0.0000173
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000156, upper bound: 0.0000159
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000162, upper bound: 0.0000174
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000173, upper bound: 0.0000162
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000175, upper bound: 0.0000162
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000147, upper bound: 0.0000149
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0000147, upper bound: 0.0000161

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0202672, -0.0202488, -0.0202672, -0.0202488, -0.0000184, 0.0000184
1: -0.0191049, -0.0185793, -0.0191049, -0.0185793, -0.0005256, 0.0005256
2: -0.0190893, -0.0184904, -0.0190893, -0.0184904, -0.0005989, 0.0005989
3: -0.0181317, -0.0172386, -0.0181317, -0.0172386, -0.0008931, 0.0008931
4: -0.0181501, -0.0173805, -0.0181501, -0.0173805, -0.0007696, 0.0007696

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202672, -0.0202488, -0.0202680, -0.0202472, -0.0000200, 0.0000192
1: -0.0191049, -0.0185793, -0.0191220, -0.0185568, -0.0005481, 0.0005427
2: -0.0190893, -0.0184904, -0.0191069, -0.0184651, -0.0006242, 0.0006165
3: -0.0181317, -0.0172386, -0.0181744, -0.0171990, -0.0009327, 0.0009358
4: -0.0181501, -0.0173805, -0.0181943, -0.0173463, -0.0008038, 0.0008138

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202680, -0.0202472, -0.0202672, -0.0202488, -0.0000192, 0.0000200
1: -0.0191220, -0.0185568, -0.0191049, -0.0185793, -0.0005427, 0.0005481
2: -0.0191069, -0.0184651, -0.0190893, -0.0184904, -0.0006165, 0.0006242
3: -0.0181744, -0.0171990, -0.0181317, -0.0172386, -0.0009358, 0.0009327
4: -0.0181943, -0.0173463, -0.0181501, -0.0173805, -0.0008138, 0.0008038

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202680, -0.0202472, -0.0202680, -0.0202472, -0.0000208, 0.0000208
1: -0.0191220, -0.0185568, -0.0191220, -0.0185568, -0.0005652, 0.0005652
2: -0.0191069, -0.0184651, -0.0191069, -0.0184651, -0.0006418, 0.0006418
3: -0.0181744, -0.0171990, -0.0181744, -0.0171990, -0.0009754, 0.0009754
4: -0.0181943, -0.0173463, -0.0181943, -0.0173463, -0.0008480, 0.0008480

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0202672, -0.0202488, -0.0202696, -0.0202498, -0.0000174, 0.0000208
1: -0.0191049, -0.0185793, -0.0190971, -0.0185301, -0.0005748, 0.0005178
2: -0.0190893, -0.0184904, -0.0190816, -0.0184358, -0.0006535, 0.0005912
3: -0.0181317, -0.0172386, -0.0181009, -0.0171365, -0.0009952, 0.0008623
4: -0.0181501, -0.0173805, -0.0181192, -0.0172883, -0.0008619, 0.0007387

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202679, -0.0202473, -0.0202668, -0.0202486, -0.0000194, 0.0000195
1: -0.0191208, -0.0185582, -0.0191057, -0.0185972, -0.0005236, 0.0005475
2: -0.0191057, -0.0184666, -0.0190900, -0.0185113, -0.0005944, 0.0006234
3: -0.0181721, -0.0172009, -0.0181347, -0.0172647, -0.0009074, 0.0009338
4: -0.0181918, -0.0173479, -0.0181532, -0.0174021, -0.0007898, 0.0008052

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202680, -0.0202472, -0.0202696, -0.0202498, -0.0000182, 0.0000224
1: -0.0191220, -0.0185568, -0.0190971, -0.0185301, -0.0005919, 0.0005403
2: -0.0191069, -0.0184651, -0.0190816, -0.0184358, -0.0006712, 0.0006165
3: -0.0181744, -0.0171990, -0.0181009, -0.0171365, -0.0010379, 0.0009019
4: -0.0181943, -0.0173463, -0.0181192, -0.0172883, -0.0009061, 0.0007730

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0202696, -0.0202498, -0.0202672, -0.0202488, -0.0000208, 0.0000174
1: -0.0190971, -0.0185301, -0.0191049, -0.0185793, -0.0005178, 0.0005748
2: -0.0190816, -0.0184358, -0.0190893, -0.0184904, -0.0005912, 0.0006535
3: -0.0181009, -0.0171365, -0.0181317, -0.0172386, -0.0008623, 0.0009952
4: -0.0181192, -0.0172883, -0.0181501, -0.0173805, -0.0007387, 0.0008619

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202696, -0.0202498, -0.0202680, -0.0202472, -0.0000224, 0.0000182
1: -0.0190971, -0.0185301, -0.0191220, -0.0185568, -0.0005403, 0.0005919
2: -0.0190816, -0.0184358, -0.0191069, -0.0184651, -0.0006165, 0.0006712
3: -0.0181009, -0.0171365, -0.0181744, -0.0171990, -0.0009019, 0.0010379
4: -0.0181192, -0.0172883, -0.0181943, -0.0173463, -0.0007730, 0.0009061

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0202696, -0.0202498, -0.0202696, -0.0202498, -0.0000198, 0.0000198
1: -0.0190971, -0.0185301, -0.0190971, -0.0185301, -0.0005671, 0.0005671
2: -0.0190816, -0.0184358, -0.0190816, -0.0184358, -0.0006458, 0.0006458
3: -0.0181009, -0.0171365, -0.0181009, -0.0171365, -0.0009644, 0.0009644
4: -0.0181192, -0.0172883, -0.0181192, -0.0172883, -0.0008310, 0.0008310

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.64 + 34.76 = 36.40 seconds
