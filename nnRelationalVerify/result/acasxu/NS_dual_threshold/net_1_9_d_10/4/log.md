## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 3.5844923581200003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910)
1: (-6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244)
2: (-3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980)
3: (-4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017)
4: (-2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.02 + 1.45 = 2.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5852094

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5823907, upper bound: 3.5850495
time: 0.56 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5852094
time: 0.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.10 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 4, lower bound: -3.5823907, upper bound: 3.5850495
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5852094

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.5853020, 0.5479245, -0.9467030, 0.8965804, -1.4818822, 1.4946275
1: -4.0650864, 1.4538300, -5.5215416, 2.2792053, -6.3442917, 6.9753714
2: -2.2032108, 1.5162734, -3.3538361, 2.2188351, -4.4220448, 4.8701091
3: -2.5908508, 1.0841043, -3.6899052, 1.6014123, -4.1922631, 4.7740088
4: -1.4666536, 1.2613025, -2.3790693, 1.9149839, -3.3816369, 3.6403718

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5800840, upper bound: 3.5849003
time: 0.46 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5823907, upper bound: 3.5850364
time: 0.46 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5809301, upper bound: 3.5850364
time: 0.47 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.1124897, 1.0891812, -1.0192617, 0.9743139, -2.0868037, 2.1084428
1: -6.9265957, 2.6608543, -5.9282703, 2.4417772, -9.3683720, 8.5891247
2: -3.9553053, 2.6250048, -3.5934830, 2.3785326, -6.3338370, 6.2184877
3: -4.5624657, 1.9085267, -3.9672160, 1.7182016, -6.2806659, 5.8757429
4: -2.8212972, 2.2729437, -2.5622408, 2.0669615, -4.8882575, 4.8351841

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850496, upper bound: 3.5823907
time: 0.76 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850496, upper bound: 3.5852093
time: 0.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.12 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 4, lower bound: -3.5823907, upper bound: 3.5850364
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 4, lower bound: -3.5809301, upper bound: 3.5850364
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 4, lower bound: -3.5850496, upper bound: 3.5823907
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 4, lower bound: -3.5850496, upper bound: 3.5852093

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.5319781, 0.4955367, -0.9454479, 0.8952616, -1.4272397, 1.4409846
1: -3.7765317, 1.3281118, -5.5138221, 2.2764354, -6.0529671, 6.8419333
2: -2.0180674, 1.3881541, -3.3496780, 2.2158611, -4.2339287, 4.7378311
3: -2.3991456, 1.0002824, -3.6848671, 1.5993605, -3.9985061, 4.6851492
4: -1.3306940, 1.1480017, -2.3759751, 1.9122429, -3.2429364, 3.5239763

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5821490, upper bound: 3.5807703
time: 0.49 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5821490, upper bound: 3.5850364
time: 0.47 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.5794569, 0.5403550, -0.9396650, 0.8893370, -1.4687939, 1.4800197
1: -3.9291410, 1.4343688, -5.4786997, 2.2635164, -6.1926570, 6.9130688
2: -2.1413770, 1.4701924, -3.3302732, 2.2035177, -4.3448944, 4.8004646
3: -2.5059297, 1.0605237, -3.6636286, 1.5900592, -4.0959883, 4.7241516
4: -1.4388697, 1.2223992, -2.3615847, 1.9009411, -3.3398108, 3.5839829

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5807703, upper bound: 3.5807703
time: 0.51 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5807703, upper bound: 3.5850364
time: 0.59 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.1124897, 1.0891812, -0.5853020, 0.5479245, -1.6604142, 1.6744832
1: -6.9265957, 2.6608543, -4.0650864, 1.4538300, -8.3804255, 6.7259398
2: -3.9553053, 2.6250048, -2.2032108, 1.5162734, -5.4715786, 4.8282137
3: -4.5624657, 1.9085267, -2.5908508, 1.0841043, -5.6465688, 4.4993773
4: -2.8212972, 2.2729437, -1.4666536, 1.2613025, -4.0825992, 3.7395973

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849003, upper bound: 3.5800840
time: 0.63 seconds

## Relational analysis of NS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850364, upper bound: 3.5823907
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850364, upper bound: 3.5809301
time: 0.48 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.1124897, 1.0891812, -1.1124897, 1.0891812, -2.2016709, 2.2016709
1: -6.9265957, 2.6608543, -6.9265957, 2.6608543, -9.5874491, 9.5874500
2: -3.9553053, 2.6250048, -3.9553053, 2.6250048, -6.5803099, 6.5803089
3: -4.5624657, 1.9085267, -4.5624657, 1.9085267, -6.4709921, 6.4709921
4: -2.8212972, 2.2729437, -2.8212972, 2.2729437, -5.0942407, 5.0942407

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850496, upper bound: 3.5851611
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850364, upper bound: 3.5851917
time: 0.53 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.94 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5821490, upper bound: 3.5807703
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5821490, upper bound: 3.5850364
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5807703, upper bound: 3.5807703
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5807703, upper bound: 3.5850364
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5850364, upper bound: 3.5823907
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5850364, upper bound: 3.5809301
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5850496, upper bound: 3.5851611
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.94
Output dim: 4, lower bound: -3.5850364, upper bound: 3.5851917

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.5319781, 0.4955367, -1.1001265, 1.0756989, -1.6076769, 1.5956632
1: -3.7765317, 1.3281118, -6.8686352, 2.6343448, -6.4108763, 8.1967459
2: -2.0180674, 1.3881541, -3.9191608, 2.5974195, -4.6154871, 5.3073149
3: -2.3991456, 1.0002824, -4.5231256, 1.8887237, -4.2878690, 5.5234079
4: -1.3306940, 1.1480017, -2.7927160, 2.2462902, -3.5769839, 3.9407177

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5742985, upper bound: 3.5836801
time: 0.40 seconds

## Relational analysis of NS_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5816315, upper bound: 3.5848285
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5794569, 0.5403550, -1.0933208, 1.0677358, -1.6471926, 1.6336757
1: -3.9291410, 1.4343688, -6.8142390, 2.6177278, -6.5468678, 8.2486076
2: -2.1413770, 1.4701924, -3.8957603, 2.5804291, -4.7218060, 5.3659525
3: -2.5059297, 1.0605237, -4.4887667, 1.8752828, -4.3812122, 5.5492907
4: -1.4388697, 1.2223992, -2.7749434, 2.2305682, -3.6694379, 3.9973421

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5800279, upper bound: 3.5849298
time: 0.42 seconds

## Relational analysis of NS_A1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5788811, upper bound: 3.5849370
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.1114266, 1.0881063, -0.5319781, 0.4955367, -1.6069633, 1.6200842
1: -6.9205351, 2.6585155, -3.7765317, 1.3281118, -8.2486448, 6.4350457
2: -3.9517827, 2.6225917, -2.0180674, 1.3881541, -5.3399363, 4.6406593
3: -4.5584331, 1.9068655, -2.3991456, 1.0002824, -5.5587153, 4.3060107
4: -2.8187375, 2.2707751, -1.3306940, 1.1480017, -3.9667392, 3.6014690

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5841265, upper bound: 3.5822640
time: 0.70 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848903, upper bound: 3.5823567
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.1037471, 1.0792294, -0.5794569, 0.5403550, -1.6441017, 1.6586862
1: -6.8623571, 2.6400712, -3.9291410, 1.4343688, -8.2967262, 6.5692091
2: -3.9259319, 2.6037395, -2.1413770, 1.4701924, -5.3961239, 4.7451162
3: -4.5214481, 1.8920832, -2.5059297, 1.0605237, -5.5819716, 4.3980112
4: -2.7990019, 2.2532501, -1.4388697, 1.2223992, -4.0214005, 3.6921186

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849298, upper bound: 3.5802425
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849370, upper bound: 3.5790956
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.0273520, 1.0056329, -1.1114266, 1.0881063, -2.1154578, 2.1170592
1: -6.4656796, 2.4740899, -6.9205351, 2.6585155, -9.1241951, 9.3946228
2: -3.6766818, 2.4367743, -3.9517827, 2.6225917, -6.2992735, 6.3885565
3: -4.2514768, 1.7788718, -4.5584331, 1.9068655, -6.1583424, 6.3373046
4: -2.6175458, 2.1039290, -2.8187375, 2.2707751, -4.8883209, 4.9226665

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851611
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851611
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.9848136, 0.9734467, -1.1037471, 1.0792294, -2.0640430, 2.0771933
1: -6.4386358, 2.3843112, -6.8623571, 2.6400712, -9.0787039, 9.2466679
2: -3.5363216, 2.3778179, -3.9259319, 2.6037395, -6.1400604, 6.3037496
3: -4.1827745, 1.7388850, -4.5214481, 1.8920832, -6.0748572, 6.2603331
4: -2.4861708, 2.0413406, -2.7990019, 2.2532501, -4.7394209, 4.8403425

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5813090, upper bound: 3.5827456
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851396, upper bound: 3.5851204
time: 0.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.02 seconds
NS_A1_A1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5742985, upper bound: 3.5836801
NS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5816315, upper bound: 3.5848285
NS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5800279, upper bound: 3.5849298
NS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5788811, upper bound: 3.5849370
NS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5841265, upper bound: 3.5822640
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5848903, upper bound: 3.5823567
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5849298, upper bound: 3.5802425
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5849370, upper bound: 3.5790956
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851611
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851611
NS_A2_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5813090, upper bound: 3.5827456
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 4, lower bound: -3.5851396, upper bound: 3.5851204

## BFS NS instance: NS_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.5168161, 0.4792042, -1.1001265, 1.0756989, -1.5925150, 1.5793307
1: -3.7153072, 1.2934246, -6.8686352, 2.6343448, -6.3496523, 8.1620588
2: -1.9735659, 1.3561270, -3.9191608, 2.5974195, -4.5709844, 5.2752876
3: -2.3557849, 0.9780343, -4.5231256, 1.8887237, -4.2445078, 5.5011597
4: -1.2946410, 1.1183815, -2.7927160, 2.2462902, -3.5409312, 3.9110975

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_A2_A1

### Relational analysis result of NS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5813849, upper bound: 3.5847176
time: 0.53 seconds

## Relational analysis of NS_A1_A1_B2_A2_A2

### Relational analysis result of NS_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5698297, upper bound: 3.5847385
time: 0.56 seconds

## BFS NS instance: NS_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5197653, 0.4778241, -1.0933208, 1.0677358, -1.5875010, 1.5711449
1: -3.5102820, 1.2897317, -6.8142390, 2.6177278, -6.1280093, 8.1039696
2: -1.9258542, 1.3158989, -3.8957603, 2.5804291, -4.5062833, 5.2116594
3: -2.2408814, 0.9519234, -4.4887667, 1.8752828, -4.1161642, 5.4406900
4: -1.2813401, 1.0940826, -2.7749434, 2.2305682, -3.5119083, 3.8690257

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5801827, upper bound: 3.5837325
time: 0.48 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5802082, upper bound: 3.5847751
time: 0.46 seconds

## BFS NS instance: NS_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.5855126, 0.5380251, -1.0933208, 1.0677358, -1.6532484, 1.6313459
1: -3.7470031, 1.4501450, -6.8142390, 2.6177278, -6.3647304, 8.2643843
2: -2.1421211, 1.4519448, -3.8957603, 2.5804291, -4.7225504, 5.3477049
3: -2.4156415, 1.0549080, -4.4887667, 1.8752828, -4.2909241, 5.5436745
4: -1.4396001, 1.2183769, -2.7749434, 2.2305682, -3.6701684, 3.9933190

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5772140, upper bound: 3.5784333
time: 0.46 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5775077, upper bound: 3.5847355
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1.0829937, 1.0572182, -0.5319781, 0.4955367, -1.5785302, 1.5891960
1: -6.7634592, 2.5928497, -3.7765317, 1.3281118, -8.0915699, 6.3693814
2: -3.8548331, 2.5620289, -2.0180674, 1.3881541, -5.2429872, 4.5800962
3: -4.4495683, 1.8610864, -2.3991456, 1.0002824, -5.4498506, 4.2602320
4: -2.7488444, 2.2132304, -1.3306940, 1.1480017, -3.8968456, 3.5439239

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5836773, upper bound: 3.5744365
time: 0.50 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846690, upper bound: 3.5821247
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -1.1037471, 1.0792294, -0.5197653, 0.4778241, -1.5815712, 1.5989946
1: -6.8623571, 2.6400712, -3.5102820, 1.2897317, -8.1520891, 6.1503515
2: -3.9259319, 2.6037395, -1.9258542, 1.3158989, -5.2418308, 4.5295930
3: -4.5214481, 1.8920832, -2.2408814, 0.9519234, -5.4733715, 4.1329646
4: -2.7990019, 2.2532501, -1.2813401, 1.0940826, -3.8930840, 3.5345900

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5837325, upper bound: 3.5801827
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847751, upper bound: 3.5802082
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -1.1037471, 1.0792294, -0.5855126, 0.5380251, -1.6417722, 1.6647420
1: -6.8623571, 2.6400712, -3.7470031, 1.4501450, -8.3125019, 6.3870726
2: -3.9259319, 2.6037395, -2.1421211, 1.4519448, -5.3778768, 4.7458606
3: -4.5214481, 1.8920832, -2.4156415, 1.0549080, -5.5763559, 4.3077240
4: -2.7990019, 2.2532501, -1.4396001, 1.2183769, -4.0173779, 3.6928496

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5784333, upper bound: 3.5773811
time: 0.75 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847355, upper bound: 3.5775077
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.0273520, 1.0056329, -1.0273520, 1.0056329, -2.0329843, 2.0329843
1: -6.4656796, 2.4740899, -6.4656796, 2.4740899, -8.9397697, 8.9397697
2: -3.6766818, 2.4367743, -3.6766818, 2.4367743, -6.1134558, 6.1134558
3: -4.2514768, 1.7788718, -4.2514768, 1.7788718, -6.0303488, 6.0303488
4: -2.6175458, 2.1039290, -2.6175458, 2.1039290, -4.7214746, 4.7214746

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850349, upper bound: 3.5814033
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5850769
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.0273520, 1.0056329, -0.9848136, 0.9734467, -2.0007985, 1.9904459
1: -6.4656796, 2.4740899, -6.4386358, 2.3843112, -8.8499908, 8.9127254
2: -3.6766818, 2.4367743, -3.5363216, 2.3778179, -6.0544996, 5.9730949
3: -4.2514768, 1.7788718, -4.1827745, 1.7388850, -5.9903617, 5.9616451
4: -2.6175458, 2.1039290, -2.4861708, 2.0413406, -4.6588864, 4.5900998

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850349, upper bound: 3.5814033
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5850769
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.1396009, 1.1990455, -1.0908453, 1.0648979, -2.2044978, 2.2898908
1: -8.3805723, 2.7883079, -6.7825513, 2.6066382, -10.9872103, 9.5708590
2: -4.2246008, 2.8196661, -3.8857012, 2.5708313, -6.7954321, 6.7053671
3: -5.3370137, 2.1121480, -4.4637890, 1.8663213, -7.2033348, 6.5759354
4: -2.9356136, 2.4462025, -2.7680063, 2.2231665, -5.1587801, 5.2142086

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5851204
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5851204
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.31 seconds
NS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5813849, upper bound: 3.5847176
NS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5698297, upper bound: 3.5847385
NS_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5801827, upper bound: 3.5837325
NS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5802082, upper bound: 3.5847751
NS_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5772140, upper bound: 3.5784333
NS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5775077, upper bound: 3.5847355
NS_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5836773, upper bound: 3.5744365
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5846690, upper bound: 3.5821247
NS_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5837325, upper bound: 3.5801827
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5847751, upper bound: 3.5802082
NS_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5784333, upper bound: 3.5773811
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5847355, upper bound: 3.5775077
NS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5850349, upper bound: 3.5814033
NS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5850769
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5850349, upper bound: 3.5814033
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5850769
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5851204
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 4, lower bound: -3.5851330, upper bound: 3.5851204

## BFS NS instance: NS_A1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.4585693, 0.4178070, -1.1001265, 1.0756989, -1.5342681, 1.5179335
1: -3.3594925, 1.1605518, -6.8686352, 2.6343448, -5.9938374, 8.0291862
2: -1.7860143, 1.2129331, -3.9191608, 2.5974195, -4.3834329, 5.1320939
3: -2.1246309, 0.8812460, -4.5231256, 1.8887237, -4.0133548, 5.4043717
4: -1.1309596, 0.9925737, -2.7927160, 2.2462902, -3.3772497, 3.7852898

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5795595, upper bound: 3.5784132
time: 0.47 seconds

## Relational analysis of NS_A1_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5795596, upper bound: 3.5846709
time: 0.54 seconds

## BFS NS instance: NS_A1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.5026631, 0.4498851, -1.1001265, 1.0756989, -1.5783620, 1.5500116
1: -3.3779643, 1.2705357, -6.8686352, 2.6343448, -6.0123091, 8.1391706
2: -1.9132642, 1.2897635, -3.9191608, 2.5974195, -4.5106831, 5.2089243
3: -2.1676135, 0.9355350, -4.5231256, 1.8887237, -4.0563364, 5.4586596
4: -1.2523308, 1.0681732, -2.7927160, 2.2462902, -3.4986210, 3.8608892

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5775890, upper bound: 3.5784333
time: 0.46 seconds

## Relational analysis of NS_A1_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5775891, upper bound: 3.5846641
time: 0.53 seconds

## BFS NS instance: NS_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5197653, 0.4778241, -1.0653226, 1.0373821, -1.5571474, 1.5431467
1: -3.5102820, 1.2897317, -6.6589355, 2.5530047, -6.0632868, 7.9486675
2: -1.9258542, 1.3158989, -3.8000662, 2.5208368, -4.4466910, 5.1159649
3: -2.2408814, 0.9519234, -4.3811250, 1.8301524, -4.0710340, 5.3330479
4: -1.2813401, 1.0940826, -2.7060444, 2.1739492, -3.4552894, 3.8001270

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5780914, upper bound: 3.5782554
time: 0.58 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5782253, upper bound: 3.5845501
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5855126, 0.5380251, -1.0562047, 1.0277854, -1.6132981, 1.5942296
1: -3.7470031, 1.4501450, -6.6346488, 2.5324912, -6.2794929, 8.0847912
2: -2.1421211, 1.4519448, -3.7815390, 2.5050836, -4.6472044, 5.2334838
3: -2.4156415, 1.0549080, -4.3620949, 1.8176227, -4.2332640, 5.4170027
4: -1.4396001, 1.2183769, -2.6843405, 2.1580725, -3.5976725, 3.9027166

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5746802, upper bound: 3.5839230
time: 0.53 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5746802, upper bound: 3.5847355
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0829937, 1.0572182, -0.5168161, 0.4792042, -1.5621976, 1.5740342
1: -6.7634592, 2.5928497, -3.7153072, 1.2934246, -8.0568829, 6.3081560
2: -3.8548331, 2.5620289, -1.9735659, 1.3561270, -5.2109594, 4.5355949
3: -4.4495683, 1.8610864, -2.3557849, 0.9780343, -5.4276028, 4.2168708
4: -2.7488444, 2.2132304, -1.2946410, 1.1183815, -3.8672256, 3.5078714

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845501, upper bound: 3.5813524
time: 0.51 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845695, upper bound: 3.5781086
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0754910, 1.0484991, -0.5197653, 0.4778241, -1.5533150, 1.5682644
1: -6.7056823, 2.5747271, -3.5102820, 1.2897317, -7.9954138, 6.0850067
2: -3.8294687, 2.5434422, -1.9258542, 1.3158989, -5.1453676, 4.4692965
3: -4.4129324, 1.8464693, -2.2408814, 0.9519234, -5.3648553, 4.0873508
4: -2.7295151, 2.1959331, -1.2813401, 1.0940826, -3.8235974, 3.4772732

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5782554, upper bound: 3.5780914
time: 0.46 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845501, upper bound: 3.5782253
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0646404, 1.0371002, -0.5855126, 0.5380251, -1.6026652, 1.6226128
1: -6.6740546, 2.5505240, -3.7470031, 1.4501450, -8.1241999, 6.2975268
2: -3.8061781, 2.5241504, -2.1421211, 1.4519448, -5.2581229, 4.6662712
3: -4.3888259, 1.8312016, -2.4156415, 1.0549080, -5.4437342, 4.2468433
4: -2.7039649, 2.1767035, -1.4396001, 1.2183769, -3.9223406, 3.6163034

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5839230, upper bound: 3.5746802
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5839230, upper bound: 3.5775077
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1.0273520, 1.0056329, -0.8266253, 0.8195326, -1.8468845, 1.8322581
1: -6.4656796, 2.4740899, -5.6839504, 2.0457618, -8.5114412, 8.1580400
2: -3.6766818, 2.4367743, -3.0678792, 2.0530486, -5.7297306, 5.5046535
3: -4.2514768, 1.7788718, -3.6650279, 1.5048720, -5.7563486, 5.4438996
4: -2.6175458, 2.1039290, -2.1045895, 1.7514262, -4.3689718, 4.2085180

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5817724
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5817724
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1.0144935, 0.9909928, -1.1746768, 1.2235795, -2.2380729, 2.1656694
1: -6.3876376, 2.4408455, -8.4050074, 2.8732171, -9.2608547, 10.8458529
2: -3.6375842, 2.4044783, -4.3243637, 2.8570030, -6.4945865, 6.7288404
3: -4.1949697, 1.7533797, -5.4014664, 2.1423640, -6.3373327, 7.1548462
4: -2.5870728, 2.0741713, -3.0409644, 2.4777367, -5.0648093, 5.1151357

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5830009
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5850783
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.0273520, 1.0056329, -0.8154888, 0.8182118, -1.8455637, 1.8211217
1: -6.4656796, 2.4740899, -5.7563534, 2.0190501, -8.4847298, 8.2304430
2: -3.6766818, 2.4367743, -3.0316694, 2.0544627, -5.7311444, 5.4684424
3: -4.2514768, 1.7788718, -3.6736217, 1.5062127, -5.7576895, 5.4524927
4: -2.6175458, 2.1039290, -2.0668054, 1.7498134, -4.3673592, 4.1707339

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5815780, upper bound: 3.5813052
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849805, upper bound: 3.5788854
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850264, upper bound: 3.5811740
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.0144935, 0.9909928, -1.1396009, 1.1990455, -2.2135389, 2.1305933
1: -6.3876376, 2.4408455, -8.3805723, 2.7883079, -9.1759453, 10.8214169
2: -3.6375842, 2.4044783, -4.2246008, 2.8196661, -6.4572496, 6.6290789
3: -4.1949697, 1.7533797, -5.3370137, 2.1121480, -6.3071160, 7.0903935
4: -2.5870728, 2.0741713, -2.9356136, 2.4462025, -5.0332756, 5.0097847

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5815780, upper bound: 3.5813069
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5815780, upper bound: 3.5850769
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1396009, 1.1990455, -1.0144935, 0.9909928, -2.1305933, 2.2135389
1: -8.3805723, 2.7883079, -6.3876376, 2.4408455, -10.8214178, 9.1759453
2: -4.2246008, 2.8196661, -3.6375842, 2.4044783, -6.6290789, 6.4572501
3: -5.3370137, 2.1121480, -4.1949697, 1.7533797, -7.0903935, 6.3071160
4: -2.9356136, 2.4462025, -2.5870728, 2.0741713, -5.0097847, 5.0332756

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5843532, upper bound: 3.5849740
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850150, upper bound: 3.5850085
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1396009, 1.1990455, -0.9714246, 0.9586650, -2.0982654, 2.1704702
1: -8.3805723, 2.7883079, -6.3648095, 2.3431084, -10.7236795, 9.1531162
2: -4.2246008, 2.8196661, -3.4794559, 2.3432698, -6.5678706, 6.2991219
3: -5.3370137, 2.1121480, -4.1278048, 1.7133278, -7.0503411, 6.2399511
4: -2.9356136, 2.4462025, -2.4520850, 2.0101967, -4.9458103, 4.8982878

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5843532, upper bound: 3.5849740
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850150, upper bound: 3.5850085
time: 0.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.36 seconds
NS_A1_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5795595, upper bound: 3.5784132
NS_A1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5795596, upper bound: 3.5846709
NS_A1_A1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5775890, upper bound: 3.5784333
NS_A1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5775891, upper bound: 3.5846641
NS_A1_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5780914, upper bound: 3.5782554
NS_A1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5782253, upper bound: 3.5845501
NS_A1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5746802, upper bound: 3.5839230
NS_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5746802, upper bound: 3.5847355
NS_A2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5845501, upper bound: 3.5813524
NS_A2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5845695, upper bound: 3.5781086
NS_A2_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5782554, upper bound: 3.5780914
NS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5845501, upper bound: 3.5782253
NS_A2_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5839230, upper bound: 3.5746802
NS_A2_B1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5839230, upper bound: 3.5775077
NS_A2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5817724
NS_A2_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5817724
NS_A2_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5830009
NS_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5817157, upper bound: 3.5850783
NS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5849805, upper bound: 3.5788854
NS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5850264, upper bound: 3.5811740
NS_A2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5815780, upper bound: 3.5813069
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5815780, upper bound: 3.5850769
NS_A2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5843532, upper bound: 3.5849740
NS_A2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5850150, upper bound: 3.5850085
NS_A2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5843532, upper bound: 3.5849740
NS_A2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.36
Output dim: 4, lower bound: -3.5850150, upper bound: 3.5850085

## BFS NS instance: NS_A1_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4585693, 0.4178070, -1.0625346, 1.0353074, -1.4938767, 1.4803416
1: -3.3594925, 1.1605518, -6.6862850, 2.5479808, -5.9074736, 7.8468370
2: -1.7860143, 1.2129331, -3.8035555, 2.5211487, -4.3071628, 5.0164886
3: -2.1246309, 0.8812460, -4.3945513, 1.8302741, -3.9549050, 5.2757974
4: -1.1309596, 0.9925737, -2.7010381, 2.1730034, -3.3039625, 3.6936119

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5793567, upper bound: 3.5783994
time: 0.48 seconds

## Relational analysis of NS_A1_A1_B2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5795596, upper bound: 3.5821269
time: 0.50 seconds

## Relational analysis of NS_A1_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5795596, upper bound: 3.5846709
time: 0.55 seconds

## BFS NS instance: NS_A1_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.5026631, 0.4498851, -1.0625346, 1.0353074, -1.5379703, 1.5124197
1: -3.3779643, 1.2705357, -6.6862850, 2.5479808, -5.9259448, 7.9568205
2: -1.9132642, 1.2897635, -3.8035555, 2.5211487, -4.4344125, 5.0933189
3: -2.1676135, 0.9355350, -4.3945513, 1.8302741, -3.9978867, 5.3300853
4: -1.2523308, 1.0681732, -2.7010381, 2.1730034, -3.4253333, 3.7692113

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5695988, upper bound: 3.5821417
time: 0.60 seconds

## Relational analysis of NS_A1_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5695988, upper bound: 3.5840887
time: 0.49 seconds

## BFS NS instance: NS_A1_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.5197653, 0.4778241, -1.0315453, 1.0018941, -1.5216594, 1.5093694
1: -3.5102820, 1.2897317, -6.5016499, 2.4752855, -5.9855657, 7.7913809
2: -1.9258542, 1.3158989, -3.6964540, 2.4536235, -4.3794770, 5.0123529
3: -2.2408814, 0.9519234, -4.2689466, 1.7785511, -4.0194325, 5.2208700
4: -1.2813401, 1.0940826, -2.6236300, 2.1092348, -3.3905749, 3.7177119

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5751390, upper bound: 3.5838880
time: 0.45 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5751390, upper bound: 3.5845501
time: 0.55 seconds

## BFS NS instance: NS_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.5684510, 0.5201157, -1.0562047, 1.0277854, -1.5962362, 1.5763202
1: -3.6783888, 1.4118460, -6.6346488, 2.5324912, -6.2108788, 8.0464935
2: -2.0929317, 1.4184625, -3.7815390, 2.5050836, -4.5980153, 5.2000012
3: -2.3671312, 1.0303986, -4.3620949, 1.8176227, -4.1847529, 5.3924932
4: -1.3946905, 1.1875691, -2.6843405, 2.1580725, -3.5527630, 3.8719096

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5744105, upper bound: 3.5802628
time: 0.59 seconds

## Relational analysis of NS_A1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5744105, upper bound: 3.5817955
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1.0829937, 1.0572182, -0.4585693, 0.4178070, -1.5008006, 1.5157876
1: -6.7634592, 2.5928497, -3.3594925, 1.1605518, -7.9240103, 5.9523420
2: -3.8548331, 2.5620289, -1.7860143, 1.2129331, -5.0677662, 4.3480434
3: -4.4495683, 1.8610864, -2.1246309, 0.8812460, -5.3308144, 3.9857173
4: -2.7488444, 2.2132304, -1.1309596, 0.9925737, -3.7414179, 3.3441899

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5781240, upper bound: 3.5795374
time: 0.48 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5781240, upper bound: 3.5812382
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1.0829937, 1.0572182, -0.5026631, 0.4498851, -1.5328788, 1.5598810
1: -6.7634592, 2.5928497, -3.3779643, 1.2705357, -8.0339937, 5.9708128
2: -3.8548331, 2.5620289, -1.9132642, 1.2897635, -5.1445966, 4.4752927
3: -4.4495683, 1.8610864, -2.1676135, 0.9355350, -5.3851032, 4.0286999
4: -2.7488444, 2.2132304, -1.2523308, 1.0681732, -3.8170173, 3.4655607

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5778319, upper bound: 3.5775637
time: 0.44 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5778319, upper bound: 3.5697304
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.0397445, 1.0111041, -0.5197653, 0.4778241, -1.5175686, 1.5308694
1: -6.5398731, 2.4927959, -3.5102820, 1.2897317, -7.8296032, 6.0030780
2: -3.7203512, 2.4718947, -1.9258542, 1.3158989, -5.0362501, 4.3977489
3: -4.2948694, 1.7917624, -2.2408814, 0.9519234, -5.2467928, 4.0326433
4: -2.6426744, 2.1272521, -1.2813401, 1.0940826, -3.7367570, 3.4085922

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5838880, upper bound: 3.5751390
time: 0.69 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5838880, upper bound: 3.5782253
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1746768, 1.2235795, -1.1746768, 1.2235795, -2.3982558, 2.3982561
1: -8.4050074, 2.8732171, -8.4050074, 2.8732171, -11.2782249, 11.2782249
2: -4.3243637, 2.8570030, -4.3243637, 2.8570030, -7.1813660, 7.1813655
3: -5.4014664, 2.1423640, -5.4014664, 2.1423640, -7.5438304, 7.5438304
4: -3.0409644, 2.4777367, -3.0409644, 2.4777367, -5.5187011, 5.5187011

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5813989, upper bound: 3.5845930
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5814629, upper bound: 3.5848308
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -1.0273520, 1.0056329, -0.7812371, 0.7850540, -1.8124059, 1.7868698
1: -6.4656796, 2.4740899, -5.5880322, 1.9425139, -8.4081936, 8.0621223
2: -3.6766818, 2.4367743, -2.9162917, 1.9816000, -5.6582813, 5.3530655
3: -4.2514768, 1.7788718, -3.5537672, 1.4542670, -5.7057438, 5.3326378
4: -2.6175458, 2.1039290, -1.9772143, 1.6851131, -4.3026590, 4.0811434

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5788239
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5788854
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1.0269600, 1.0052434, -0.8402016, 0.8526391, -1.8795990, 1.8454450
1: -6.4641409, 2.4731836, -6.0031509, 2.0796695, -8.5438099, 8.4763346
2: -3.6755106, 2.4359894, -3.1320608, 2.1219366, -5.7974472, 5.5680499
3: -4.2503247, 1.7782835, -3.8195639, 1.5597293, -5.8100538, 5.5978475
4: -2.6165967, 2.1031644, -2.1362066, 1.8189569, -4.4355536, 4.2393713

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5811125
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5811740
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.1746768, 1.2235795, -1.1396009, 1.1990455, -2.3737221, 2.3631797
1: -8.4050074, 2.8732171, -8.3805723, 2.7883079, -11.1933155, 11.2537899
2: -4.3243637, 2.8570030, -4.2246008, 2.8196661, -7.1440287, 7.0816031
3: -5.4014664, 2.1423640, -5.3370137, 2.1121480, -7.5136147, 7.4793768
4: -3.0409644, 2.4777367, -2.9356136, 2.4462025, -5.4871669, 5.4133501

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5803653, upper bound: 3.5816343
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5803747, upper bound: 3.5850240
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.0048838, 1.0414437, -0.9937585, 0.9652315, -1.9701152, 2.0352023
1: -7.2492638, 2.4539721, -6.1896858, 2.3930874, -9.6423512, 8.6436577
2: -3.6965237, 2.4734848, -3.5649517, 2.3497908, -6.0463133, 6.0384364
3: -4.6196918, 1.8447443, -4.0715075, 1.7106615, -6.3303533, 5.9162512
4: -2.5781016, 2.1296673, -2.5322380, 2.0239363, -4.6020379, 4.6619053

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842970, upper bound: 3.5844330
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5843521, upper bound: 3.5849740
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.1250600, 1.1854312, -1.0144935, 0.9909928, -2.1160526, 2.1999245
1: -8.2963867, 2.7542291, -6.3876376, 2.4408455, -10.7372313, 9.1418667
2: -4.1737981, 2.7882361, -3.6375842, 2.4044783, -6.5782747, 6.4258184
3: -5.2779379, 2.0888665, -4.1949697, 1.7533797, -7.0313177, 6.2838354
4: -2.8987038, 2.4198971, -2.5870728, 2.0741713, -4.9728751, 5.0069699

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849588, upper bound: 3.5845648
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850139, upper bound: 3.5850087
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.0048838, 1.0414437, -0.9491435, 0.9306507, -1.9355342, 1.9905872
1: -7.2492638, 2.4539721, -6.1542182, 2.2835574, -9.5328217, 8.6081896
2: -3.6965237, 2.4734848, -3.3892078, 2.2841470, -5.9806681, 5.8626928
3: -4.6196918, 1.8447443, -3.9951735, 1.6650883, -6.2847800, 5.8399177
4: -2.5781016, 2.1296673, -2.3924720, 1.9562614, -4.5343628, 4.5221386

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5780224, upper bound: 3.5832659
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5841152, upper bound: 3.5847659
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.1250600, 1.1854312, -0.9714246, 0.9586650, -2.0837245, 2.1568556
1: -8.2963867, 2.7542291, -6.3648095, 2.3431084, -10.6394930, 9.1190357
2: -4.1737981, 2.7882361, -3.4794559, 2.3432698, -6.5170674, 6.2676907
3: -5.2779379, 2.0888665, -4.1278048, 1.7133278, -6.9912653, 6.2166710
4: -2.8987038, 2.4198971, -2.4520850, 2.0101967, -4.9089003, 4.8719821

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5771298, upper bound: 3.5831912
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848178
time: 0.52 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.20 seconds
NS_A1_A1_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5795596, upper bound: 3.5821269
NS_A1_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5795596, upper bound: 3.5846709
NS_A1_A1_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5695988, upper bound: 3.5821417
NS_A1_A1_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5695988, upper bound: 3.5840887
NS_A1_A2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5751390, upper bound: 3.5838880
NS_A1_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5751390, upper bound: 3.5845501
NS_A1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5744105, upper bound: 3.5802628
NS_A1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5744105, upper bound: 3.5817955
NS_A2_B1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5781240, upper bound: 3.5795374
NS_A2_B1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5781240, upper bound: 3.5812382
NS_A2_B1_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5778319, upper bound: 3.5775637
NS_A2_B1_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5778319, upper bound: 3.5697304
NS_A2_B1_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5838880, upper bound: 3.5751390
NS_A2_B1_B2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5838880, upper bound: 3.5782253
NS_A2_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5813989, upper bound: 3.5845930
NS_A2_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5814629, upper bound: 3.5848308
NS_A2_B2_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5788239
NS_A2_B2_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5788854
NS_A2_B2_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5811125
NS_A2_B2_A1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5843817, upper bound: 3.5811740
NS_A2_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5803653, upper bound: 3.5816343
NS_A2_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5803747, upper bound: 3.5850240
NS_A2_B2_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5842970, upper bound: 3.5844330
NS_A2_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5843521, upper bound: 3.5849740
NS_A2_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5849588, upper bound: 3.5845648
NS_A2_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5850139, upper bound: 3.5850087
NS_A2_B2_A2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5780224, upper bound: 3.5832659
NS_A2_B2_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5841152, upper bound: 3.5847659
NS_A2_B2_A2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5771298, upper bound: 3.5831912
NS_A2_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.20
Output dim: 4, lower bound: -3.5848400, upper bound: 3.5848178

## BFS NS instance: NS_A1_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.4585693, 0.4178070, -1.0898525, 1.0747662, -1.5333354, 1.5076594
1: -3.3594925, 1.1605518, -6.9608240, 2.6146796, -5.9741721, 8.1213760
2: -1.7860143, 1.2129331, -3.9044423, 2.5868773, -4.3728909, 5.1173754
3: -2.1246309, 0.8812460, -4.5600891, 1.8918591, -4.0164900, 5.4413352
4: -1.1309596, 0.9925737, -2.7644236, 2.2382236, -3.3691831, 3.7569971

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.5041122, 0.4607088, -1.0315453, 1.0018941, -1.5060064, 1.4922540
1: -3.4480660, 1.2542088, -6.5016499, 2.4752855, -5.9233508, 7.7558575
2: -1.8823076, 1.2855999, -3.6964540, 2.4536235, -4.3359303, 4.9820538
3: -2.1972117, 0.9300270, -4.2689466, 1.7785511, -3.9757628, 5.1989737
4: -1.2380376, 1.0634465, -2.6236300, 2.1092348, -3.3472724, 3.6870761

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5748794, upper bound: 3.5795484
time: 0.50 seconds

## Relational analysis of NS_A1_A2_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5748794, upper bound: 3.5815573
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1.1680543, 1.2167004, -1.1746768, 1.2235795, -2.3916337, 2.3913772
1: -8.3758125, 2.8588417, -8.4050074, 2.8732171, -11.2490292, 11.2638493
2: -4.3045883, 2.8440266, -4.3243637, 2.8570030, -7.1615901, 7.1683898
3: -5.3807573, 2.1324902, -5.4014664, 2.1423640, -7.5231209, 7.5339565
4: -3.0245042, 2.4643795, -3.0409644, 2.4777367, -5.5022411, 5.5053439

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5819060, upper bound: 3.5834798
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842774, upper bound: 3.5842202
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5842774, upper bound: 3.5842202
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.1543911, 1.2086012, -1.1741550, 1.2231010, -2.3774920, 2.3827560
1: -8.3168993, 2.8258462, -8.4022999, 2.8720429, -11.1889420, 11.2281456
2: -4.2552066, 2.8197148, -4.3225927, 2.8558912, -7.1110978, 7.1423073
3: -5.3355007, 2.1166425, -5.3995419, 2.1415863, -7.4770870, 7.5161843
4: -2.9883013, 2.4443679, -3.0396357, 2.4767661, -5.4650674, 5.4840031

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848092, upper bound: 3.5845632
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848569, upper bound: 3.5842793
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848569, upper bound: 3.5848308
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1746768, 1.2235795, -1.1546327, 1.2303580, -2.4050348, 2.3782117
1: -8.4050074, 2.8732171, -8.6585398, 2.8364723, -11.2414799, 11.5317554
2: -4.3243637, 2.8570030, -4.3077645, 2.8787565, -7.2031202, 7.1647658
3: -5.4014664, 2.1423640, -5.5003366, 2.1697199, -7.5711861, 7.6427007
4: -3.0409644, 2.4777367, -2.9838982, 2.5018837, -5.5428481, 5.4616346

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845671, upper bound: 3.5841096
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845671, upper bound: 3.5850240
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.0043812, 1.0409350, -1.0141900, 0.9964507, -2.0008314, 2.0551250
1: -7.2469211, 2.4528680, -6.4488573, 2.4421802, -9.6891003, 8.9017258
2: -3.6950538, 2.4724531, -3.6426477, 2.4129181, -6.1079721, 6.1151009
3: -4.6180396, 1.8439829, -4.2225165, 1.7601776, -6.3782172, 6.0664992
4: -2.5768557, 2.1286840, -2.5898273, 2.0847797, -4.6616354, 4.7185116

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5834367, upper bound: 3.5839836
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.1250600, 1.1854312, -0.9604172, 0.9392071, -2.0642672, 2.1458478
1: -8.2963867, 2.7542291, -6.1543775, 2.3206921, -10.6170788, 8.9086065
2: -4.1737981, 2.7882361, -3.4646647, 2.2945170, -6.4683151, 6.2529001
3: -5.2779379, 2.0888665, -4.0242820, 1.6755333, -6.9534712, 6.1131477
4: -2.8987038, 2.4198971, -2.4499452, 1.9714994, -4.8702030, 4.8698416

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5792782, upper bound: 3.5826251
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5792782, upper bound: 3.5845648
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.1246219, 1.1849763, -1.0366288, 1.0246944, -2.1493163, 2.2216048
1: -8.2943087, 2.7532430, -6.6566601, 2.4972024, -10.7915115, 9.4099026
2: -4.1724854, 2.7873275, -3.7202590, 2.4712701, -6.6437545, 6.5075860
3: -5.2764688, 2.0881939, -4.3527389, 1.8082861, -7.0847549, 6.4409323
4: -2.8976126, 2.4190259, -2.6483195, 2.1390946, -5.0367069, 5.0673456

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5793017, upper bound: 3.5826251
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5793017, upper bound: 3.5850087
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.9876192, 1.0232985, -0.9491435, 0.9306507, -1.9182699, 1.9724419
1: -7.1647511, 2.4135404, -6.1542182, 2.2835574, -9.4483070, 8.5677567
2: -3.6398017, 2.4363956, -3.3892078, 2.2841470, -5.9239478, 5.8256035
3: -4.5596161, 1.8169737, -3.9951735, 1.6650883, -6.2247033, 5.8121467
4: -2.5302417, 2.0939023, -2.3924720, 1.9562614, -4.4865026, 4.4863739

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5839101, upper bound: 3.5839857
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5839101, upper bound: 3.5847659
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.1077844, 1.1673827, -0.9714246, 0.9586650, -2.0664494, 2.1388073
1: -8.2163334, 2.7146485, -6.3648095, 2.3431084, -10.5594406, 9.0794563
2: -4.1185603, 2.7517543, -3.4794559, 2.3432698, -6.4618301, 6.2312098
3: -5.2204242, 2.0617173, -4.1278048, 1.7133278, -6.9337521, 6.1895218
4: -2.8517923, 2.3837020, -2.4520850, 2.0101967, -4.8619890, 4.8357868

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5839103, upper bound: 3.5839282
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5839103, upper bound: 3.5848097
time: 0.61 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.67 seconds
NS_A1_A2_B2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5748794, upper bound: 3.5795484
NS_A1_A2_B2_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5748794, upper bound: 3.5815573
NS_A2_B2_A1_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5842774, upper bound: 3.5842202
NS_A2_B2_A1_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5842774, upper bound: 3.5842202
NS_A2_B2_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5848569, upper bound: 3.5842793
NS_A2_B2_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5848569, upper bound: 3.5848308
NS_A2_B2_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5845671, upper bound: 3.5841096
NS_A2_B2_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5845671, upper bound: 3.5850240
NS_A2_B2_A2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5792782, upper bound: 3.5826251
NS_A2_B2_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5792782, upper bound: 3.5845648
NS_A2_B2_A2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5793017, upper bound: 3.5826251
NS_A2_B2_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5793017, upper bound: 3.5850087
NS_A2_B2_A2_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5839101, upper bound: 3.5839857
NS_A2_B2_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5839101, upper bound: 3.5847659
NS_A2_B2_A2_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5839103, upper bound: 3.5839282
NS_A2_B2_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.67
Output dim: 4, lower bound: -3.5839103, upper bound: 3.5848097

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1.1543911, 1.2086012, -1.1680543, 1.2167004, -2.3710914, 2.3766553
1: -8.3168993, 2.8258462, -8.3758125, 2.8588417, -11.1757412, 11.2016582
2: -4.2552066, 2.8197148, -4.3045883, 2.8440266, -7.0992327, 7.1243029
3: -5.3355007, 2.1166425, -5.3807573, 2.1324902, -7.4679904, 7.4973993
4: -2.9883013, 2.4443679, -3.0245042, 2.4643795, -5.4526806, 5.4688721

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846883, upper bound: 3.5830598
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847766, upper bound: 3.5841025
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1543911, 1.2086012, -1.1543911, 1.2086012, -2.3629920, 2.3629916
1: -8.3168993, 2.8258462, -8.3168993, 2.8258462, -11.1427450, 11.1427450
2: -4.2552066, 2.8197148, -4.2552066, 2.8197148, -7.0749211, 7.0749216
3: -5.3355007, 2.1166425, -5.3355007, 2.1166425, -7.4521432, 7.4521432
4: -2.9883013, 2.4443679, -2.9883013, 2.4443679, -5.4326687, 5.4326687

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846884, upper bound: 3.5830598
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847766, upper bound: 3.5847379
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.0548408, 1.1092948, -1.1546327, 1.2303580, -2.2851987, 2.2639272
1: -7.7648439, 2.6008685, -8.6585398, 2.8364723, -10.6013165, 11.2594061
2: -3.9206018, 2.6038184, -4.3077645, 2.8787565, -6.7993584, 6.9115825
3: -4.9442225, 1.9587140, -5.5003366, 2.1697199, -7.1139421, 7.4590507
4: -2.7252047, 2.2597320, -2.9838982, 2.5018837, -5.2270880, 5.2436295

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.1992341, 1.2571125, -1.1546327, 1.2303580, -2.4295921, 2.4117451
1: -8.6479301, 2.9330635, -8.6585398, 2.8364723, -11.4844017, 11.5916023
2: -4.4166646, 2.9268548, -4.3077645, 2.8787565, -7.2954206, 7.2346187
3: -5.5491881, 2.2009287, -5.5003366, 2.1697199, -7.7189083, 7.7012653
4: -3.1024356, 2.5463421, -2.9838982, 2.5018837, -5.6043196, 5.5302391

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.1509404, 1.2231280, -0.9604172, 0.9392071, -2.0901475, 2.1835451
1: -8.5587692, 2.8176200, -6.1543775, 2.3206921, -10.8794613, 8.9719973
2: -4.2771101, 2.8624699, -3.4646647, 2.2945170, -6.5716267, 6.3271341
3: -5.4312677, 2.1465538, -4.0242820, 1.6755333, -7.1068010, 6.1708341
4: -2.9724481, 2.4952433, -2.4499452, 1.9714994, -4.9439473, 4.9451885

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## BFS NS instance: NS_A2_B2_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.1509404, 1.2231280, -1.0366288, 1.0246944, -2.1756349, 2.2597568
1: -8.5587692, 2.8176200, -6.6566601, 2.4972024, -11.0559702, 9.4742804
2: -4.2771101, 2.8624699, -3.7202590, 2.4712701, -6.7483797, 6.5827289
3: -5.4312677, 2.1465538, -4.3527389, 1.8082861, -7.2395539, 6.4992924
4: -2.9724481, 2.4952433, -2.6483195, 2.1390946, -5.1115427, 5.1435628

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A2_B2_A2_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9876192, 1.0232985, -0.9535132, 0.9420314, -1.9296505, 1.9768114
1: -7.1647511, 2.4135404, -6.2701802, 2.3044097, -9.4691601, 8.6837206
2: -3.6398017, 2.4363956, -3.4218030, 2.3070021, -5.9468040, 5.8581982
3: -4.5596161, 1.8169737, -4.0610456, 1.6861219, -6.2457371, 5.8780193
4: -2.5302417, 2.0939023, -2.4082758, 1.9777622, -4.5080032, 4.5021782

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5833429, upper bound: 3.5844688
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5831649, upper bound: 3.5842854
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.1077844, 1.1673827, -0.9535132, 0.9420314, -2.0498157, 2.1208959
1: -8.2163334, 2.7146485, -6.2701802, 2.3044097, -10.5207434, 8.9848280
2: -4.1185603, 2.7517543, -3.4218030, 2.3070021, -6.4255624, 6.1735563
3: -5.2204242, 2.0617173, -4.0610456, 1.6861219, -6.9065452, 6.1227627
4: -2.8517923, 2.3837020, -2.4082758, 1.9777622, -4.8295546, 4.7919779

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5831736, upper bound: 3.5841467
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5831657, upper bound: 3.5844776
time: 0.54 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.68 seconds
NS_A2_B2_A1_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5846883, upper bound: 3.5830598
NS_A2_B2_A1_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5847766, upper bound: 3.5841025
NS_A2_B2_A1_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5846884, upper bound: 3.5830598
NS_A2_B2_A1_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5847766, upper bound: 3.5847379
NS_A2_B2_A2_A2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5833429, upper bound: 3.5844688
NS_A2_B2_A2_A2_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5831649, upper bound: 3.5842854
NS_A2_B2_A2_A2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5831736, upper bound: 3.5841467
NS_A2_B2_A2_A2_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.68
Output dim: 4, lower bound: -3.5831657, upper bound: 3.5844776

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.1543911, 1.2086012, -1.0481580, 1.1021967, -2.2565877, 2.2567589
1: -8.3168993, 2.8258462, -7.7338257, 2.5861380, -10.9030371, 10.5596714
2: -4.2552066, 2.8197148, -3.9004059, 2.5902092, -6.8454161, 6.7201204
3: -5.3355007, 2.1166425, -4.9227819, 1.9484457, -7.2839465, 7.0394230
4: -2.9883013, 2.4443679, -2.7086835, 2.2459922, -5.2342935, 5.1530504

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844395, upper bound: 3.5829647
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844395, upper bound: 3.5830598
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.1543911, 1.2086012, -1.1919879, 1.2489727, -2.4033637, 2.4005890
1: -8.3168993, 2.8258462, -8.6136522, 2.9165590, -11.2334585, 11.4394979
2: -4.2552066, 2.8197148, -4.3940277, 2.9118948, -7.1671009, 7.2137423
3: -5.3355007, 2.1166425, -5.5247860, 2.1893816, -7.5248823, 7.6414285
4: -2.9883013, 2.4443679, -3.0840321, 2.5311277, -5.5194292, 5.5283999

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845278, upper bound: 3.5837508
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845278, upper bound: 3.5841025
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1.1543911, 1.2086012, -1.0330921, 1.0923276, -2.2467186, 2.2416930
1: -8.3168993, 2.8258462, -7.6646342, 2.5498228, -10.8667212, 10.4904804
2: -4.2552066, 2.8197148, -3.8463831, 2.5584176, -6.8136239, 6.6660976
3: -5.3355007, 2.1166425, -4.8724470, 1.9301119, -7.2656126, 6.9890895
4: -2.9883013, 2.4443679, -2.6673665, 2.2213893, -5.2096906, 5.1117334

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5843777
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5844441
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1.1543911, 1.2086012, -1.1800718, 1.2432634, -2.3976543, 2.3886721
1: -8.3168993, 2.8258462, -8.5601511, 2.8877528, -11.2046509, 11.3859978
2: -4.2552066, 2.8197148, -4.3495908, 2.8903055, -7.1455107, 7.1693053
3: -5.3355007, 2.1166425, -5.4840117, 2.1756845, -7.5111837, 7.6006541
4: -2.9883013, 2.4443679, -3.0504618, 2.5169337, -5.5052347, 5.4948292

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845359, upper bound: 3.5843777
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5845359, upper bound: 3.5847379
time: 0.59 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 2.23 seconds
NS_A2_B2_A1_B1_B2_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5844395, upper bound: 3.5829647
NS_A2_B2_A1_B1_B2_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5844395, upper bound: 3.5830598
NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5845278, upper bound: 3.5837508
NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5845278, upper bound: 3.5841025
NS_A2_B2_A1_B1_B2_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5843777
NS_A2_B2_A1_B1_B2_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5844441
NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5845359, upper bound: 3.5843777
NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.23
Output dim: 4, lower bound: -3.5845359, upper bound: 3.5847379

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1.0330921, 1.0923276, -1.1919879, 1.2489727, -2.2820649, 2.2843156
1: -7.6646342, 2.5498228, -8.6136522, 2.9165590, -10.5811930, 11.1634750
2: -3.8463831, 2.5584176, -4.3940277, 2.9118948, -6.7582779, 6.9524450
3: -4.8724470, 1.9301119, -5.5247860, 2.1893816, -7.0618286, 7.4548979
4: -2.6673665, 2.2213893, -3.0840321, 2.5311277, -5.1984940, 5.3054214

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1800718, 1.2432634, -1.1919879, 1.2489727, -2.4290445, 2.4352512
1: -8.5601511, 2.8877528, -8.6136522, 2.9165590, -11.4767103, 11.5014029
2: -4.3495908, 2.8903055, -4.3940277, 2.9118948, -7.2614851, 7.2843332
3: -5.4840117, 2.1756845, -5.5247860, 2.1893816, -7.6733932, 7.7004690
4: -3.0504618, 2.5169337, -3.0840321, 2.5311277, -5.5815897, 5.6009655

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.0330921, 1.0923276, -1.1800718, 1.2432634, -2.2763553, 2.2723994
1: -7.6646342, 2.5498228, -8.5601511, 2.8877528, -10.5523863, 11.1099739
2: -3.8463831, 2.5584176, -4.3495908, 2.8903055, -6.7366867, 6.9080081
3: -4.8724470, 1.9301119, -5.4840117, 2.1756845, -7.0481315, 7.4141235
4: -2.6673665, 2.2213893, -3.0504618, 2.5169337, -5.1843004, 5.2718511

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

## BFS NS instance: NS_A2_B2_A1_B1_B2_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.1800718, 1.2432634, -1.1800718, 1.2432634, -2.4233351, 2.4233348
1: -8.5601511, 2.8877528, -8.5601511, 2.8877528, -11.4479027, 11.4479027
2: -4.3495908, 2.8903055, -4.3495908, 2.8903055, -7.2398949, 7.2398953
3: -5.4840117, 2.1756845, -5.4840117, 2.1756845, -7.6596951, 7.6596947
4: -3.0504618, 2.5169337, -3.0504618, 2.5169337, -5.5673952, 5.5673952

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.47 + 173.11 = 175.58 seconds
