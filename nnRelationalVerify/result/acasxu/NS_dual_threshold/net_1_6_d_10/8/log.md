## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 17.96064755562


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.7510505, 7.4405704, -2.7510505, 7.4405704, -10.1916208, 10.1916208)
1: (-7.5991325, 11.2530985, -7.5991325, 11.2530985, -18.8522301, 18.8522301)
2: (-4.7880874, 10.3202887, -4.7880874, 10.3202887, -15.1083755, 15.1083755)
3: (-8.5250254, 12.5358915, -8.5250254, 12.5358915, -21.0609169, 21.0609169)
4: (-5.5422077, 12.5079746, -5.5422077, 12.5079746, -18.0501823, 18.0501823)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 1.75 = 2.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -17.9624438, upper bound: 17.9624438

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9623897, upper bound: 17.9612123
time: 0.48 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9615450
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 3, lower bound: -17.9623897, upper bound: 17.9612123
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 3, lower bound: -17.9615450, upper bound: 17.9615450

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.7510505, 7.4405704, -9.3307257, 7.8515010
1: -5.1917362, 7.8175325, -7.5991325, 11.2530985, -16.4448357, 15.4166651
2: -3.2490828, 7.0843124, -4.7880874, 10.3202887, -13.5693712, 11.8724003
3: -5.8050518, 8.6047878, -8.5250254, 12.5358915, -18.3409424, 17.1298141
4: -3.7534659, 8.5734529, -5.5422077, 12.5079746, -16.2614403, 14.1156607

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
time: 0.69 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
time: 0.65 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.4610372, 6.5926552, -2.7510505, 7.4405704, -9.9016075, 9.3437052
1: -6.7975974, 10.0043488, -7.5991325, 11.2530985, -18.0506954, 17.6034813
2: -4.2729445, 9.1532860, -4.7880874, 10.3202887, -14.5932331, 13.9413729
3: -7.6107635, 11.1077843, -8.5250254, 12.5358915, -20.1466522, 19.6328087
4: -4.9338398, 11.0912857, -5.5422077, 12.5079746, -17.4418144, 16.6334934

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
time: 0.50 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.06 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9612123
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.8901551, 5.1004510, -6.9906063, 6.9906063
1: -5.1917362, 7.8175325, -5.1917362, 7.8175325, -13.0092678, 13.0092678
2: -3.2490828, 7.0843124, -3.2490828, 7.0843124, -10.3333950, 10.3333950
3: -5.8050518, 8.6047878, -5.8050518, 8.6047878, -14.4098396, 14.4098396
4: -3.7534659, 8.5734529, -3.7534659, 8.5734529, -12.3269186, 12.3269186

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522597
time: 0.52 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
time: 0.69 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.4610372, 6.5926552, -8.4828100, 7.5614882
1: -5.1917362, 7.8175325, -6.7975974, 10.0043488, -15.1960831, 14.6151295
2: -3.2490828, 7.0843124, -4.2729445, 9.1532860, -12.4023685, 11.3572569
3: -5.8050518, 8.6047878, -7.6107635, 11.1077843, -16.9128361, 16.2155476
4: -3.7534659, 8.5734529, -4.9338398, 11.0912857, -14.8447514, 13.5072927

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522597
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
time: 0.53 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.4610372, 6.5926552, -1.8901551, 5.1004510, -7.5614882, 8.4828100
1: -6.7975974, 10.0043488, -5.1917362, 7.8175325, -14.6151295, 15.1960831
2: -4.2729445, 9.1532860, -3.2490828, 7.0843124, -11.3572569, 12.4023685
3: -7.6107635, 11.1077843, -5.8050518, 8.6047878, -16.2155476, 16.9128361
4: -4.9338398, 11.0912857, -3.7534659, 8.5734529, -13.5072927, 14.8447514

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450
time: 0.56 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.4610372, 6.5926552, -2.4610372, 6.5926552, -9.0536909, 9.0536909
1: -6.7975974, 10.0043488, -6.7975974, 10.0043488, -16.8019466, 16.8019466
2: -4.2729445, 9.1532860, -4.2729445, 9.1532860, -13.4262304, 13.4262304
3: -7.6107635, 11.1077843, -7.6107635, 11.1077843, -18.7185440, 18.7185440
4: -4.9338398, 11.0912857, -4.9338398, 11.0912857, -16.0251255, 16.0251255

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9304902, upper bound: 17.9604256
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.02 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522597
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9616311, upper bound: 17.9522597
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9618502, upper bound: 17.9611951
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9304902, upper bound: 17.9604256
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.02
Output dim: 3, lower bound: -17.9612123, upper bound: 17.9615450

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.6415975, 4.4590487, -6.3492041, 6.7420483
1: -5.1917362, 7.8175325, -4.4953952, 6.8639817, -12.0557156, 12.3129272
2: -3.2490828, 7.0843124, -2.8065214, 6.1872668, -9.4363480, 9.8908339
3: -5.8050518, 8.6047878, -5.0186200, 7.5167313, -13.3217831, 13.6234074
4: -3.7534659, 8.5734529, -3.2377799, 7.4721994, -11.2256651, 11.8112307

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618155
time: 0.85 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618502
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.7814304, 4.7954335, -6.6855888, 6.8818812
1: -5.1917362, 7.8175325, -4.8894196, 7.3645873, -12.5563221, 12.7069521
2: -3.2490828, 7.0843124, -3.0549450, 6.6544957, -9.9035769, 10.1392565
3: -5.8050518, 8.6047878, -5.4630013, 8.0912886, -13.8963404, 14.0677872
4: -3.7534659, 8.5734529, -3.5216219, 8.0479536, -11.8014193, 12.0950747

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603382, upper bound: 17.9546188
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9618203, upper bound: 17.9618203
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.0445442, 5.4841022, -7.3742571, 7.1449952
1: -5.1917362, 7.8175325, -5.6337681, 8.3748035, -13.5665379, 13.4513006
2: -3.2490828, 7.0843124, -3.5288885, 7.6191778, -10.8682585, 10.6132002
3: -5.8050518, 8.6047878, -6.3012056, 9.2464848, -15.0515366, 14.9059935
4: -3.7534659, 8.5734529, -4.0707402, 9.2292709, -12.9827366, 12.6441927

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9481172
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9515824
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.3734298, 6.3504786, -8.2406340, 7.4738808
1: -5.1917362, 7.8175325, -6.5536866, 9.6473646, -14.8390989, 14.3712196
2: -3.2490828, 7.0843124, -4.1152163, 8.8181734, -12.0672550, 11.1995277
3: -5.8050518, 8.6047878, -7.3351488, 10.6968727, -16.5019245, 15.9399366
4: -3.7534659, 8.5734529, -4.7487473, 10.6886234, -14.4420891, 13.3221998

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9576771
time: 1.21 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9619985, upper bound: 17.9610416
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.0445442, 5.4841022, -1.8901551, 5.1004510, -7.1449952, 7.3742571
1: -5.6337681, 8.3748035, -5.1917362, 7.8175325, -13.4512997, 13.5665379
2: -3.5288885, 7.6191778, -3.2490828, 7.0843124, -10.6132011, 10.8682585
3: -6.3012056, 9.2464848, -5.8050518, 8.6047878, -14.9059935, 15.0515366
4: -4.0707402, 9.2292709, -3.7534659, 8.5734529, -12.6441927, 12.9827366

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9481172, upper bound: 17.9613381
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515824, upper bound: 17.9614295
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.3734298, 6.3504786, -1.8901551, 5.1004510, -7.4738808, 8.2406340
1: -6.5536866, 9.6473646, -5.1917362, 7.8175325, -14.3712196, 14.8390989
2: -4.1152163, 8.8181734, -3.2490828, 7.0843124, -11.1995277, 12.0672550
3: -7.3351488, 10.6968727, -5.8050518, 8.6047878, -15.9399347, 16.5019245
4: -4.7487473, 10.6886234, -3.7534659, 8.5734529, -13.3221998, 14.4420891

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9618297
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9619985
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.4600651, 6.5900831, -2.4610372, 6.5926552, -9.0527191, 9.0511198
1: -6.7949347, 10.0005121, -6.7975974, 10.0043488, -16.7992840, 16.7981091
2: -4.2712293, 9.1497383, -4.2729445, 9.1532860, -13.4245148, 13.4226828
3: -7.6077557, 11.1034203, -7.6107635, 11.1077843, -18.7155399, 18.7141819
4: -4.9318337, 11.0870132, -4.9338398, 11.0912857, -16.0231190, 16.0208530

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.21 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618155
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9618155, upper bound: 17.9618502
NS_A1_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9603382, upper bound: 17.9546188
NS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9618203, upper bound: 17.9618203
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9481172
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9515824
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9613381, upper bound: 17.9576771
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9619985, upper bound: 17.9610416
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9481172, upper bound: 17.9613381
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9515824, upper bound: 17.9614295
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9618297
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9619985
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9522597, upper bound: 17.9609760
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.21
Output dim: 3, lower bound: -17.9611951, upper bound: 17.9615450

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -1.6415975, 4.4590487, -6.1006460, 6.1006460
1: -4.4953952, 6.8639817, -4.4953952, 6.8639817, -11.3593760, 11.3593750
2: -2.8065214, 6.1872668, -2.8065214, 6.1872668, -8.9937868, 8.9937859
3: -5.0186200, 7.5167313, -5.0186200, 7.5167313, -12.5353508, 12.5353508
4: -3.2377799, 7.4721994, -3.2377799, 7.4721994, -10.7099771, 10.7099771

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9616664
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9617330
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -1.6415975, 4.4590487, -6.2404790, 6.4370308
1: -4.8894196, 7.3645873, -4.4953952, 6.8639817, -11.7534008, 11.8599806
2: -3.0549450, 6.6544957, -2.8065214, 6.1872668, -9.2422085, 9.4610167
3: -5.4630013, 8.0912886, -5.0186200, 7.5167313, -12.9797287, 13.1099091
4: -3.5216219, 8.0479536, -3.2377799, 7.4721994, -10.9938202, 11.2857304

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9586849
time: 0.53 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9617676
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.7559581, 4.7282553, -6.6184101, 6.8564091
1: -5.1917362, 7.8175325, -4.8172703, 7.2626877, -12.4544220, 12.6348028
2: -3.2490828, 7.0843124, -3.0090370, 6.5592184, -9.8083000, 10.0933485
3: -5.8050518, 8.6047878, -5.3817005, 7.9752512, -13.7803030, 13.9864883
4: -3.7534659, 8.5734529, -3.4683015, 7.9320216, -11.6854877, 12.0417528

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B2_B2_B1

### Relational analysis result of NS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615477, upper bound: 17.9582737
time: 0.53 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616141, upper bound: 17.9616141
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.8538499, 4.9839005, -6.8740559, 6.9543009
1: -5.1917362, 7.8175325, -5.0963554, 7.6366177, -12.8283529, 12.9138880
2: -3.2490828, 7.0843124, -3.2001376, 6.9270759, -10.1761580, 10.2844505
3: -5.8050518, 8.6047878, -5.7014980, 8.4203701, -14.2254219, 14.3062859
4: -3.7534659, 8.5734529, -3.6966069, 8.3848782, -12.1383438, 12.2700596

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9480508
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9480508
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.0258384, 5.4558244, -7.3459797, 7.1262894
1: -5.1917362, 7.8175325, -5.5796051, 8.3446941, -13.5364294, 13.3971376
2: -3.2490828, 7.0843124, -3.5107706, 7.5853653, -10.8344479, 10.5950832
3: -5.8050518, 8.6047878, -6.2471113, 9.2124949, -15.0175467, 14.8518991
4: -3.7534659, 8.5734529, -4.0569630, 9.1839800, -12.9374456, 12.6304159

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9515478
time: 0.54 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9515824
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.1817567, 5.8508301, -7.7409849, 7.2822075
1: -5.1917362, 7.8175325, -6.0132799, 8.9072704, -14.0990057, 13.8308125
2: -3.2490828, 7.0843124, -3.7834954, 8.1247444, -11.3738270, 10.8678074
3: -5.8050518, 8.6047878, -6.7310286, 9.8678417, -15.6728935, 15.3358154
4: -3.7534659, 8.5734529, -4.3695273, 9.8453188, -13.5987844, 12.9429798

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613025, upper bound: 17.9545587
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9617952, upper bound: 17.9576771
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.3526335, 6.3142529, -8.2044077, 7.4530840
1: -5.1917362, 7.8175325, -6.4942160, 9.6076832, -14.7994184, 14.3117485
2: -3.2490828, 7.0843124, -4.0940924, 8.7726021, -12.0216846, 11.1784048
3: -5.8050518, 8.6047878, -7.2752800, 10.6551113, -16.4601631, 15.8800669
4: -3.7534659, 8.5734529, -4.7291746, 10.6296968, -14.3831625, 13.3026276

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9586940, upper bound: 17.9609752
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9586940, upper bound: 17.9610416
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.8901551, 5.1004510, -6.9543009, 6.8740559
1: -5.0963554, 7.6366177, -5.1917362, 7.8175325, -12.9138880, 12.8283529
2: -3.2001376, 6.9270759, -3.2490828, 7.0843124, -10.2844505, 10.1761580
3: -5.7014980, 8.4203701, -5.8050518, 8.6047878, -14.3062859, 14.2254219
4: -3.6966069, 8.3848782, -3.7534659, 8.5734529, -12.2700596, 12.1383438

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9580337
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9613381
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.8901551, 5.1004510, -7.1262894, 7.3459797
1: -5.5796051, 8.3446941, -5.1917362, 7.8175325, -13.3971376, 13.5364294
2: -3.5107706, 7.5853653, -3.2490828, 7.0843124, -10.5950832, 10.8344479
3: -6.2471113, 9.2124949, -5.8050518, 8.6047878, -14.8518991, 15.0175467
4: -4.0569630, 9.1839800, -3.7534659, 8.5734529, -12.6304159, 12.9374456

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9614295
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515478, upper bound: 17.9614295
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2.1817567, 5.8508301, -1.8901551, 5.1004510, -7.2822075, 7.7409849
1: -6.0132799, 8.9072704, -5.1917362, 7.8175325, -13.8308125, 14.0990057
2: -3.7834954, 8.1247444, -3.2490828, 7.0843124, -10.8678074, 11.3738270
3: -6.7310286, 9.8678417, -5.8050518, 8.6047878, -15.3358164, 15.6728935
4: -4.3695273, 9.8453188, -3.7534659, 8.5734529, -12.9429798, 13.5987844

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9545587, upper bound: 17.9613025
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9617952
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.8901551, 5.1004510, -7.4530840, 8.2044077
1: -6.4942160, 9.6076832, -5.1917362, 7.8175325, -14.3117485, 14.7994184
2: -4.0940924, 8.7726021, -3.2490828, 7.0843124, -11.1784048, 12.0216846
3: -7.2752800, 10.6551113, -5.8050518, 8.6047878, -15.8800650, 16.4601631
4: -4.7291746, 10.6296968, -3.7534659, 8.5734529, -13.3026276, 14.3831625

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9586940
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9619985
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2.0436969, 5.4818344, -2.4610372, 6.5926552, -8.6363525, 7.9428716
1: -5.6314449, 8.3714676, -6.7975974, 10.0043488, -15.6357927, 15.1690655
2: -3.5274024, 7.6160297, -4.2729445, 9.1532860, -12.6806889, 11.8889713
3: -6.2986040, 9.2426901, -7.6107635, 11.1077843, -17.4063873, 16.8534508
4: -4.0690117, 9.2254801, -4.9338398, 11.0912857, -15.1602964, 14.1593199

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9481172, upper bound: 17.9607312
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515824, upper bound: 17.9608225
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2.3725164, 6.3480577, -2.4610372, 6.5926552, -8.9651718, 8.8090944
1: -6.5511780, 9.6438160, -6.7975974, 10.0043488, -16.5555267, 16.4414120
2: -4.1136031, 8.8148260, -4.2729445, 9.1532860, -13.2668877, 13.0877705
3: -7.3323154, 10.6928082, -7.6107635, 11.1077843, -18.4400978, 18.3035679
4: -4.7468591, 10.6845999, -4.9338398, 11.0912857, -15.8381443, 15.6184397

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9612227
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9613915
time: 0.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.12 seconds
NS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9616664
NS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9617330
NS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9615473, upper bound: 17.9586849
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9616139, upper bound: 17.9617676
NS_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9615477, upper bound: 17.9582737
NS_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9616141, upper bound: 17.9616141
NS_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9480508
NS_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9480508
NS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9515478
NS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9614295, upper bound: 17.9515824
NS_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9613025, upper bound: 17.9545587
NS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9617952, upper bound: 17.9576771
NS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9586940, upper bound: 17.9609752
NS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9586940, upper bound: 17.9610416
NS_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9580337
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9613381
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9614295
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9515478, upper bound: 17.9614295
NS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9545587, upper bound: 17.9613025
NS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9617952
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9586940
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9619985
NS_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9481172, upper bound: 17.9607312
NS_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9515824, upper bound: 17.9608225
NS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9612227
NS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.12
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9613915

## BFS NS instance: NS_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1.4669125, 4.0071440, -1.6415975, 4.4590487, -5.9259615, 5.6487412
1: -4.0064602, 6.2006745, -4.4953952, 6.8639817, -10.8704405, 10.6960697
2: -2.5055699, 5.5651913, -2.8065214, 6.1872668, -8.6928368, 8.3717117
3: -4.4741712, 6.7811985, -5.0186200, 7.5167313, -11.9909010, 11.7998180
4: -2.8958118, 6.7110949, -3.2377799, 7.4721994, -10.3680115, 9.9488745

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9585836, upper bound: 17.9585836
time: 0.73 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9585836, upper bound: 17.9616664
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1.6265631, 4.4408922, -1.6415975, 4.4590487, -6.0856118, 6.0824900
1: -4.4537382, 6.8536711, -4.4953952, 6.8639817, -11.3177195, 11.3490658
2: -2.7905805, 6.1733203, -2.8065214, 6.1872668, -8.9778452, 8.9798412
3: -4.9789329, 7.5049791, -5.0186200, 7.5167313, -12.4956636, 12.5235996
4: -3.2266197, 7.4491405, -3.2377799, 7.4721994, -10.6988182, 10.6869183

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616664, upper bound: 17.9586503
time: 0.52 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616664, upper bound: 17.9617330
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -1.4669125, 4.0071440, -5.7885728, 6.2623453
1: -4.8894196, 7.3645873, -4.0064602, 6.2006745, -11.0900936, 11.3710470
2: -3.0549450, 6.6544957, -2.5055699, 5.5651913, -8.6201353, 9.1600657
3: -5.4630013, 8.0912886, -4.4741712, 6.7811985, -12.2441969, 12.5654593
4: -3.5216219, 8.0479536, -2.8958118, 6.7110949, -10.2327166, 10.9437647

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -1.6265631, 4.4408922, -6.2223225, 6.4219961
1: -4.8894196, 7.3645873, -4.4537382, 6.8536711, -11.7430906, 11.8183250
2: -3.0549450, 6.6544957, -2.7905805, 6.1733203, -9.2282648, 9.4450750
3: -5.4630013, 8.0912886, -4.9789329, 7.5049791, -12.9679794, 13.0702209
4: -3.5216219, 8.0479536, -3.2266197, 7.4491405, -10.9707623, 11.2745714

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9617012
time: 0.90 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9617012
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.5829084, 4.2865834, -6.1767387, 6.6833591
1: -5.1917362, 7.8175325, -4.3325753, 6.6108027, -11.8025389, 12.1501064
2: -3.2490828, 7.0843124, -2.7101820, 5.9499145, -9.1989965, 9.7944946
3: -5.8050518, 8.6047878, -4.8427486, 7.2483263, -13.0533781, 13.4475365
4: -3.7534659, 8.5734529, -3.1318765, 7.1877618, -10.9412279, 11.7053289

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582073, upper bound: 17.9582073
time: 0.55 seconds

## Relational analysis of NS_A1_B1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9582073, upper bound: 17.9582073
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -1.7394853, 4.7084341, -6.5985889, 6.8399358
1: -5.1917362, 7.8175325, -4.7698765, 7.2454658, -12.4372005, 12.5874090
2: -3.2490828, 7.0843124, -2.9917207, 6.5442080, -9.7932911, 10.0760326
3: -5.8050518, 8.6047878, -5.3356533, 7.9606481, -13.7656994, 13.9404402
4: -3.7534659, 8.5734529, -3.4611101, 7.9156199, -11.6690855, 12.0345631

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_B2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582737, upper bound: 17.9615477
time: 0.74 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582737, upper bound: 17.9616141
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6415975, 4.4590487, -2.0258384, 5.4558244, -7.0974216, 6.4848871
1: -4.4953952, 6.8639817, -5.5796051, 8.3446941, -12.8400888, 12.4435863
2: -2.8065214, 6.1872668, -3.5107706, 7.5853653, -10.3918858, 9.6980362
3: -5.0186200, 7.5167313, -6.2471113, 9.2124949, -14.2311153, 13.7638407
4: -3.2377799, 7.4721994, -4.0569630, 9.1839800, -12.4217577, 11.5291624

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9514811
time: 0.61 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9480159
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.7814304, 4.7954335, -2.0258384, 5.4558244, -7.2372541, 6.8212719
1: -4.8894196, 7.3645873, -5.5796051, 8.3446941, -13.2341137, 12.9441919
2: -3.0549450, 6.6544957, -3.5107706, 7.5853653, -10.6403084, 10.1652660
3: -5.4630013, 8.0912886, -6.2471113, 9.2124949, -14.6754942, 14.3383989
4: -3.5216219, 8.0479536, -4.0569630, 9.1839800, -12.7055998, 12.1049166

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9515160
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9515824
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.0836692, 5.5584378, -7.4485931, 7.1841202
1: -5.1917362, 7.8175325, -5.7272377, 8.4556704, -13.6474066, 13.5447702
2: -3.2490828, 7.0843124, -3.6248012, 7.7101402, -10.9592228, 10.7091141
3: -5.8050518, 8.6047878, -6.4037743, 9.3579464, -15.1629982, 15.0085621
4: -3.7534659, 8.5734529, -4.1912947, 9.3349752, -13.0884409, 12.7647476

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9579621, upper bound: 17.9544923
time: 0.47 seconds

## Relational analysis of NS_A1_B2_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9579621, upper bound: 17.9544923
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1.8901551, 5.1004510, -2.1504512, 5.7671428, -7.6572981, 7.2509022
1: -5.1917362, 7.8175325, -5.9255209, 8.7834377, -13.9751720, 13.7430525
2: -3.2490828, 7.0843124, -3.7280650, 8.0094328, -11.2585154, 10.8123760
3: -5.8050518, 8.6047878, -6.6320848, 9.7263088, -15.5313606, 15.2368717
4: -3.7534659, 8.5734529, -4.3051324, 9.7046118, -13.4580774, 12.8785849

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9584548, upper bound: 17.9576106
time: 0.84 seconds

## Relational analysis of NS_A1_B2_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9584548, upper bound: 17.9576771
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.7085392, 4.6322861, -2.3526335, 6.3142529, -8.0227919, 6.9849195
1: -4.6837349, 7.1263371, -6.4942160, 9.6076832, -14.2914181, 13.6205530
2: -2.9348574, 6.4350877, -4.0940924, 8.7726021, -11.7074585, 10.5291805
3: -5.2395458, 7.8343787, -7.2752800, 10.6551113, -15.8946571, 15.1096563
4: -3.3983858, 7.7850680, -4.7291746, 10.6296968, -14.0280828, 12.5142422

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9369521, upper bound: 17.9595944
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9585253, upper bound: 17.9576106
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.8698349, 5.0807242, -2.3526335, 6.3142529, -8.1840878, 7.4333572
1: -5.1344204, 7.7851038, -6.4942160, 9.6076832, -14.7421017, 14.2793198
2: -3.2255132, 7.0570145, -4.0940924, 8.7726021, -11.9981155, 11.1511068
3: -5.7478380, 8.5735712, -7.2752800, 10.6551113, -16.4029484, 15.8488503
4: -3.7377400, 8.5425339, -4.7291746, 10.6296968, -14.3674364, 13.2717066

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9575758
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9605462
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.8698349, 5.0807242, -6.9345741, 6.8537354
1: -5.0963554, 7.6366177, -5.1344204, 7.7851038, -12.8814583, 12.7710361
2: -3.2001376, 6.9270759, -3.2255132, 7.0570145, -10.2571526, 10.1525888
3: -5.7014980, 8.4203701, -5.7478380, 8.5735712, -14.2750692, 14.1682081
4: -3.6966069, 8.3848782, -3.7377400, 8.5425339, -12.2391396, 12.1226168

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9613381
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9613381
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.6415975, 4.4590487, -6.4848871, 7.0974216
1: -5.5796051, 8.3446941, -4.4953952, 6.8639817, -12.4435863, 12.8400888
2: -3.5107706, 7.5853653, -2.8065214, 6.1872668, -9.6980362, 10.3918858
3: -6.2471113, 9.2124949, -5.0186200, 7.5167313, -13.7638407, 14.2311153
4: -4.0569630, 9.1839800, -3.2377799, 7.4721994, -11.5291624, 12.4217577

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9581251
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9610202
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.7814304, 4.7954335, -6.8212719, 7.2372541
1: -5.5796051, 8.3446941, -4.8894196, 7.3645873, -12.9441910, 13.2341137
2: -3.5107706, 7.5853653, -3.0549450, 6.6544957, -10.1652660, 10.6403093
3: -6.2471113, 9.2124949, -5.4630013, 8.0912886, -14.3383989, 14.6754932
4: -4.0569630, 9.1839800, -3.5216219, 8.0479536, -12.1049166, 12.7055998

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9581251
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9610202
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -2.0836692, 5.5584378, -1.8901551, 5.1004510, -7.1841202, 7.4485931
1: -5.7272377, 8.4556704, -5.1917362, 7.8175325, -13.5447702, 13.6474066
2: -3.6248012, 7.7101402, -3.2490828, 7.0843124, -10.7091141, 10.9592228
3: -6.4037743, 9.3579464, -5.8050518, 8.6047878, -15.0085611, 15.1629982
4: -4.1912947, 9.3349752, -3.7534659, 8.5734529, -12.7647476, 13.0884409

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_A1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9544923, upper bound: 17.9579621
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_A1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9544923, upper bound: 17.9579621
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -2.1504512, 5.7671428, -1.8901551, 5.1004510, -7.2509022, 7.6572981
1: -5.9255209, 8.7834377, -5.1917362, 7.8175325, -13.7430534, 13.9751720
2: -3.7280650, 8.0094328, -3.2490828, 7.0843124, -10.8123760, 11.2585154
3: -6.6320848, 9.7263088, -5.8050518, 8.6047878, -15.2368708, 15.5313606
4: -4.3051324, 9.7046118, -3.7534659, 8.5734529, -12.8785849, 13.4580774

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_A1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9584548
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9617952
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7085392, 4.6322861, -6.9849195, 8.0227919
1: -6.4942160, 9.6076832, -4.6837349, 7.1263371, -13.6205530, 14.2914181
2: -4.0940924, 8.7726021, -2.9348574, 6.4350877, -10.5291796, 11.7074585
3: -7.2752800, 10.6551113, -5.2395458, 7.8343787, -15.1096563, 15.8946571
4: -4.7291746, 10.6296968, -3.3983858, 7.7850680, -12.5142422, 14.0280828

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9595944, upper bound: 17.9371208
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609751, upper bound: 17.9586940
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.8698349, 5.0807242, -7.4333577, 8.1840878
1: -6.4942160, 9.6076832, -5.1344204, 7.7851038, -14.2793188, 14.7421017
2: -4.0940924, 8.7726021, -3.2255132, 7.0570145, -11.1511068, 11.9981155
3: -7.2752800, 10.6551113, -5.7478380, 8.5735712, -15.8488512, 16.4029503
4: -4.7291746, 10.6296968, -3.7377400, 8.5425339, -13.2717075, 14.3674364

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9586940
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9586940
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -1.8530533, 4.9817724, -2.4610372, 6.5926552, -8.4457083, 7.4428096
1: -5.0941758, 7.6334920, -6.7975974, 10.0043488, -15.0985231, 14.4310894
2: -3.1987631, 6.9241257, -4.2729445, 9.1532860, -12.3520470, 11.1970701
3: -5.6990623, 8.4168243, -7.6107635, 11.1077843, -16.8068447, 16.0275860
4: -3.6949906, 8.3813257, -4.9338398, 11.0912857, -14.7862759, 13.3151655

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -2.0249863, 5.4535413, -2.4610372, 6.5926552, -8.6176405, 7.9145784
1: -5.5772629, 8.3412933, -6.7975974, 10.0043488, -15.5816097, 15.1388912
2: -3.5092976, 7.5822034, -4.2729445, 9.1532860, -12.6625824, 11.8551464
3: -6.2444925, 9.2086458, -7.6107635, 11.1077843, -17.3522739, 16.8194065
4: -4.0552330, 9.1801662, -4.9338398, 11.0912857, -15.1465187, 14.1140060

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9574580
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9608225
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -2.1809726, 5.8487449, -2.4610372, 6.5926552, -8.7736282, 8.3097801
1: -6.0111499, 8.9041996, -6.7975974, 10.0043488, -16.0154991, 15.7017975
2: -3.7821393, 8.1218777, -4.2729445, 9.1532860, -12.9354248, 12.3948221
3: -6.7286444, 9.8643560, -7.6107635, 11.1077843, -17.8364277, 17.4751148
4: -4.3679342, 9.8418579, -4.9338398, 11.0912857, -15.4592190, 14.7756977

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_A2_A1_A1

### Relational analysis result of NS_A2_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9547077, upper bound: 17.9607292
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_A2_A1_A2

### Relational analysis result of NS_A2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9612227
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -2.3517973, 6.3120275, -2.4610372, 6.5926552, -8.9444523, 8.7730637
1: -6.4919486, 9.6043711, -6.7975974, 10.0043488, -16.4962978, 16.4019680
2: -4.0926495, 8.7695465, -4.2729445, 9.1532860, -13.2459354, 13.0424910
3: -7.2727432, 10.6513538, -7.6107635, 11.1077843, -18.3805256, 18.2621174
4: -4.7274809, 10.6260061, -4.9338398, 11.0912857, -15.8187647, 15.5598450

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_A2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9588315, upper bound: 17.9609912
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9613915
time: 0.98 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.40 seconds
NS_A1_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9585836, upper bound: 17.9585836
NS_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9585836, upper bound: 17.9616664
NS_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9616664, upper bound: 17.9586503
NS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9616664, upper bound: 17.9617330
NS_A1_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
NS_A1_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9586185
NS_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9617012
NS_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9583095, upper bound: 17.9617012
NS_A1_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9582073, upper bound: 17.9582073
NS_A1_B1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9582073, upper bound: 17.9582073
NS_A1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9582737, upper bound: 17.9615477
NS_A1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9582737, upper bound: 17.9616141
NS_A1_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9514811
NS_A1_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9480159
NS_A1_B2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9515160
NS_A1_B2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9515824
NS_A1_B2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9579621, upper bound: 17.9544923
NS_A1_B2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9579621, upper bound: 17.9544923
NS_A1_B2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9584548, upper bound: 17.9576106
NS_A1_B2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9584548, upper bound: 17.9576771
NS_A1_B2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9369521, upper bound: 17.9595944
NS_A1_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9585253, upper bound: 17.9576106
NS_A1_B2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9575758
NS_A1_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9580337, upper bound: 17.9605462
NS_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9613381
NS_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9613381
NS_A2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9581251
NS_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9610202
NS_A2_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9480159, upper bound: 17.9581251
NS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9610202
NS_A2_B1_A2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9544923, upper bound: 17.9579621
NS_A2_B1_A2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9544923, upper bound: 17.9579621
NS_A2_B1_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9584548
NS_A2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9617952
NS_A2_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9595944, upper bound: 17.9371208
NS_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9609751, upper bound: 17.9586940
NS_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9586940
NS_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9586940
NS_A2_B2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
NS_A2_B2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9480508, upper bound: 17.9573666
NS_A2_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9574580
NS_A2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9608225
NS_A2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9547077, upper bound: 17.9607292
NS_A2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9612227
NS_A2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9588315, upper bound: 17.9609912
NS_A2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9613915

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1.4669125, 4.0071440, -1.6265631, 4.4408922, -5.9078045, 5.6337061
1: -4.0064602, 6.2006745, -4.4537382, 6.8536711, -10.8601313, 10.6544132
2: -2.5055699, 5.5651913, -2.7905805, 6.1733203, -8.6788902, 8.3557720
3: -4.4741712, 6.7811985, -4.9789329, 7.5049791, -11.9791508, 11.7601318
4: -2.8958118, 6.7110949, -3.2266197, 7.4491405, -10.3449516, 9.9377146

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9462781, upper bound: 17.9606577
time: 0.93 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9585836, upper bound: 17.9616664
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -1.6265631, 4.4408922, -1.4669125, 4.0071440, -5.6337061, 5.9078045
1: -4.4537382, 6.8536711, -4.0064602, 6.2006745, -10.6544132, 10.8601313
2: -2.7905805, 6.1733203, -2.5055699, 5.5651913, -8.3557711, 8.6788902
3: -4.9789329, 7.5049791, -4.4741712, 6.7811985, -11.7601309, 11.9791508
4: -3.2266197, 7.4491405, -2.8958118, 6.7110949, -9.9377146, 10.3449526

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606577, upper bound: 17.9463448
time: 0.71 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616663, upper bound: 17.9586503
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1.6265631, 4.4408922, -1.6265631, 4.4408922, -6.0674553, 6.0674548
1: -4.4537382, 6.8536711, -4.4537382, 6.8536711, -11.3074093, 11.3074093
2: -2.7905805, 6.1733203, -2.7905805, 6.1733203, -8.9639006, 8.9639006
3: -4.9789329, 7.5049791, -4.9789329, 7.5049791, -12.4839115, 12.4839115
4: -3.2266197, 7.4491405, -3.2266197, 7.4491405, -10.6757603, 10.6757603

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606577, upper bound: 17.9615645
time: 0.58 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9616664, upper bound: 17.9586503
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6045514, 4.3435383, -1.6265631, 4.4408922, -6.0454435, 5.9701009
1: -4.3935027, 6.6973906, -4.4537382, 6.8536711, -11.2471733, 11.1511288
2: -2.7491412, 6.0295582, -2.7905805, 6.1733203, -8.9224615, 8.8201370
3: -4.9114966, 7.3468657, -4.9789329, 7.5049791, -12.4164734, 12.3257980
4: -3.1771119, 7.2852683, -3.2266197, 7.4491405, -10.6262512, 10.5118885

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9366697, upper bound: 17.9572378
time: 0.50 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9617012
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.6265631, 4.4408922, -6.2050276, 6.4008837
1: -4.8395271, 7.3438787, -4.4537382, 6.8536711, -11.6931982, 11.7976170
2: -3.0361805, 6.6364141, -2.7905805, 6.1733203, -9.2095003, 9.4269943
3: -5.4143553, 8.0728092, -4.9789329, 7.5049791, -12.9193335, 13.0517426
4: -3.5127008, 8.0279589, -3.2266197, 7.4491405, -10.9618397, 11.2545776

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9572342, upper bound: 17.9463130
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9617545
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.7085392, 4.6322861, -1.7394853, 4.7084341, -6.4169731, 6.3717709
1: -4.6837349, 7.1263371, -4.7698765, 7.2454658, -11.9292011, 11.8962116
2: -2.9348574, 6.4350877, -2.9917207, 6.5442080, -9.4790649, 9.4268084
3: -5.2395458, 7.8343787, -5.3356533, 7.9606481, -13.2001934, 13.1700287
4: -3.3983858, 7.7850680, -3.4611101, 7.9156199, -11.3140059, 11.2461777

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9615128
time: 0.57 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9615477
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.8698349, 5.0807242, -1.7394853, 4.7084341, -6.5782690, 6.8202095
1: -5.1344204, 7.7851038, -4.7698765, 7.2454658, -12.3798828, 12.5549784
2: -3.2255132, 7.0570145, -2.9917207, 6.5442080, -9.7697210, 10.0487347
3: -5.7478380, 8.5735712, -5.3356533, 7.9606481, -13.7084866, 13.9092236
4: -3.7377400, 8.5425339, -3.4611101, 7.9156199, -11.6533585, 12.0036430

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9581724
time: 1.03 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9582073
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.6265631, 4.4408922, -6.2947421, 6.6104631
1: -5.0963554, 7.6366177, -4.4537382, 6.8536711, -11.9500265, 12.0903559
2: -3.2001376, 6.9270759, -2.7905805, 6.1733203, -9.3734579, 9.7176561
3: -5.7014980, 8.4203701, -4.9789329, 7.5049791, -13.2064772, 13.3993034
4: -3.6966069, 8.3848782, -3.2266197, 7.4491405, -11.1457472, 11.6114969

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9479608, upper bound: 17.9609431
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.7641354, 4.7743206, -6.6281705, 6.7480354
1: -5.0963554, 7.6366177, -4.8395271, 7.3438787, -12.4402342, 12.4761448
2: -3.2001376, 6.9270759, -3.0361805, 6.6364141, -9.8365517, 9.9632540
3: -5.7014980, 8.4203701, -5.4143553, 8.0728092, -13.7743063, 13.8347254
4: -3.6966069, 8.3848782, -3.5127008, 8.0279589, -11.7245655, 11.8975773

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9479608, upper bound: 17.9609431
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.6265631, 4.4408922, -6.4667306, 7.0823870
1: -5.5796051, 8.3446941, -4.4537382, 6.8536711, -12.4332762, 12.7984324
2: -3.5107706, 7.5853653, -2.7905805, 6.1733203, -9.6840906, 10.3759451
3: -6.2471113, 9.2124949, -4.9789329, 7.5049791, -13.7520905, 14.1914272
4: -4.0569630, 9.1839800, -3.2266197, 7.4491405, -11.5061035, 12.4105988

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9504726, upper bound: 17.9612184
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9584659
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.7641354, 4.7743206, -6.8001590, 7.2199593
1: -5.5796051, 8.3446941, -4.8395271, 7.3438787, -12.9234838, 13.1842213
2: -3.5107706, 7.5853653, -3.0361805, 6.6364141, -10.1471844, 10.6215439
3: -6.2471113, 9.2124949, -5.4143553, 8.0728092, -14.3199205, 14.6268492
4: -4.0569630, 9.1839800, -3.5127008, 8.0279589, -12.0849218, 12.6966782

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9501353, upper bound: 17.9593483
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9610202
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2.1504512, 5.7671428, -1.8698349, 5.0807242, -7.2311749, 7.6369767
1: -5.9255209, 8.7834377, -5.1344204, 7.7851038, -13.7106218, 13.9178572
2: -3.7280650, 8.0094328, -3.2255132, 7.0570145, -10.7850780, 11.2349463
3: -6.6320848, 9.7263088, -5.7478380, 8.5735712, -15.2056561, 15.4741468
4: -4.3051324, 9.7046118, -3.7377400, 8.5425339, -12.8476639, 13.4423513

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9584548
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9617952
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7063978, 4.6262894, -6.9789224, 8.0206499
1: -6.4942160, 9.6076832, -4.6777611, 7.1175847, -13.6118011, 14.2854443
2: -4.0940924, 8.7726021, -2.9309602, 6.4269266, -10.5210190, 11.7035608
3: -7.2752800, 10.6551113, -5.2327356, 7.8243399, -15.0996189, 15.8878469
4: -4.7291746, 10.6296968, -3.3937197, 7.7752075, -12.5043821, 14.0234165

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609402, upper bound: 17.9586940
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9586940
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.6265631, 4.4408922, -6.7935257, 7.9408154
1: -6.4942160, 9.6076832, -4.4537382, 6.8536711, -13.3478870, 14.0614214
2: -4.0940924, 8.7726021, -2.7905805, 6.1733203, -10.2674122, 11.5631828
3: -7.2752800, 10.6551113, -4.9789329, 7.5049791, -14.7802591, 15.6340446
4: -4.7291746, 10.6296968, -3.2266197, 7.4491405, -12.1783142, 13.8563166

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608851, upper bound: 17.9615953
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619845
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7641354, 4.7743206, -7.1269541, 8.0783882
1: -6.4942160, 9.6076832, -4.8395271, 7.3438787, -13.8380947, 14.4472103
2: -4.0940924, 8.7726021, -3.0361805, 6.6364141, -10.7305050, 11.8087797
3: -7.2752800, 10.6551113, -5.4143553, 8.0728092, -15.3480873, 16.0694656
4: -4.7291746, 10.6296968, -3.5127008, 8.0279589, -12.7571325, 14.1423969

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608851, upper bound: 17.9615953
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619845
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2.0249863, 5.4535413, -2.4365036, 6.5491772, -8.5741615, 7.8900452
1: -5.5772629, 8.3412933, -6.7268949, 9.9520082, -15.5292702, 15.0681877
2: -3.5092976, 7.5822034, -4.2440166, 9.0934563, -12.6027508, 11.8262196
3: -6.2444925, 9.2086458, -7.5375009, 11.0490761, -17.2935658, 16.7461472
4: -4.0552330, 9.1801662, -4.9053960, 11.0185871, -15.0738201, 14.0855618

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9512721, upper bound: 17.9513526
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9512721, upper bound: 17.9574580
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -2.0830095, 5.5566773, -2.4610372, 6.5926552, -8.6756649, 8.0177145
1: -5.7254672, 8.4531155, -6.7975974, 10.0043488, -15.7298155, 15.2507133
2: -3.6236660, 7.7077422, -4.2729445, 9.1532860, -12.7769518, 11.9806862
3: -6.4017940, 9.3550158, -7.6107635, 11.1077843, -17.5095768, 16.9657764
4: -4.1899552, 9.3320694, -4.9338398, 11.0912857, -15.2812405, 14.2659092

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9546186, upper bound: 17.9573647
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_A2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9546186, upper bound: 17.9573647
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -2.1496613, 5.7650371, -2.4610372, 6.5926552, -8.7423162, 8.2260733
1: -5.9233732, 8.7803421, -6.7975974, 10.0043488, -15.9277220, 15.5779400
2: -3.7266986, 8.0065413, -4.2729445, 9.1532860, -12.8799849, 12.2794857
3: -6.6296806, 9.7227898, -7.6107635, 11.1077843, -17.7374649, 17.3335514
4: -4.3035269, 9.7011204, -4.9338398, 11.0912857, -15.3948126, 14.6349592

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -2.2416615, 5.9900465, -2.4610372, 6.5926552, -8.8343143, 8.4510832
1: -6.1721716, 9.0997105, -6.7975974, 10.0043488, -16.1765213, 15.8973074
2: -3.9128830, 8.3089504, -4.2729445, 9.1532860, -13.0661688, 12.5818949
3: -6.9070063, 10.0880318, -7.6107635, 11.1077843, -18.0147896, 17.6987877
4: -4.5250936, 10.0657692, -4.9338398, 11.0912857, -15.6163788, 14.9996090

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9587430, upper bound: 17.9576266
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9587431, upper bound: 17.9609912
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -2.3154230, 6.2140522, -2.4610372, 6.5926552, -8.9080772, 8.6750889
1: -6.3896255, 9.4576197, -6.7975974, 10.0043488, -16.3939743, 16.2552166
2: -4.0280309, 8.6336317, -4.2729445, 9.1532860, -13.1813164, 12.9065762
3: -7.1569982, 10.4839859, -7.6107635, 11.1077843, -18.2647820, 18.0947456
4: -4.6523867, 10.4605103, -4.9338398, 11.0912857, -15.7436714, 15.3943501

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609751, upper bound: 17.9580269
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9580269
time: 0.75 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.15 seconds
NS_A1_B1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9462781, upper bound: 17.9606577
NS_A1_B1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9585836, upper bound: 17.9616664
NS_A1_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9606577, upper bound: 17.9463448
NS_A1_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9616663, upper bound: 17.9586503
NS_A1_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9606577, upper bound: 17.9615645
NS_A1_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9616664, upper bound: 17.9586503
NS_A1_B1_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9366697, upper bound: 17.9572378
NS_A1_B1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9617012
NS_A1_B1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9572342, upper bound: 17.9463130
NS_A1_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9582428, upper bound: 17.9617545
NS_A1_B1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9615128
NS_A1_B1_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9615477
NS_A1_B1_B2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9581724
NS_A1_B1_B2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9581724, upper bound: 17.9582073
NS_A2_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9479608, upper bound: 17.9609431
NS_A2_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
NS_A2_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9479608, upper bound: 17.9609431
NS_A2_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9480825, upper bound: 17.9613381
NS_A2_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9504726, upper bound: 17.9612184
NS_A2_B1_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9514812, upper bound: 17.9584659
NS_A2_B1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9501353, upper bound: 17.9593483
NS_A2_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9515161, upper bound: 17.9610202
NS_A2_B1_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9584548
NS_A2_B1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9575758, upper bound: 17.9617952
NS_A2_B1_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9609402, upper bound: 17.9586940
NS_A2_B1_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9609403, upper bound: 17.9586940
NS_A2_B1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9608851, upper bound: 17.9615953
NS_A2_B1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619845
NS_A2_B1_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9608851, upper bound: 17.9615953
NS_A2_B1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9610069, upper bound: 17.9619845
NS_A2_B2_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9512721, upper bound: 17.9513526
NS_A2_B2_A2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9512721, upper bound: 17.9574580
NS_A2_B2_A2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9546186, upper bound: 17.9573647
NS_A2_B2_A2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9546186, upper bound: 17.9573647
NS_A2_B2_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
NS_A2_B2_A2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9576106, upper bound: 17.9578582
NS_A2_B2_A2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9587430, upper bound: 17.9576266
NS_A2_B2_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9587431, upper bound: 17.9609912
NS_A2_B2_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9609751, upper bound: 17.9580269
NS_A2_B2_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.15
Output dim: 3, lower bound: -17.9609752, upper bound: 17.9580269

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.4144485, 3.8669677, -1.6265631, 4.4408922, -5.8553410, 5.4935293
1: -3.8646359, 5.9884300, -4.4537382, 6.8536711, -10.7183075, 10.4421682
2: -2.4118309, 5.3668852, -2.7905805, 6.1733203, -8.5851517, 8.1574650
3: -4.3125520, 6.5341787, -4.9789329, 7.5049791, -11.8175306, 11.5131111
4: -2.7835054, 6.4808979, -3.2266197, 7.4491405, -10.2326450, 9.7075176

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9462328, upper bound: 17.9605134
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9462328, upper bound: 17.9606577
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.4648697, 4.0017033, -1.6265631, 4.4408922, -5.9057617, 5.6282663
1: -4.0007639, 6.1925073, -4.4537382, 6.8536711, -10.8544350, 10.6462460
2: -2.5018861, 5.5575690, -2.7905805, 6.1733203, -8.6752062, 8.3481483
3: -4.4676657, 6.7716904, -4.9789329, 7.5049791, -11.9726448, 11.7506237
4: -2.8914733, 6.7020569, -3.2266197, 7.4491405, -10.3406143, 9.9286737

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9580782, upper bound: 17.9608510
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9585601, upper bound: 17.9614618
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.6265631, 4.4408922, -1.4144485, 3.8669677, -5.4935288, 5.8553410
1: -4.4537382, 6.8536711, -3.8646359, 5.9884300, -10.4421682, 10.7183075
2: -2.7905805, 6.1733203, -2.4118309, 5.3668852, -8.1574640, 8.5851517
3: -4.9789329, 7.5049791, -4.3125520, 6.5341787, -11.5131111, 11.8175306
4: -3.2266197, 7.4491405, -2.7835054, 6.4808979, -9.7075176, 10.2326441

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605134, upper bound: 17.9462328
time: 0.88 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605134, upper bound: 17.9463448
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.6265631, 4.4408922, -1.4648697, 4.0017033, -5.6282663, 5.9057617
1: -4.4537382, 6.8536711, -4.0007639, 6.1925073, -10.6462460, 10.8544350
2: -2.7905805, 6.1733203, -2.5018861, 5.5575690, -8.3481493, 8.6752062
3: -4.9789329, 7.5049791, -4.4676657, 6.7716904, -11.7506237, 11.9726448
4: -3.2266197, 7.4491405, -2.8914733, 6.7020569, -9.9286737, 10.3406143

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608510, upper bound: 17.9580782
time: 0.54 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614618, upper bound: 17.9585601
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1.6265631, 4.4408922, -1.5698674, 4.2881880, -5.9147511, 6.0107594
1: -4.4537382, 6.8536711, -4.3001013, 6.6175771, -11.0713158, 11.1537724
2: -2.7905805, 6.1733203, -2.6881747, 5.9574351, -8.7480125, 8.8614950
3: -4.9789329, 7.5049791, -4.8039408, 7.2321057, -12.2110386, 12.3089190
4: -3.2266197, 7.4491405, -3.1042047, 7.1976261, -10.4242439, 10.5533447

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614767, upper bound: 17.9614163
time: 0.59 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614767, upper bound: 17.9615645
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1.6265631, 4.4408922, -1.6249232, 4.4365354, -6.0630984, 6.0658154
1: -4.4537382, 6.8536711, -4.4491611, 6.8472085, -11.3009472, 11.3028316
2: -2.7905805, 6.1733203, -2.7875941, 6.1672549, -8.9578342, 8.9609146
3: -4.9789329, 7.5049791, -4.9737148, 7.4975076, -12.4764404, 12.4786930
4: -3.2266197, 7.4491405, -3.2231069, 7.4419231, -10.6685429, 10.6722469

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609176, upper bound: 17.9610465
time: 0.86 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615284, upper bound: 17.9615047
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1.6024703, 4.3379574, -1.6265631, 4.4408922, -6.0433626, 5.9645200
1: -4.3877144, 6.6889501, -4.4537382, 6.8536711, -11.2413855, 11.1426888
2: -2.7453523, 6.0217762, -2.7905805, 6.1733203, -8.9186726, 8.8123550
3: -4.9048948, 7.3371639, -4.9789329, 7.5049791, -12.4098740, 12.3160973
4: -3.1725528, 7.2758012, -3.2266197, 7.4491405, -10.6216927, 10.5024204

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9539762, upper bound: 17.9610784
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582481, upper bound: 17.9616845
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.7641354, 4.7743206, -1.6249232, 4.4365354, -6.2006707, 6.3992438
1: -4.8395271, 7.3438787, -4.4491611, 6.8472085, -11.6867352, 11.7930393
2: -3.0361805, 6.6364141, -2.7875941, 6.1672549, -9.2034340, 9.4240084
3: -5.4143553, 8.0728092, -4.9737148, 7.4975076, -12.9118633, 13.0465231
4: -3.5127008, 8.0279589, -3.2231069, 7.4419231, -10.9546232, 11.2510662

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9609092, upper bound: 17.9611734
time: 0.55 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613814, upper bound: 17.9615555
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -1.4669125, 4.0071440, -1.7394853, 4.7084341, -6.1753464, 5.7466283
1: -4.0064602, 6.2006745, -4.7698765, 7.2454658, -11.2519245, 10.9705496
2: -2.5055699, 5.5651913, -2.9917207, 6.5442080, -9.0497780, 8.5569115
3: -4.4741712, 6.7811985, -5.3356533, 7.9606481, -12.4348192, 12.1168499
4: -2.8958118, 6.7110949, -3.4611101, 7.9156199, -10.8114319, 10.1722050

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9367363, upper bound: 17.9601669
time: 0.84 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582390, upper bound: 17.9615128
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1.6045514, 4.3435383, -1.7394853, 4.7084341, -6.3129854, 6.0830231
1: -4.3935027, 6.6973906, -4.7698765, 7.2454658, -11.6389675, 11.4672670
2: -2.7491412, 6.0295582, -2.9917207, 6.5442080, -9.2933493, 9.0212784
3: -4.9114966, 7.3468657, -5.3356533, 7.9606481, -12.8721418, 12.6825180
4: -3.1771119, 7.2852683, -3.4611101, 7.9156199, -11.0927305, 10.7463779

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9367363, upper bound: 17.9601669
time: 0.68 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9582390, upper bound: 17.9615477
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.5698674, 4.2881880, -6.1420379, 6.5537682
1: -5.0963554, 7.6366177, -4.3001013, 6.6175771, -11.7139311, 11.9367189
2: -3.2001376, 6.9270759, -2.6881747, 5.9574351, -9.1575727, 9.6152496
3: -5.7014980, 8.4203701, -4.8039408, 7.2321057, -12.9336033, 13.2243109
4: -3.6966069, 8.3848782, -3.1042047, 7.1976261, -10.8942318, 11.4890823

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9434484, upper bound: 17.9606814
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9478408, upper bound: 17.9612900
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.6249232, 4.4365354, -6.2903852, 6.6088238
1: -5.0963554, 7.6366177, -4.4491611, 6.8472085, -11.9435635, 12.0857782
2: -3.2001376, 6.9270759, -2.7875941, 6.1672549, -9.3673925, 9.7146692
3: -5.7014980, 8.4203701, -4.9737148, 7.4975076, -13.1990051, 13.3940840
4: -3.6966069, 8.3848782, -3.2231069, 7.4419231, -11.1385298, 11.6079845

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9447674, upper bound: 17.9609737
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9471664, upper bound: 17.9613250
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.7046560, 4.6108918, -6.4647417, 6.6885562
1: -5.0963554, 7.6366177, -4.6781445, 7.0940909, -12.1904469, 12.3147621
2: -3.2001376, 6.9270759, -2.9285951, 6.4071131, -9.6072502, 9.8556709
3: -5.7014980, 8.4203701, -5.2297997, 7.7859783, -13.4874763, 13.6501694
4: -3.6966069, 8.3848782, -3.3814113, 7.7537327, -11.4503384, 11.7662888

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9446461, upper bound: 17.9604530
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9470451, upper bound: 17.9608043
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -1.8538499, 4.9839005, -1.7624259, 4.7696519, -6.6235018, 6.7463264
1: -5.0963554, 7.6366177, -4.8347473, 7.3370724, -12.4334278, 12.4713650
2: -3.2001376, 6.9270759, -3.0330470, 6.6300502, -9.8301878, 9.9601231
3: -5.7014980, 8.4203701, -5.4089136, 8.0649805, -13.7664776, 13.8292828
4: -3.6966069, 8.3848782, -3.5089464, 8.0202703, -11.7168770, 11.8938236

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9448018, upper bound: 17.9608544
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9472008, upper bound: 17.9612056
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.5698674, 4.2881880, -6.3140264, 7.0256920
1: -5.5796051, 8.3446941, -4.3001013, 6.6175771, -12.1971817, 12.6447954
2: -3.5107706, 7.5853653, -2.6881747, 5.9574351, -9.4682055, 10.2735386
3: -6.2471113, 9.2124949, -4.8039408, 7.2321057, -13.4792175, 14.0164356
4: -4.0569630, 9.1839800, -3.1042047, 7.1976261, -11.2545891, 12.2881842

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9509602, upper bound: 17.9611631
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9509602, upper bound: 17.9612184
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.7624259, 4.7696519, -6.7954903, 7.2182503
1: -5.5796051, 8.3446941, -4.8347473, 7.3370724, -12.9166775, 13.1794415
2: -3.5107706, 7.5853653, -3.0330470, 6.6300502, -10.1408205, 10.6184120
3: -6.2471113, 9.2124949, -5.4089136, 8.0649805, -14.3120918, 14.6214075
4: -4.0569630, 9.1839800, -3.5089464, 8.0202703, -12.0772333, 12.6929255

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9511096, upper bound: 17.9596975
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9515571, upper bound: 17.9608299
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.1504512, 5.7671428, -1.7641354, 4.7743206, -6.9247718, 7.5312781
1: -5.9255209, 8.7834377, -4.8395271, 7.3438787, -13.2693977, 13.6229649
2: -3.7280650, 8.0094328, -3.0361805, 6.6364141, -10.3644781, 11.0456104
3: -6.6320848, 9.7263088, -5.4143553, 8.0728092, -14.7048931, 15.1406622
4: -4.3051324, 9.7046118, -3.5127008, 8.0279589, -12.3330898, 13.2173119

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9559744, upper bound: 17.9617911
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576424, upper bound: 17.9617952
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.4648697, 4.0017033, -6.3543367, 7.7791224
1: -6.4942160, 9.6076832, -4.0007639, 6.1925073, -12.6867237, 13.6084471
2: -4.0940924, 8.7726021, -2.5018861, 5.5575690, -9.6516590, 11.2744884
3: -7.2752800, 10.6551113, -4.4676657, 6.7716904, -14.0469675, 15.1227770
4: -4.7291746, 10.6296968, -2.8914733, 6.7020569, -11.4312296, 13.5211697

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9556970, upper bound: 17.9573333
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608099, upper bound: 17.9581323
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.6024703, 4.3379574, -6.6905899, 7.9167228
1: -6.4942160, 9.6076832, -4.3877144, 6.6889501, -13.1831665, 13.9953976
2: -4.0940924, 8.7726021, -2.7453523, 6.0217762, -10.1158667, 11.5179539
3: -7.2752800, 10.6551113, -4.9048948, 7.3371639, -14.6124430, 15.5600061
4: -4.7291746, 10.6296968, -3.1725528, 7.2758012, -12.0049753, 13.8022499

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9556970, upper bound: 17.9573333
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608100, upper bound: 17.9581323
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.5698674, 4.2881880, -6.6408215, 7.8841200
1: -6.4942160, 9.6076832, -4.3001013, 6.6175771, -13.1117926, 13.9077845
2: -4.0940924, 8.7726021, -2.6881747, 5.9574351, -10.0515261, 11.4607754
3: -7.2752800, 10.6551113, -4.8039408, 7.2321057, -14.5073833, 15.4590521
4: -4.7291746, 10.6296968, -3.1042047, 7.1976261, -11.9267998, 13.7339020

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601846, upper bound: 17.9611701
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606405, upper bound: 17.9618331
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.6249232, 4.4365354, -6.7891684, 7.9391751
1: -6.4942160, 9.6076832, -4.4491611, 6.8472085, -13.3414249, 14.0568438
2: -4.0940924, 8.7726021, -2.7875941, 6.1672549, -10.2613468, 11.5601959
3: -7.2752800, 10.6551113, -4.9737148, 7.4975076, -14.7727871, 15.6288252
4: -4.7291746, 10.6296968, -3.2231069, 7.4419231, -12.1710978, 13.8528042

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9602965, upper bound: 17.9612492
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607524, upper bound: 17.9619123
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7046560, 4.6108918, -6.9635248, 8.0189075
1: -6.4942160, 9.6076832, -4.6781445, 7.0940909, -13.5883064, 14.2858276
2: -4.0940924, 8.7726021, -2.9285951, 6.4071131, -10.5012054, 11.7011957
3: -7.2752800, 10.6551113, -5.2297997, 7.7859783, -15.0612583, 15.8849096
4: -4.7291746, 10.6296968, -3.3814113, 7.7537327, -12.4829054, 14.0111084

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601747, upper bound: 17.9607761
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606306, upper bound: 17.9614324
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7624259, 4.7696519, -7.1222854, 8.0766773
1: -6.4942160, 9.6076832, -4.8347473, 7.3370724, -13.8312883, 14.4424305
2: -4.0940924, 8.7726021, -3.0330470, 6.6300502, -10.7241421, 11.8056488
3: -7.2752800, 10.6551113, -5.4089136, 8.0649805, -15.3402576, 16.0640221
4: -4.7291746, 10.6296968, -3.5089464, 8.0202703, -12.7494450, 14.1386433

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604233, upper bound: 17.9598643
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9619060
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.2416615, 5.9900465, -2.4365036, 6.5491772, -8.7908363, 8.4265499
1: -6.1721716, 9.0997105, -6.7268949, 9.9520082, -16.1241798, 15.8266029
2: -3.9128830, 8.3089504, -4.2440166, 9.0934563, -13.0063381, 12.5529671
3: -6.9070063, 10.0880318, -7.5375009, 11.0490761, -17.9560814, 17.6255302
4: -4.5250936, 10.0657692, -4.9053960, 11.0185871, -15.5436802, 14.9711647

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_A2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9583743, upper bound: 17.9514432
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9583743, upper bound: 17.9576266
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2.3154230, 6.2140522, -2.2677357, 6.0882111, -8.4036341, 8.4817867
1: -6.3896255, 9.4576197, -6.2520523, 9.2545786, -15.6442013, 15.7096701
2: -4.0280309, 8.6336317, -3.9379964, 8.4527826, -12.4808130, 12.5716286
3: -7.1569982, 10.4839859, -7.0011187, 10.2674742, -17.4244728, 17.4851036
4: -4.6523867, 10.4605103, -4.5507307, 10.2395697, -14.8919563, 15.0112410

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607311, upper bound: 17.9482647
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607311, upper bound: 17.9580269
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2.3154230, 6.2140522, -2.4365036, 6.5491772, -8.8645992, 8.6505556
1: -6.3896255, 9.4576197, -6.7268949, 9.9520082, -16.3416328, 16.1845112
2: -4.0280309, 8.6336317, -4.2440166, 9.0934563, -13.1214867, 12.8776484
3: -7.1569982, 10.4839859, -7.5375009, 11.0490761, -18.2060738, 18.0214863
4: -4.6523867, 10.4605103, -4.9053960, 11.0185871, -15.6709738, 15.3659058

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9518435
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9580269
time: 0.71 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.43 seconds
NS_A1_B1_B1_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9462328, upper bound: 17.9605134
NS_A1_B1_B1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9462328, upper bound: 17.9606577
NS_A1_B1_B1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9580782, upper bound: 17.9608510
NS_A1_B1_B1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9585601, upper bound: 17.9614618
NS_A1_B1_B1_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9605134, upper bound: 17.9462328
NS_A1_B1_B1_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9605134, upper bound: 17.9463448
NS_A1_B1_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9608510, upper bound: 17.9580782
NS_A1_B1_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9614618, upper bound: 17.9585601
NS_A1_B1_B1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9614767, upper bound: 17.9614163
NS_A1_B1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9614767, upper bound: 17.9615645
NS_A1_B1_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9609176, upper bound: 17.9610465
NS_A1_B1_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9615284, upper bound: 17.9615047
NS_A1_B1_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9539762, upper bound: 17.9610784
NS_A1_B1_B1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9582481, upper bound: 17.9616845
NS_A1_B1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9609092, upper bound: 17.9611734
NS_A1_B1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9613814, upper bound: 17.9615555
NS_A1_B1_B2_B2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9367363, upper bound: 17.9601669
NS_A1_B1_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9582390, upper bound: 17.9615128
NS_A1_B1_B2_B2_B2_A1_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9367363, upper bound: 17.9601669
NS_A1_B1_B2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9582390, upper bound: 17.9615477
NS_A2_B1_A1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9434484, upper bound: 17.9606814
NS_A2_B1_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9478408, upper bound: 17.9612900
NS_A2_B1_A1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9447674, upper bound: 17.9609737
NS_A2_B1_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9471664, upper bound: 17.9613250
NS_A2_B1_A1_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9446461, upper bound: 17.9604530
NS_A2_B1_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9470451, upper bound: 17.9608043
NS_A2_B1_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9448018, upper bound: 17.9608544
NS_A2_B1_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9472008, upper bound: 17.9612056
NS_A2_B1_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9509602, upper bound: 17.9611631
NS_A2_B1_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9509602, upper bound: 17.9612184
NS_A2_B1_A1_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9511096, upper bound: 17.9596975
NS_A2_B1_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9515571, upper bound: 17.9608299
NS_A2_B1_A2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9559744, upper bound: 17.9617911
NS_A2_B1_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9576424, upper bound: 17.9617952
NS_A2_B1_A2_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9556970, upper bound: 17.9573333
NS_A2_B1_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9608099, upper bound: 17.9581323
NS_A2_B1_A2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9556970, upper bound: 17.9573333
NS_A2_B1_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9608100, upper bound: 17.9581323
NS_A2_B1_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9601846, upper bound: 17.9611701
NS_A2_B1_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9606405, upper bound: 17.9618331
NS_A2_B1_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9602965, upper bound: 17.9612492
NS_A2_B1_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9607524, upper bound: 17.9619123
NS_A2_B1_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9601747, upper bound: 17.9607761
NS_A2_B1_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9606306, upper bound: 17.9614324
NS_A2_B1_A2_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9604233, upper bound: 17.9598643
NS_A2_B1_A2_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9610416, upper bound: 17.9619060
NS_A2_B2_A2_A2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9583743, upper bound: 17.9514432
NS_A2_B2_A2_A2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9583743, upper bound: 17.9576266
NS_A2_B2_A2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9607311, upper bound: 17.9482647
NS_A2_B2_A2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9607311, upper bound: 17.9580269
NS_A2_B2_A2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9518435
NS_A2_B2_A2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.43
Output dim: 3, lower bound: -17.9607312, upper bound: 17.9580269

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.4144485, 3.8669677, -1.6249232, 4.4365354, -5.8509836, 5.4918904
1: -3.8646359, 5.9884300, -4.4491611, 6.8472085, -10.7118444, 10.4375906
2: -2.4118309, 5.3668852, -2.7875941, 6.1672549, -8.5790863, 8.1544781
3: -4.3125520, 6.5341787, -4.9737148, 7.4975076, -11.8100595, 11.5078917
4: -2.7835054, 6.4808979, -3.2231069, 7.4419231, -10.2254276, 9.7040043

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9458761, upper bound: 17.9516246
time: 0.53 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9459518, upper bound: 17.9606371
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.4648697, 4.0017033, -1.5060112, 4.1655335, -5.6304035, 5.5077143
1: -4.0007639, 6.1925073, -4.1330576, 6.4516354, -10.4523993, 10.3255653
2: -2.5018861, 5.5575690, -2.5870545, 5.7973061, -8.2991915, 8.1446218
3: -4.4676657, 6.7716904, -4.6237392, 7.0496340, -11.5172987, 11.3954296
4: -2.8914733, 6.7020569, -3.0038633, 6.9974103, -9.8888836, 9.7059174

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9575615, upper bound: 17.9607073
time: 0.84 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576260, upper bound: 17.9606998
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.4648697, 4.0017033, -1.6048422, 4.3873596, -5.8522291, 5.6065454
1: -4.0007639, 6.1925073, -4.3947482, 6.7741451, -10.7749090, 10.5872555
2: -2.5018861, 5.5575690, -2.7527990, 6.0993452, -8.6012316, 8.3103676
3: -4.4676657, 6.7716904, -4.9127541, 7.4169321, -11.8845978, 11.6844416
4: -2.8914733, 6.7020569, -3.1832728, 7.3600473, -10.2515202, 9.8853264

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9490191, upper bound: 17.9613160
time: 0.54 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9585297, upper bound: 17.9614505
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1.5060112, 4.1655335, -1.4648697, 4.0017033, -5.5077143, 5.6304035
1: -4.1330576, 6.4516354, -4.0007639, 6.1925073, -10.3255653, 10.4523993
2: -2.5870545, 5.7973061, -2.5018861, 5.5575690, -8.1446218, 8.2991905
3: -4.6237392, 7.0496340, -4.4676657, 6.7716904, -11.3954296, 11.5172997
4: -3.0038633, 6.9974103, -2.8914733, 6.7020569, -9.7059183, 9.8888836

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607073, upper bound: 17.9575615
time: 0.56 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606998, upper bound: 17.9576260
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.6048422, 4.3873596, -1.4648697, 4.0017033, -5.6065454, 5.8522291
1: -4.3947482, 6.7741451, -4.0007639, 6.1925073, -10.5872555, 10.7749090
2: -2.7527990, 6.0993452, -2.5018861, 5.5575690, -8.3103676, 8.6012316
3: -4.9127541, 7.4169321, -4.4676657, 6.7716904, -11.6844416, 11.8845978
4: -3.1832728, 7.3600473, -2.8914733, 6.7020569, -9.8853273, 10.2515202

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613160, upper bound: 17.9490191
time: 0.52 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614505, upper bound: 17.9584640
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1.5698674, 4.2881880, -1.5698674, 4.2881880, -5.8580551, 5.8580551
1: -4.3001013, 6.6175771, -4.3001013, 6.6175771, -10.9176788, 10.9176788
2: -2.6881747, 5.9574351, -2.6881747, 5.9574351, -8.6456079, 8.6456079
3: -4.8039408, 7.2321057, -4.8039408, 7.2321057, -12.0360460, 12.0360451
4: -3.1042047, 7.1976261, -3.1042047, 7.1976261, -10.3018284, 10.3018284

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9520815, upper bound: 17.9612198
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614582, upper bound: 17.9613999
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1.6249232, 4.4365354, -1.5698674, 4.2881880, -5.9131112, 6.0064025
1: -4.4491611, 6.8472085, -4.3001013, 6.6175771, -11.0667372, 11.1473103
2: -2.7875941, 6.1672549, -2.6881747, 5.9574351, -8.7450266, 8.8554287
3: -4.9737148, 7.4975076, -4.8039408, 7.2321057, -12.2058191, 12.3014488
4: -3.2231069, 7.4419231, -3.1042047, 7.1976261, -10.4207325, 10.5461273

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9520815, upper bound: 17.9613183
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9614582, upper bound: 17.9615512
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.5060112, 4.1655335, -1.6249232, 4.4365354, -5.9425468, 5.7904568
1: -4.1330576, 6.4516354, -4.4491611, 6.8472085, -10.9802666, 10.9007969
2: -2.5870545, 5.7973061, -2.7875941, 6.1672549, -8.7543097, 8.5848989
3: -4.6237392, 7.0496340, -4.9737148, 7.4975076, -12.1212463, 12.0233479
4: -3.0038633, 6.9974103, -3.2231069, 7.4419231, -10.4457865, 10.2205172

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607280, upper bound: 17.9602305
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607205, upper bound: 17.9603126
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.6048422, 4.3873596, -1.6249232, 4.4365354, -6.0413775, 6.0122828
1: -4.3947482, 6.7741451, -4.4491611, 6.8472085, -11.2419567, 11.2233067
2: -2.7527990, 6.0993452, -2.7875941, 6.1672549, -8.9200535, 8.8869390
3: -4.9127541, 7.4169321, -4.9737148, 7.4975076, -12.4102612, 12.3906460
4: -3.1832728, 7.3600473, -3.2231069, 7.4419231, -10.6251965, 10.5831547

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613374, upper bound: 17.9613121
time: 0.57 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613374, upper bound: 17.9615047
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -1.5395411, 4.1634321, -1.6265631, 4.4408922, -5.9804335, 5.7899952
1: -4.1997929, 6.4172425, -4.4537382, 6.8536711, -11.0534639, 10.8709812
2: -2.6338506, 5.7670598, -2.7905805, 6.1733203, -8.8071709, 8.5576391
3: -4.6916542, 7.0376282, -4.9789329, 7.5049791, -12.1966333, 12.0165606
4: -3.0435202, 6.9502764, -3.2266197, 7.4491405, -10.4926605, 10.1768942

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9533320, upper bound: 17.9602650
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9538128, upper bound: 17.9608736
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -1.5935837, 4.3134270, -1.6265631, 4.4408922, -6.0344753, 5.9399896
1: -4.3630157, 6.6516109, -4.4537382, 6.8536711, -11.2166862, 11.1053486
2: -2.7292690, 5.9871588, -2.7905805, 6.1733203, -8.9025898, 8.7777376
3: -4.8770127, 7.2944579, -4.9789329, 7.5049791, -12.3819904, 12.2733908
4: -3.1531167, 7.2330418, -3.2266197, 7.4491405, -10.6022568, 10.4596605

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9564673, upper bound: 17.9612839
time: 0.57 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576866, upper bound: 17.9615539
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6566859, 4.5420566, -1.6249232, 4.4365354, -6.0932212, 6.1669798
1: -4.5533586, 6.9835386, -4.4491611, 6.8472085, -11.4005671, 11.4326992
2: -2.8570082, 6.3125820, -2.7875941, 6.1672549, -9.0242624, 9.1001749
3: -5.0970364, 7.6685061, -4.9737148, 7.4975076, -12.5945435, 12.6422195
4: -3.3121867, 7.6437159, -3.2231069, 7.4419231, -10.7541103, 10.8668232

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9551902, upper bound: 17.9570877
time: 0.57 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608959, upper bound: 17.9611627
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7380224, 4.7118812, -1.6249232, 4.4365354, -6.1745567, 6.3368044
1: -4.7691450, 7.2523766, -4.4491611, 6.8472085, -11.6163540, 11.7015352
2: -2.9913461, 6.5501766, -2.7875941, 6.1672549, -9.1586008, 9.3377705
3: -5.3357091, 7.9691801, -4.9737148, 7.4975076, -12.8332167, 12.9428949
4: -3.4616387, 7.9244447, -3.2231069, 7.4419231, -10.9035616, 11.1475515

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9552868, upper bound: 17.9568612
time: 0.87 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613685, upper bound: 17.9615432
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -1.4648697, 4.0017033, -1.7394853, 4.7084341, -6.1733036, 5.7411885
1: -4.0007639, 6.1925073, -4.7698765, 7.2454658, -11.2462292, 10.9623833
2: -2.5018861, 5.5575690, -2.9917207, 6.5442080, -9.0460939, 8.5492897
3: -4.4676657, 6.7716904, -5.3356533, 7.9606481, -12.4283133, 12.1073427
4: -2.8914733, 6.7020569, -3.4611101, 7.9156199, -10.8070927, 10.1631660

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9569550, upper bound: 17.9555988
time: 0.53 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9581025, upper bound: 17.9613874
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -1.6024703, 4.3379574, -1.7394853, 4.7084341, -6.3109045, 6.0774417
1: -4.3877144, 6.6889501, -4.7698765, 7.2454658, -11.6331806, 11.4588261
2: -2.7453523, 6.0217762, -2.9917207, 6.5442080, -9.2895603, 9.0134964
3: -4.9048948, 7.3371639, -5.3356533, 7.9606481, -12.8655434, 12.6728153
4: -3.1725528, 7.2758012, -3.4611101, 7.9156199, -11.0881729, 10.7369118

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9558913, upper bound: 17.9551596
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576824, upper bound: 17.9614194
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1.7851632, 4.7765241, -1.5698674, 4.2881880, -6.0733509, 6.3463917
1: -4.8910799, 7.3296819, -4.3001013, 6.6175771, -11.5086555, 11.6297836
2: -3.0777755, 6.6302996, -2.6881747, 5.9574351, -9.0352087, 9.3184738
3: -5.4690371, 8.0826683, -4.8039408, 7.2321057, -12.7011414, 12.8866091
4: -3.5572319, 8.0239801, -3.1042047, 7.1976261, -10.7548580, 11.1281853

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9431217, upper bound: 17.9606509
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9430163, upper bound: 17.9605209
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1.8455555, 4.9597502, -1.5698674, 4.2881880, -6.1337433, 6.5296173
1: -5.0733457, 7.6012111, -4.3001013, 6.6175771, -11.6909199, 11.9013119
2: -3.1850111, 6.8936429, -2.6881747, 5.9574351, -9.1424437, 9.5818167
3: -5.6755304, 8.3803148, -4.8039408, 7.2321057, -12.9076357, 13.1842556
4: -3.6786118, 8.3449745, -3.1042047, 7.1976261, -10.8762379, 11.4491787

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9444394, upper bound: 17.9607809
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9469244, upper bound: 17.9611528
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6705869, 4.4756937, -1.6249232, 4.4365354, -6.1071224, 6.1006169
1: -4.5681796, 6.8725495, -4.4491611, 6.8472085, -11.4153881, 11.3217106
2: -2.8821464, 6.2137852, -2.7875941, 6.1672549, -9.0494013, 9.0013781
3: -5.1131873, 7.5820637, -4.9737148, 7.4975076, -12.6106949, 12.5557775
4: -3.3373265, 7.5138359, -3.2231069, 7.4419231, -10.7792492, 10.7369423

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9442597, upper bound: 17.9601587
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9447396, upper bound: 17.9607695
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.8139337, 4.8666945, -1.6249232, 4.4365354, -6.2504687, 6.4916177
1: -4.9869347, 7.4631629, -4.4491611, 6.8472085, -11.8341427, 11.9123240
2: -3.1260049, 6.7634478, -2.7875941, 6.1672549, -9.2932596, 9.5510416
3: -5.5779891, 8.2200403, -4.9737148, 7.4975076, -13.0754957, 13.1937551
4: -3.6069016, 8.1930723, -3.2231069, 7.4419231, -11.0488243, 11.4161797

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9466598, upper bound: 17.9605100
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9471396, upper bound: 17.9611208
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1.8139337, 4.8666945, -1.7046560, 4.6108918, -6.4248257, 6.5713501
1: -4.9869347, 7.4631629, -4.6781445, 7.0940909, -12.0810261, 12.1413078
2: -3.1260049, 6.7634478, -2.9285951, 6.4071131, -9.5331173, 9.6920433
3: -5.5779891, 8.2200403, -5.2297997, 7.7859783, -13.3639660, 13.4498405
4: -3.6069016, 8.1930723, -3.3814113, 7.7537327, -11.3606319, 11.5744829

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9452972, upper bound: 17.9603770
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 30

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.6705869, 4.4756937, -1.7624259, 4.7696519, -6.4402390, 6.2381196
1: -4.5681796, 6.8725495, -4.8347473, 7.3370724, -11.9052525, 11.7072964
2: -2.8821464, 6.2137852, -3.0330470, 6.6300502, -9.5121965, 9.2468319
3: -5.1131873, 7.5820637, -5.4089136, 8.0649805, -13.1781654, 12.9909754
4: -3.3373265, 7.5138359, -3.5089464, 8.0202703, -11.3575964, 11.0227814

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9443859, upper bound: 17.9601490
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9447740, upper bound: 17.9606217
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.8139337, 4.8666945, -1.7624259, 4.7696519, -6.5835857, 6.6291199
1: -4.9869347, 7.4631629, -4.8347473, 7.3370724, -12.3240070, 12.2979107
2: -3.1260049, 6.7634478, -3.0330470, 6.6300502, -9.7560549, 9.7964945
3: -5.5779891, 8.2200403, -5.4089136, 8.0649805, -13.6429653, 13.6289539
4: -3.6069016, 8.1930723, -3.5089464, 8.0202703, -11.6271725, 11.7020178

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9467859, upper bound: 17.9605003
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9471741, upper bound: 17.9609729
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1.9617697, 5.2819810, -1.5698674, 4.2881880, -6.2499576, 6.8518486
1: -5.4064646, 8.0788879, -4.3001013, 6.6175771, -12.0240412, 12.3789892
2: -3.3930345, 7.3429985, -2.6881747, 5.9574351, -9.3504686, 10.0311737
3: -6.0497794, 8.9081545, -4.8039408, 7.2321057, -13.2818842, 13.7120953
4: -3.9171157, 8.8984909, -3.1042047, 7.1976261, -11.1147385, 12.0026951

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9505892, upper bound: 17.9611357
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

## BFS NS instance: NS_A2_B1_A1_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2.0242922, 5.4517026, -1.5698674, 4.2881880, -6.3124800, 7.0215702
1: -5.5753055, 8.3385353, -4.3001013, 6.6175771, -12.1928806, 12.6386366
2: -3.5078702, 7.5796757, -2.6881747, 5.9574351, -9.4653006, 10.2678480
3: -6.2422271, 9.2054749, -4.8039408, 7.2321057, -13.4743328, 14.0094147
4: -4.0535598, 9.1772156, -3.1042047, 7.1976261, -11.2511845, 12.2814188

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9505892, upper bound: 17.9611909
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_A2_B1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -2.0258384, 5.4558244, -1.7363042, 4.7071791, -6.7330174, 7.1921282
1: -5.5796051, 8.3446941, -4.7643337, 7.2455111, -12.8251143, 13.1090260
2: -3.5107706, 7.5853653, -2.9881930, 6.5437622, -10.0545321, 10.5735579
3: -6.2471113, 9.2124949, -5.3302326, 7.9612885, -14.2083998, 14.5427275
4: -4.0569630, 9.1839800, -3.4578595, 7.9166951, -11.9736576, 12.6418400

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9509039, upper bound: 17.9607805
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_A2_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9509039, upper bound: 17.9594656
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2.0099237, 5.3995810, -1.7641354, 4.7743206, -6.7842445, 7.1637163
1: -5.5343218, 8.2522211, -4.8395271, 7.3438787, -12.8782005, 13.0917482
2: -3.4881237, 7.4808788, -3.0361805, 6.6364141, -10.1245365, 10.5170574
3: -6.1995683, 9.0983219, -5.4143553, 8.0728092, -14.2723770, 14.5126772
4: -4.0267377, 9.0553761, -3.5127008, 8.0279589, -12.0546970, 12.5680742

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9558527, upper bound: 17.9614306
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9560091, upper bound: 17.9617911
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2.1024954, 5.6391344, -1.7641354, 4.7743206, -6.8768158, 7.4032698
1: -5.7942529, 8.5928297, -4.8395271, 7.3438787, -13.1381321, 13.4323568
2: -3.6448104, 7.8338251, -3.0361805, 6.6364141, -10.2812233, 10.8700047
3: -6.4846878, 9.5106916, -5.4143553, 8.0728092, -14.5574970, 14.9250460
4: -4.2082725, 9.4921455, -3.5127008, 8.0279589, -12.2362309, 13.0048466

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9575206, upper bound: 17.9614347
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9617952
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -1.4648697, 4.0017033, -6.2865124, 7.5792999
1: -6.3056726, 9.3081608, -4.0007639, 6.1925073, -12.4981794, 13.3089247
2: -3.9664235, 8.4968262, -2.5018861, 5.5575690, -9.5239906, 10.9987116
3: -7.0619555, 10.3120298, -4.4676657, 6.7716904, -13.8336449, 14.7796955
4: -4.5765290, 10.2993307, -2.8914733, 6.7020569, -11.2785835, 13.1908035

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9600990, upper bound: 17.9577914
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605553, upper bound: 17.9584498
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2848091, 6.1144300, -1.6024703, 4.3379574, -6.6227655, 7.7168994
1: -6.3056726, 9.3081608, -4.3877144, 6.6889501, -12.9946232, 13.6958752
2: -3.9664235, 8.4968262, -2.7453523, 6.0217762, -9.9881983, 11.2421780
3: -7.0619555, 10.3120298, -4.9048948, 7.3371639, -14.3991184, 15.2169247
4: -4.5765290, 10.2993307, -3.1725528, 7.2758012, -11.8523302, 13.4718828

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9597486, upper bound: 17.9565614
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_A2_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605873, upper bound: 17.9579570
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2.2680049, 6.1405296, -1.5698674, 4.2881880, -6.5561929, 7.7103968
1: -6.2710662, 9.3523111, -4.3001013, 6.6175771, -12.8886433, 13.6524124
2: -3.9533379, 8.5302296, -2.6881747, 5.9574351, -9.9107714, 11.2184019
3: -7.0305719, 10.3618450, -4.8039408, 7.2321057, -14.2626762, 15.1657858
4: -4.5682459, 10.3433199, -3.1042047, 7.1976261, -11.7658691, 13.4475250

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9541087, upper bound: 17.9461642
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9601672, upper bound: 17.9441489
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2.3220739, 6.2402458, -1.5698674, 4.2881880, -6.6102619, 7.8101130
1: -6.4108419, 9.4974470, -4.3001013, 6.6175771, -13.0284195, 13.7975483
2: -4.0414753, 8.6704102, -2.6881747, 5.9574351, -9.9989061, 11.3585835
3: -7.1822367, 10.5304642, -4.8039408, 7.2321057, -14.4143429, 15.3344049
4: -4.6688733, 10.5066061, -3.1042047, 7.1976261, -11.8664989, 13.6108093

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9542292, upper bound: 17.9486021
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9606235, upper bound: 17.9618145
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2.2680049, 6.1405296, -1.6249232, 4.4365354, -6.7045403, 7.7654529
1: -6.2710662, 9.3523111, -4.4491611, 6.8472085, -13.1182747, 13.8014717
2: -3.9533379, 8.5302296, -2.7875941, 6.1672549, -10.1205931, 11.3178225
3: -7.0305719, 10.3618450, -4.9737148, 7.4975076, -14.5280800, 15.3355589
4: -4.5682459, 10.3433199, -3.2231069, 7.4419231, -12.0101681, 13.5664272

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9550733, upper bound: 17.9575574
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9602783, upper bound: 17.9612384
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -2.3220739, 6.2402458, -1.6249232, 4.4365354, -6.7586088, 7.8651690
1: -6.4108419, 9.4974470, -4.4491611, 6.8472085, -13.2580509, 13.9466076
2: -4.0414753, 8.6704102, -2.7875941, 6.1672549, -10.2087288, 11.4580021
3: -7.1822367, 10.5304642, -4.9737148, 7.4975076, -14.6797447, 15.5041780
4: -4.6688733, 10.5066061, -3.2231069, 7.4419231, -12.1107960, 13.7297134

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9552633, upper bound: 17.9576021
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607346, upper bound: 17.9618969
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -2.2680049, 6.1405296, -1.7046560, 4.6108918, -6.8788967, 7.8451858
1: -6.2710662, 9.3523111, -4.6781445, 7.0940909, -13.3651571, 14.0304556
2: -3.9533379, 8.5302296, -2.9285951, 6.4071131, -10.3604498, 11.4588242
3: -7.0305719, 10.3618450, -5.2297997, 7.7859783, -14.8165503, 15.5916443
4: -4.5682459, 10.3433199, -3.3814113, 7.7537327, -12.3219757, 13.7247314

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9537590, upper bound: 17.9395693
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9601570, upper bound: 17.9607581
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2.3220739, 6.2402458, -1.7046560, 4.6108918, -6.9329653, 7.9449019
1: -6.4108419, 9.4974470, -4.6781445, 7.0940909, -13.5049324, 14.1755905
2: -4.0414753, 8.6704102, -2.9285951, 6.4071131, -10.4485855, 11.5990047
3: -7.1822367, 10.5304642, -5.2297997, 7.7859783, -14.9682150, 15.7602634
4: -4.6688733, 10.5066061, -3.3814113, 7.7537327, -12.4226055, 13.8880157

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9579843, upper bound: 17.9609160
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9579843, upper bound: 17.9614324
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_A2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -2.3526335, 6.3142529, -1.7173457, 4.6521316, -7.0047646, 8.0315981
1: -6.4942160, 9.6076832, -4.7113400, 7.1634054, -13.6576204, 14.3190231
2: -4.0940924, 8.7726021, -2.9544590, 6.4689240, -10.5630150, 11.7270613
3: -7.2752800, 10.6551113, -5.2706518, 7.8690810, -15.1443577, 15.9257631
4: -4.7291746, 10.6296968, -3.4182832, 7.8244667, -12.5536404, 14.0479803

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9603312, upper bound: 17.9610627
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_A2_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607870, upper bound: 17.9616834
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -2.3154230, 6.2140522, -1.8538499, 4.9839005, -7.2993236, 8.0679016
1: -6.3896255, 9.4576197, -5.0963554, 7.6366177, -14.0262423, 14.5539742
2: -4.0280309, 8.6336317, -3.2001376, 6.9270759, -10.9551067, 11.8337688
3: -7.1569982, 10.4839859, -5.7014980, 8.4203701, -15.5773678, 16.1854839
4: -4.6523867, 10.4605103, -3.6966069, 8.3848782, -13.0372639, 14.1571169

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9600029, upper bound: 17.9475709
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604766, upper bound: 17.9482346
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2.3154230, 6.2140522, -2.1817567, 5.8508301, -8.1662531, 8.3958073
1: -6.3896255, 9.4576197, -6.0132799, 8.9072704, -15.2968950, 15.4708996
2: -4.0280309, 8.6336317, -3.7834954, 8.1247444, -12.1527748, 12.4171276
3: -7.1569982, 10.4839859, -6.7310286, 9.8678417, -17.0248394, 17.2150135
4: -4.6523867, 10.4605103, -4.3695273, 9.8453188, -14.4977055, 14.8300381

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9553821, upper bound: 17.9564586
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9605936, upper bound: 17.9572576
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.3154230, 6.2140522, -2.0258384, 5.4558244, -7.7712474, 8.2398911
1: -6.3896255, 9.4576197, -5.5796051, 8.3446941, -14.7343178, 15.0372248
2: -4.0280309, 8.6336317, -3.5107706, 7.5853653, -11.6133957, 12.1444025
3: -7.1569982, 10.4839859, -6.2471113, 9.2124949, -16.3694935, 16.7310982
4: -4.6523867, 10.4605103, -4.0569630, 9.1839800, -13.8363657, 14.5174732

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607800, upper bound: 17.9514265
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9518434
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.3154230, 6.2140522, -2.3526335, 6.3142529, -8.6296759, 8.5666847
1: -6.3896255, 9.4576197, -6.4942160, 9.6076832, -15.9973078, 15.9518356
2: -4.0280309, 8.6336317, -4.0940924, 8.7726021, -12.8006325, 12.7277241
3: -7.1569982, 10.4839859, -7.2752800, 10.6551113, -17.8121090, 17.7592659
4: -4.6523867, 10.4605103, -4.7291746, 10.6296968, -15.2820835, 15.1896849

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9580024, upper bound: 17.9606915
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9613172
time: 0.82 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.95 seconds
NS_A1_B1_B1_A1_A1_B2_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9458761, upper bound: 17.9516246
NS_A1_B1_B1_A1_A1_B2_A1_B2_B2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9459518, upper bound: 17.9606371
NS_A1_B1_B1_A1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9575615, upper bound: 17.9607073
NS_A1_B1_B1_A1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9576260, upper bound: 17.9606998
NS_A1_B1_B1_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9490191, upper bound: 17.9613160
NS_A1_B1_B1_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9585297, upper bound: 17.9614505
NS_A1_B1_B1_A1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9607073, upper bound: 17.9575615
NS_A1_B1_B1_A1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9606998, upper bound: 17.9576260
NS_A1_B1_B1_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9613160, upper bound: 17.9490191
NS_A1_B1_B1_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9614505, upper bound: 17.9584640
NS_A1_B1_B1_A1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9520815, upper bound: 17.9612198
NS_A1_B1_B1_A1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9614582, upper bound: 17.9613999
NS_A1_B1_B1_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9520815, upper bound: 17.9613183
NS_A1_B1_B1_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9614582, upper bound: 17.9615512
NS_A1_B1_B1_A1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9607280, upper bound: 17.9602305
NS_A1_B1_B1_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9607205, upper bound: 17.9603126
NS_A1_B1_B1_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9613374, upper bound: 17.9613121
NS_A1_B1_B1_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9613374, upper bound: 17.9615047
NS_A1_B1_B1_A2_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9533320, upper bound: 17.9602650
NS_A1_B1_B1_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9538128, upper bound: 17.9608736
NS_A1_B1_B1_A2_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9564673, upper bound: 17.9612839
NS_A1_B1_B1_A2_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9576866, upper bound: 17.9615539
NS_A1_B1_B1_A2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9551902, upper bound: 17.9570877
NS_A1_B1_B1_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9608959, upper bound: 17.9611627
NS_A1_B1_B1_A2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9552868, upper bound: 17.9568612
NS_A1_B1_B1_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9613685, upper bound: 17.9615432
NS_A1_B1_B2_B2_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9569550, upper bound: 17.9555988
NS_A1_B1_B2_B2_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9581025, upper bound: 17.9613874
NS_A1_B1_B2_B2_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9558913, upper bound: 17.9551596
NS_A1_B1_B2_B2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9576824, upper bound: 17.9614194
NS_A2_B1_A1_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9444394, upper bound: 17.9607809
NS_A2_B1_A1_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9469244, upper bound: 17.9611528
NS_A2_B1_A1_A1_B2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9442597, upper bound: 17.9601587
NS_A2_B1_A1_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9447396, upper bound: 17.9607695
NS_A2_B1_A1_A1_B2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9466598, upper bound: 17.9605100
NS_A2_B1_A1_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9471396, upper bound: 17.9611208
NS_A2_B1_A1_A1_B2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9443859, upper bound: 17.9601490
NS_A2_B1_A1_A1_B2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9447740, upper bound: 17.9606217
NS_A2_B1_A1_A1_B2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9467859, upper bound: 17.9605003
NS_A2_B1_A1_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9471741, upper bound: 17.9609729
NS_A2_B1_A1_A2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9509039, upper bound: 17.9607805
NS_A2_B1_A1_A2_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9509039, upper bound: 17.9594656
NS_A2_B1_A2_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9558527, upper bound: 17.9614306
NS_A2_B1_A2_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9560091, upper bound: 17.9617911
NS_A2_B1_A2_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9575206, upper bound: 17.9614347
NS_A2_B1_A2_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9576771, upper bound: 17.9617952
NS_A2_B1_A2_A2_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9600990, upper bound: 17.9577914
NS_A2_B1_A2_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9605553, upper bound: 17.9584498
NS_A2_B1_A2_A2_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9597486, upper bound: 17.9565614
NS_A2_B1_A2_A2_B1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9605873, upper bound: 17.9579570
NS_A2_B1_A2_A2_B2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9541087, upper bound: 17.9461642
NS_A2_B1_A2_A2_B2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9601672, upper bound: 17.9441489
NS_A2_B1_A2_A2_B2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9542292, upper bound: 17.9486021
NS_A2_B1_A2_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9606235, upper bound: 17.9618145
NS_A2_B1_A2_A2_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9550733, upper bound: 17.9575574
NS_A2_B1_A2_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9602783, upper bound: 17.9612384
NS_A2_B1_A2_A2_B2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9552633, upper bound: 17.9576021
NS_A2_B1_A2_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9607346, upper bound: 17.9618969
NS_A2_B1_A2_A2_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9537590, upper bound: 17.9395693
NS_A2_B1_A2_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9601570, upper bound: 17.9607581
NS_A2_B1_A2_A2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9579843, upper bound: 17.9609160
NS_A2_B1_A2_A2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9579843, upper bound: 17.9614324
NS_A2_B1_A2_A2_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9603312, upper bound: 17.9610627
NS_A2_B1_A2_A2_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9607870, upper bound: 17.9616834
NS_A2_B2_A2_A2_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9600029, upper bound: 17.9475709
NS_A2_B2_A2_A2_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9604766, upper bound: 17.9482346
NS_A2_B2_A2_A2_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9553821, upper bound: 17.9564586
NS_A2_B2_A2_A2_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9605936, upper bound: 17.9572576
NS_A2_B2_A2_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9607800, upper bound: 17.9514265
NS_A2_B2_A2_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9518434
NS_A2_B2_A2_A2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9580024, upper bound: 17.9606915
NS_A2_B2_A2_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 3, lower bound: -17.9608225, upper bound: 17.9613172

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.4648697, 4.0017033, -1.4140668, 3.9275508, -5.3924208, 5.4157701
1: -4.0007639, 6.1925073, -3.8799617, 6.1013632, -10.1021271, 10.0724678
2: -2.5018861, 5.5575690, -2.4250388, 5.4681420, -7.9700279, 7.9826078
3: -4.4676657, 6.7716904, -4.3398733, 6.6551151, -11.1227808, 11.1115637
4: -2.8914733, 6.7020569, -2.8155425, 6.5970173, -9.4884901, 9.5175991

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9561736, upper bound: 17.9598136
time: 0.53 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9561736, upper bound: 17.9607073
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.4648697, 4.0017033, -1.4821296, 4.1072822, -5.5721521, 5.4838328
1: -4.0007639, 6.1925073, -4.0661507, 6.3671107, -10.3678741, 10.2586575
2: -2.5018861, 5.5575690, -2.5462053, 5.7169380, -8.2188244, 8.1037731
3: -4.4676657, 6.7716904, -4.5496559, 6.9543586, -11.4220219, 11.3213453
4: -2.8914733, 6.7020569, -2.9573431, 6.9000597, -9.7915316, 9.6594000

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9562380, upper bound: 17.9598060
time: 0.56 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9562380, upper bound: 17.9606998
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.4149915, 3.8622062, -1.6048422, 4.3873596, -5.8023505, 5.4670486
1: -3.8623416, 5.9860153, -4.3947482, 6.7741451, -10.6364870, 10.3807631
2: -2.4125664, 5.3618217, -2.7527990, 6.0993452, -8.5119114, 8.1146202
3: -4.3112707, 6.5367799, -4.9127541, 7.4169321, -11.7282019, 11.4495335
4: -2.7860272, 6.4699354, -3.1832728, 7.3600473, -10.1460743, 9.6532078

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9488971, upper bound: 17.9524514
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9488971, upper bound: 17.9524514
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.4514389, 3.9689660, -1.6048422, 4.3873596, -5.8387985, 5.5738082
1: -3.9645674, 6.1450500, -4.3947482, 6.7741451, -10.7387123, 10.5397968
2: -2.4785023, 5.5131779, -2.7527990, 6.0993452, -8.5778465, 8.2659769
3: -4.4272966, 6.7182412, -4.9127541, 7.4169321, -11.8442268, 11.6309929
4: -2.8646266, 6.6479735, -3.1832728, 7.3600473, -10.2246742, 9.8312464

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9584077, upper bound: 17.9525859
time: 0.65 seconds

## Relational analysis of NS_A1_B1_B1_A1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9584077, upper bound: 17.9614505
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -1.4140668, 3.9275508, -1.4648697, 4.0017033, -5.4157701, 5.3924208
1: -3.8799617, 6.1013632, -4.0007639, 6.1925073, -10.0724669, 10.1021271
2: -2.4250388, 5.4681420, -2.5018861, 5.5575690, -7.9826078, 7.9700279
3: -4.3398733, 6.6551151, -4.4676657, 6.7716904, -11.1115637, 11.1227808
4: -2.8155425, 6.5970173, -2.8914733, 6.7020569, -9.5175982, 9.4884901

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598135, upper bound: 17.9561736
time: 0.87 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598135, upper bound: 17.9575615
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1.4821296, 4.1072822, -1.4648697, 4.0017033, -5.4838328, 5.5721521
1: -4.0661507, 6.3671107, -4.0007639, 6.1925073, -10.2586575, 10.3678741
2: -2.5462053, 5.7169380, -2.5018861, 5.5575690, -8.1037712, 8.2188244
3: -4.5496559, 6.9543586, -4.4676657, 6.7716904, -11.3213453, 11.4220219
4: -2.9573431, 6.9000597, -2.8914733, 6.7020569, -9.6593990, 9.7915325

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598061, upper bound: 17.9562380
time: 0.60 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9598060, upper bound: 17.9576260
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.6048422, 4.3873596, -1.4149915, 3.8622062, -5.4670486, 5.8023505
1: -4.3947482, 6.7741451, -3.8623416, 5.9860153, -10.3807631, 10.6364870
2: -2.7527990, 6.0993452, -2.4125664, 5.3618217, -8.1146202, 8.5119114
3: -4.9127541, 7.4169321, -4.3112707, 6.5367799, -11.4495335, 11.7282019
4: -3.1832728, 7.3600473, -2.7860272, 6.4699354, -9.6532078, 10.1460743

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9524514, upper bound: 17.9488971
time: 0.91 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9524514, upper bound: 17.9488971
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.6048422, 4.3873596, -1.4514389, 3.9689660, -5.5738082, 5.8387985
1: -4.3947482, 6.7741451, -3.9645674, 6.1450500, -10.5397949, 10.7387123
2: -2.7527990, 6.0993452, -2.4785023, 5.5131779, -8.2659769, 8.5778475
3: -4.9127541, 7.4169321, -4.4272966, 6.7182412, -11.6309929, 11.8442278
4: -3.1832728, 7.3600473, -2.8646266, 6.6479735, -9.8312464, 10.2246733

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9525859, upper bound: 17.9584077
time: 0.58 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9525859, upper bound: 17.9585297
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1.5204883, 4.1499004, -1.5698674, 4.2881880, -5.8086762, 5.7197676
1: -4.1627750, 6.4084959, -4.3001013, 6.6175771, -10.7803497, 10.7085972
2: -2.5996075, 5.7626190, -2.6881747, 5.9574351, -8.5570402, 8.4507914
3: -4.6483974, 6.9986467, -4.8039408, 7.2321057, -11.8805008, 11.8025856
4: -2.9995980, 6.9670229, -3.1042047, 7.1976261, -10.1972218, 10.0712261

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9520058, upper bound: 17.9520058
time: 0.61 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9520058, upper bound: 17.9612199
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1.5563982, 4.2552500, -1.5698674, 4.2881880, -5.8445864, 5.8251171
1: -4.2637115, 6.5695858, -4.3001013, 6.6175771, -10.8812866, 10.8696871
2: -2.6646888, 5.9126687, -2.6881747, 5.9574351, -8.6221218, 8.6008434
3: -4.7633390, 7.1781459, -4.8039408, 7.2321057, -11.9954453, 11.9820862
4: -3.0772569, 7.1431713, -3.1042047, 7.1976261, -10.2748833, 10.2473755

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613825, upper bound: 17.9520801
time: 0.63 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613825, upper bound: 17.9613999
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1.5756693, 4.2981930, -1.5698674, 4.2881880, -5.8638573, 5.8680601
1: -4.3120289, 6.6387205, -4.3001013, 6.6175771, -10.9296036, 10.9388218
2: -2.6990612, 5.9734564, -2.6881747, 5.9574351, -8.6564922, 8.6616306
3: -4.8183880, 7.2646513, -4.8039408, 7.2321057, -12.0504923, 12.0685921
4: -3.1187191, 7.2116652, -3.1042047, 7.1976261, -10.3163452, 10.3158693

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9525181, upper bound: 17.9520659
time: 0.57 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9525181, upper bound: 17.9613184
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.6113889, 4.4038148, -1.5698674, 4.2881880, -5.8995771, 5.9736824
1: -4.4126368, 6.7996907, -4.3001013, 6.6175771, -11.0302124, 11.0997925
2: -2.7640035, 6.1226826, -2.6881747, 5.9574351, -8.7214365, 8.8108549
3: -4.9329348, 7.4434462, -4.8039408, 7.2321057, -12.1650410, 12.2473869
4: -3.1967762, 7.3875341, -3.1042047, 7.1976261, -10.3944016, 10.4917383

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615308, upper bound: 17.9521962
time: 0.56 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9615308, upper bound: 17.9615512
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -1.4140668, 3.9275508, -1.6249232, 4.4365354, -5.8506021, 5.5524740
1: -3.8799617, 6.1013632, -4.4491611, 6.8472085, -10.7271690, 10.5505238
2: -2.4250388, 5.4681420, -2.7875941, 6.1672549, -8.5922928, 8.2557344
3: -4.3398733, 6.6551151, -4.9737148, 7.4975076, -11.8373814, 11.6288290
4: -2.8155425, 6.5970173, -3.2231069, 7.4419231, -10.2574654, 9.8201237

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603019, upper bound: 17.9601972
time: 0.80 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9603019, upper bound: 17.9602305
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1.4821296, 4.1072822, -1.6249232, 4.4365354, -5.9186649, 5.7322054
1: -4.0661507, 6.3671107, -4.4491611, 6.8472085, -10.9133587, 10.8162718
2: -2.5462053, 5.7169380, -2.7875941, 6.1672549, -8.7134590, 8.5045309
3: -4.5496559, 6.9543586, -4.9737148, 7.4975076, -12.0471630, 11.9280701
4: -2.9573431, 6.9000597, -3.2231069, 7.4419231, -10.3992662, 10.1231670

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607205, upper bound: 17.9603126
time: 0.64 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9607205, upper bound: 17.9603126
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1.5484185, 4.2349319, -1.6249232, 4.4365354, -5.9849539, 5.8598547
1: -4.2418933, 6.5383139, -4.4491611, 6.8472085, -11.0891018, 10.9874744
2: -2.6508532, 5.8840308, -2.7875941, 6.1672549, -8.8181076, 8.6716242
3: -4.7386723, 7.1449890, -4.9737148, 7.4975076, -12.2361794, 12.1187019
4: -3.0611992, 7.1093707, -3.2231069, 7.4419231, -10.5031223, 10.3324776

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9612429, upper bound: 17.9525321
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613186, upper bound: 17.9612973
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.6032224, 4.3830395, -1.6249232, 4.4365354, -6.0397577, 6.0079627
1: -4.3902259, 6.7677393, -4.4491611, 6.8472085, -11.2374344, 11.2168999
2: -2.7498484, 6.0933371, -2.7875941, 6.1672549, -8.9171028, 8.8809309
3: -4.9075952, 7.4095287, -4.9737148, 7.4975076, -12.4051027, 12.3832436
4: -3.1797235, 7.3528981, -3.2231069, 7.4419231, -10.6216469, 10.5760050

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9520815, upper bound: 17.9612225
time: 0.61 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613186, upper bound: 17.9614936
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1.5395411, 4.1634321, -1.6048422, 4.3873596, -5.9269004, 5.7682743
1: -4.1997929, 6.4172425, -4.3947482, 6.7741451, -10.9739380, 10.8119907
2: -2.6338506, 5.7670598, -2.7527990, 6.0993452, -8.7331953, 8.5198593
3: -4.6916542, 7.0376282, -4.9127541, 7.4169321, -12.1085863, 11.9503794
4: -3.0435202, 6.9502764, -3.1832728, 7.3600473, -10.4035673, 10.1335468

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9447790, upper bound: 17.9607124
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9537669, upper bound: 17.9608581
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -1.4250324, 3.8649578, -1.6265631, 4.4408922, -5.8659248, 5.4915199
1: -3.8771226, 5.9671264, -4.4537382, 6.8536711, -10.7307940, 10.4208641
2: -2.4352989, 5.3531961, -2.7905805, 6.1733203, -8.6086197, 8.1437750
3: -4.3343601, 6.5446258, -4.9789329, 7.5049791, -11.8393383, 11.5235586
4: -2.8219013, 6.4443226, -3.2266197, 7.4491405, -10.2710419, 9.6709414

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9558293, upper bound: 17.9604709
time: 0.58 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9563081, upper bound: 17.9610794
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -1.5554630, 4.2079387, -1.6265631, 4.4408922, -5.9963551, 5.8345017
1: -4.2588296, 6.4890919, -4.4537382, 6.8536711, -11.1125002, 10.9428301
2: -2.6591625, 5.8370543, -2.7905805, 6.1733203, -8.8324833, 8.6276350
3: -4.7593107, 7.1048093, -4.9789329, 7.5049791, -12.2642899, 12.0837421
4: -3.0663376, 7.0526652, -3.2266197, 7.4491405, -10.5154781, 10.2792854

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9570307, upper bound: 17.9607409
time: 0.62 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A1_A2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9575094, upper bound: 17.9613494
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1.5926161, 4.3499174, -1.6249232, 4.4365354, -6.0291510, 5.9748402
1: -4.3748636, 6.7082291, -4.4491611, 6.8472085, -11.2220726, 11.1573896
2: -2.7355225, 6.0480204, -2.7875941, 6.1672549, -8.9027777, 8.8356133
3: -4.8960218, 7.3511605, -4.9737148, 7.4975076, -12.3935299, 12.3248749
4: -3.1659622, 7.3247833, -3.2231069, 7.4419231, -10.6078854, 10.5478897

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9604160, upper bound: 17.9605519
time: 0.76 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9604160, upper bound: 17.9611627
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.6758846, 4.5282412, -1.6249232, 4.4365354, -6.1124201, 6.1531644
1: -4.5962725, 6.9845934, -4.4491611, 6.8472085, -11.4434814, 11.4337540
2: -2.8751144, 6.2948365, -2.7875941, 6.1672549, -9.0423679, 9.0824299
3: -5.1412001, 7.6606855, -4.9737148, 7.4975076, -12.6387081, 12.6343994
4: -3.3191807, 7.6150193, -3.2231069, 7.4419231, -10.7611027, 10.8381262

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9528454, upper bound: 17.9612719
time: 0.58 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_A2_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_B2_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9613563, upper bound: 17.9615325
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1.4648697, 4.0017033, -1.6773915, 4.5253205, -5.9901905, 5.6790948
1: -4.0007639, 6.1925073, -4.5972762, 6.9775386, -10.9783020, 10.7897835
2: -2.5018861, 5.5575690, -2.8755770, 6.2891951, -8.7910814, 8.4331455
3: -4.4676657, 6.7716904, -5.1413808, 7.6522713, -12.1199369, 11.9130697
4: -2.8914733, 6.7020569, -3.3186591, 7.6067209, -10.4981937, 10.0207157

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9576257, upper bound: 17.9606608
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9580138, upper bound: 17.9611600
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_B2_B2_B2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.6024703, 4.3379574, -1.6773915, 4.5253205, -6.1277909, 6.0153484
1: -4.3877144, 6.6889501, -4.5972762, 6.9775386, -11.3652534, 11.2862263
2: -2.7453523, 6.0217762, -2.8755770, 6.2891951, -9.0345469, 8.8973532
3: -4.9048948, 7.3371639, -5.1413808, 7.6522713, -12.5571661, 12.4785433
4: -3.1725528, 7.2758012, -3.3186591, 7.6067209, -10.7792730, 10.5944605

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9570271, upper bound: 17.9606928
time: 0.69 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9575070, upper bound: 17.9611920
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1.6609077, 4.4472446, -1.5698674, 4.2881880, -5.9490957, 6.0171118
1: -4.5409904, 6.8310790, -4.3001013, 6.6175771, -11.1585665, 11.1311798
2: -2.8644402, 6.1739812, -2.6881747, 5.9574351, -8.8218737, 8.8621559
3: -5.0823493, 7.5348854, -4.8039408, 7.2321057, -12.3144550, 12.3388262
4: -3.3159158, 7.4657698, -3.1042047, 7.1976261, -10.5135403, 10.5699749

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9440937, upper bound: 17.9607486
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9438923, upper bound: 17.9606009
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.8068879, 4.8463931, -1.5698674, 4.2881880, -6.0950756, 6.4162607
1: -4.9672456, 7.4326029, -4.3001013, 6.6175771, -11.5848207, 11.7327042
2: -3.1132579, 6.7350345, -2.6881747, 5.9574351, -9.0706911, 9.4232063
3: -5.5557961, 8.1855698, -4.8039408, 7.2321057, -12.7879019, 12.9895105
4: -3.5918558, 8.1588516, -3.1042047, 7.1976261, -10.7894821, 11.2630558

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9465747, upper bound: 17.9611228
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9451304, upper bound: 17.9608122
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6705869, 4.4756937, -1.6032224, 4.3830395, -6.0536265, 6.0789161
1: -4.5681796, 6.8725495, -4.3902259, 6.7677393, -11.3359184, 11.2627754
2: -2.8821464, 6.2137852, -2.7498484, 6.0933371, -8.9754829, 8.9636326
3: -5.1131873, 7.5820637, -4.9075952, 7.4095287, -12.5227156, 12.4896574
4: -3.3373265, 7.5138359, -3.1797235, 7.3528981, -10.6902237, 10.6935596

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9443977, upper bound: 17.9607406
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9441892, upper bound: 17.9605989
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9440495, upper bound: 17.9599808
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.8139337, 4.8666945, -1.6032224, 4.3830395, -6.1969733, 6.4699168
1: -4.9869347, 7.4631629, -4.3902259, 6.7677393, -11.7546740, 11.8533888
2: -3.1260049, 6.7634478, -2.7498484, 6.0933371, -9.2193422, 9.5132961
3: -5.5779891, 8.2200403, -4.9075952, 7.4095287, -12.9875164, 13.1276350
4: -3.6069016, 8.1930723, -3.1797235, 7.3528981, -10.9597998, 11.3727961

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9467931, upper bound: 17.9610943
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9453903, upper bound: 17.9608027
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9452506, upper bound: 17.9601846
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.8139337, 4.8666945, -1.7363042, 4.7071791, -6.5211129, 6.6029983
1: -4.9869347, 7.4631629, -4.7643337, 7.2455111, -12.2324429, 12.2274952
2: -3.1260049, 6.7634478, -2.9881930, 6.5437622, -9.6697655, 9.7516403
3: -5.5779891, 8.2200403, -5.3302326, 7.9612885, -13.5392771, 13.5502729
4: -3.6069016, 8.1930723, -3.4578595, 7.9166951, -11.5235968, 11.6509323

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9459314, upper bound: 17.9550704
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9459315, upper bound: 17.9550704
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.9617697, 5.2819810, -1.7363042, 4.7071791, -6.6689487, 7.0182848
1: -5.4064646, 8.0788879, -4.7643337, 7.2455111, -12.6519756, 12.8432217
2: -3.3930345, 7.3429985, -2.9881930, 6.5437622, -9.9367962, 10.3311920
3: -6.0497794, 8.9081545, -5.3302326, 7.9612885, -14.0110683, 14.2383871
4: -3.9171157, 8.8984909, -3.4578595, 7.9166951, -11.8338089, 12.3563499

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 7

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.0099237, 5.3995810, -1.7046560, 4.6108918, -6.6208153, 7.1042371
1: -5.5343218, 8.2522211, -4.6781445, 7.0940909, -12.6284122, 12.9303646
2: -3.4881237, 7.4808788, -2.9285951, 6.4071131, -9.8952370, 10.4094734
3: -6.1995683, 9.0983219, -5.2297997, 7.7859783, -13.9855461, 14.3281212
4: -4.0267377, 9.0553761, -3.3814113, 7.7537327, -11.7804689, 12.4367857

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9548673, upper bound: 17.9611425
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9551261, upper bound: 17.9612822
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.0099237, 5.3995810, -1.7624259, 4.7696519, -6.7795753, 7.1620069
1: -5.5343218, 8.2522211, -4.8347473, 7.3370724, -12.8713942, 13.0869684
2: -3.4881237, 7.4808788, -3.0330470, 6.6300502, -10.1181736, 10.5139256
3: -6.1995683, 9.0983219, -5.4089136, 8.0649805, -14.2645473, 14.5072336
4: -4.0267377, 9.0553761, -3.5089464, 8.0202703, -12.0470085, 12.5643206

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9550230, upper bound: 17.9615093
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9552818, upper bound: 17.9616491
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.1024954, 5.6391344, -1.7046560, 4.6108918, -6.7133875, 7.3437896
1: -5.7942529, 8.5928297, -4.6781445, 7.0940909, -12.8883438, 13.2709742
2: -3.6448104, 7.8338251, -2.9285951, 6.4071131, -10.0519228, 10.7624197
3: -6.4846878, 9.5106916, -5.2297997, 7.7859783, -14.2706661, 14.7404909
4: -4.2082725, 9.4921455, -3.3814113, 7.7537327, -11.9620047, 12.8735571

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9315875, upper bound: 17.9596546
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9315875, upper bound: 17.9614347
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A2_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.1024954, 5.6391344, -1.7624259, 4.7696519, -6.8721476, 7.4015598
1: -5.7942529, 8.5928297, -4.8347473, 7.3370724, -13.1313248, 13.4275770
2: -3.6448104, 7.8338251, -3.0330470, 6.6300502, -10.2748604, 10.8668718
3: -6.4846878, 9.5106916, -5.4089136, 8.0649805, -14.5496674, 14.9196053
4: -4.2082725, 9.4921455, -3.5089464, 8.0202703, -12.2285423, 13.0010920

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -17.9570393, upper bound: 17.9604692
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A1_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -17.9570393, upper bound: 17.9614582
time: 0.62 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.52 + 418.32 = 420.84 seconds
