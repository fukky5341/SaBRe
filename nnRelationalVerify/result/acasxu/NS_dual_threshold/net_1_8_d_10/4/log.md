## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.134888934


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.6358709, 7.2639017, -8.6358709, 7.2639017, -15.8997726, 15.8997726)
1: (-32.9253387, 26.6185799, -32.9253387, 26.6185799, -59.5439186, 59.5439186)
2: (-17.6399612, 27.3018894, -17.6399612, 27.3018894, -44.9418449, 44.9418449)
3: (-29.9307785, 24.8325806, -29.9307785, 24.8325806, -54.7633591, 54.7633591)
4: (-22.0805244, 27.8851662, -22.0805244, 27.8851662, -49.9656906, 49.9656906)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.95 + 2.27 = 3.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -42.1770660, upper bound: 42.1770660

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1428570, upper bound: 42.1708969
time: 0.52 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210
time: 0.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.10 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 4, lower bound: -42.1428570, upper bound: 42.1708969
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.10
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -8.6358709, 7.2639017, -8.4596863, 7.1113238, -15.7471943, 15.7235880
1: -32.9253387, 26.6185799, -32.2558517, 26.0723686, -58.9977036, 58.8744316
2: -17.6399612, 27.3018894, -17.2642612, 26.7137508, -44.3537140, 44.5661469
3: -29.9307785, 24.8325806, -29.3213120, 24.3124962, -54.2432709, 54.1538887
4: -22.0805244, 27.8851662, -21.6306686, 27.2836914, -49.3642120, 49.5158348

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1428570, upper bound: 42.1708969
time: 0.58 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1426285, upper bound: 42.1494132
time: 0.51 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -8.5989227, 7.2360892, -9.1637344, 7.6889644, -16.2878876, 16.3998203
1: -32.7832947, 26.5194092, -35.0688629, 28.2125988, -60.9958954, 61.5882721
2: -17.5722504, 27.2011757, -18.6883316, 28.7631664, -46.3354149, 45.8895035
3: -29.7999458, 24.7408047, -31.8515282, 26.3088284, -56.1087723, 56.5923309
4: -21.9847946, 27.7842197, -23.4624557, 29.3769855, -51.3617783, 51.2466736

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1420652, upper bound: 42.1091245
time: 0.51 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210
time: 0.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.16 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -42.1428570, upper bound: 42.1708969
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -42.1426285, upper bound: 42.1494132
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -42.1420652, upper bound: 42.1091245
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 4, lower bound: -42.1424210, upper bound: 42.1424210

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -8.6358709, 7.2639017, -8.2933187, 6.9742584, -15.6101274, 15.5572205
1: -32.9253387, 26.6185799, -31.6038723, 25.5721340, -58.4974747, 58.2224503
2: -17.6399612, 27.3018894, -16.9427586, 26.2260132, -43.8659668, 44.2446442
3: -29.9307785, 24.8325806, -28.7310162, 23.8492527, -53.7800293, 53.5635986
4: -22.0805244, 27.8851662, -21.1968632, 26.7868099, -48.8673325, 49.0820312

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1093320, upper bound: 42.1490574
time: 0.63 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1093320, upper bound: 42.1494132
time: 0.67 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -8.5574675, 7.2025161, -8.7134962, 7.2355103, -15.7929783, 15.9160118
1: -32.6196442, 26.3958626, -33.2927551, 26.4121399, -59.0317764, 59.6886177
2: -17.4913406, 27.0824890, -17.5778656, 27.2043705, -44.6957092, 44.6603546
3: -29.6511002, 24.6269283, -30.2742939, 24.7073841, -54.3584824, 54.9012222
4: -21.8756447, 27.6637421, -22.2384758, 27.6935310, -49.5691757, 49.9022179

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_B2_B1

### Relational analysis result of NS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1417704, upper bound: 42.1494030
time: 0.82 seconds

## Relational analysis of NS_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_B1

### Relational analysis result of NS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1424283, upper bound: 42.1424904
time: 0.58 seconds

## Relational analysis of NS_B1_B2_B2

### Relational analysis result of NS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1426020, upper bound: 42.1493808
time: 0.52 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -8.5989227, 7.2360892, -9.0269470, 7.5751252, -16.1740475, 16.2630329
1: -32.7832947, 26.5194092, -34.5322113, 27.7942753, -60.5775681, 61.0516205
2: -17.5722504, 27.2011757, -18.4265404, 28.3571224, -45.9293747, 45.6277122
3: -29.7999458, 24.7408047, -31.3686218, 25.9227657, -55.7227058, 56.1094284
4: -21.9847946, 27.7842197, -23.1083031, 28.9631805, -50.9479752, 50.8925247

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1087687
time: 0.48 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1091245
time: 0.52 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -8.5202522, 7.1745110, -8.9121161, 7.3865962, -15.9068489, 16.0866280
1: -32.4765701, 26.2960396, -34.1363716, 27.0254211, -59.5019913, 60.4323959
2: -17.4231968, 26.9810696, -17.9496460, 27.7107887, -45.1339874, 44.9307098
3: -29.5193253, 24.5345383, -31.0244980, 25.1958065, -54.7151337, 55.5590363
4: -21.7792358, 27.5620956, -22.7767200, 28.1984425, -49.9776764, 50.3388138

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1091245, upper bound: 42.1420652
time: 0.56 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1091245, upper bound: 42.1424210
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.65 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1093320, upper bound: 42.1490574
NS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1093320, upper bound: 42.1494132
NS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1424283, upper bound: 42.1424904
NS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1426020, upper bound: 42.1493808
NS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1087687
NS_B2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1091245
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1091245, upper bound: 42.1420652
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 4, lower bound: -42.1091245, upper bound: 42.1424210

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -8.4692030, 7.1270952, -8.2933187, 6.9742584, -15.4434614, 15.4204130
1: -32.2723083, 26.1194496, -31.6038723, 25.5721340, -57.8444443, 57.7233200
2: -17.3193321, 26.8134594, -16.9427586, 26.2260132, -43.5453453, 43.7562103
3: -29.3394032, 24.3703098, -28.7310162, 23.8492527, -53.1886559, 53.1013260
4: -21.6460209, 27.3883381, -21.1968632, 26.7868099, -48.4328308, 48.5852013

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_A1_B1

### Relational analysis result of NS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1086498, upper bound: 42.1490774
time: 0.60 seconds

## Relational analysis of NS_B1_B1_A1_B2

### Relational analysis result of NS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1086483, upper bound: 42.1602948
time: 0.68 seconds

## BFS NS instance: NS_B1_B1_A2

### Backsubstitution after applying NS history:
0: -8.8602457, 7.3549442, -8.2933187, 6.9742584, -15.8345032, 15.6482630
1: -33.8532410, 26.8415222, -31.6038723, 25.5721340, -59.4253769, 58.4453964
2: -17.8718166, 27.6687317, -16.9427586, 26.2260132, -44.0978203, 44.6114883
3: -30.7879772, 25.1082878, -28.7310162, 23.8492527, -54.6372299, 53.8393021
4: -22.6179657, 28.1568451, -21.1968632, 26.7868099, -49.4047775, 49.3537064

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1093915, upper bound: 42.1690408
time: 0.90 seconds

## Relational analysis of NS_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B1

### Relational analysis result of NS_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1092846, upper bound: 42.1688956
time: 0.56 seconds

## Relational analysis of NS_B1_B1_A2_B2

### Relational analysis result of NS_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1092846, upper bound: 42.1708903
time: 0.58 seconds

## BFS NS instance: NS_B1_B2_B1

### Backsubstitution after applying NS history:
0: -8.5574675, 7.2025161, -8.0391760, 6.7095132, -15.2669811, 15.2416916
1: -32.6196442, 26.3958626, -30.6760406, 24.5716076, -57.1912537, 57.0719032
2: -17.4913406, 27.0824890, -16.2474995, 25.2017174, -42.6930580, 43.3299789
3: -29.6511002, 24.6269283, -27.8779030, 22.9841671, -52.6352654, 52.5048294
4: -21.8756447, 27.6637421, -20.5025425, 25.7676125, -47.6432571, 48.1662827

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_B1_B1

### Relational analysis result of NS_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1359601, upper bound: 42.1413332
time: 0.65 seconds

## Relational analysis of NS_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_B1_A1

### Relational analysis result of NS_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424117
time: 0.67 seconds

## Relational analysis of NS_B1_B2_B1_A2

### Relational analysis result of NS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424904
time: 0.53 seconds

## BFS NS instance: NS_B1_B2_B2

### Backsubstitution after applying NS history:
0: -8.2378216, 6.9429421, -8.8497143, 7.3543906, -15.5922127, 15.7926550
1: -31.3875084, 25.4571056, -33.7861099, 26.9046745, -58.2921791, 59.2432175
2: -16.8616028, 26.1132927, -17.8904190, 27.5339260, -44.3955307, 44.0037117
3: -28.5322914, 23.7514591, -30.7347507, 25.1340694, -53.6663589, 54.4862061
4: -21.0587158, 26.6850395, -22.6292763, 27.9977207, -49.0564346, 49.3143158

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_B2_B2_A1

### Relational analysis result of NS_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1384216, upper bound: 42.1493021
time: 0.62 seconds

## Relational analysis of NS_B1_B2_B2_A2

### Relational analysis result of NS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1384216, upper bound: 42.1493808
time: 0.57 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -8.4325409, 7.0995564, -8.9121161, 7.3865962, -15.8191376, 16.0116730
1: -32.1313896, 26.0213261, -34.1363716, 27.0254211, -59.1568108, 60.1576920
2: -17.2523384, 26.7138424, -17.9496460, 27.7107887, -44.9631271, 44.6634827
3: -29.2095242, 24.2795219, -31.0244980, 25.1958065, -54.4053307, 55.3040161
4: -21.5509720, 27.2885113, -22.7767200, 28.1984425, -49.7494087, 50.0652313

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1420652
time: 0.58 seconds

## Relational analysis of NS_B2_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1420652
time: 0.57 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -8.8182955, 7.3202605, -8.9121161, 7.3865962, -16.2048912, 16.2323761
1: -33.6927490, 26.7180290, -34.1363716, 27.0254211, -60.7181702, 60.8543930
2: -17.7876453, 27.5408783, -17.9496460, 27.7107887, -45.4984360, 45.4905167
3: -30.6416149, 24.9917488, -31.0244980, 25.1958065, -55.8374214, 56.0162468
4: -22.5108376, 28.0258331, -22.7767200, 28.1984425, -50.7092819, 50.8025513

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1424210
time: 0.52 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1424210
time: 0.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.41 seconds
NS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1086498, upper bound: 42.1490774
NS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1086483, upper bound: 42.1602948
NS_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1092846, upper bound: 42.1688956
NS_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1092846, upper bound: 42.1708903
NS_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424117
NS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424904
NS_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1384216, upper bound: 42.1493021
NS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1384216, upper bound: 42.1493808
NS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1420652
NS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1420652
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1424210
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.41
Output dim: 4, lower bound: -42.1087687, upper bound: 42.1424210

## BFS NS instance: NS_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.4692030, 7.1270952, -8.0216084, 6.7661204, -15.2353230, 15.1487036
1: -32.2723083, 26.1194496, -30.5634289, 24.8550777, -57.1273842, 56.6828728
2: -17.3193321, 26.8134594, -16.3621063, 25.4245720, -42.7439041, 43.1755638
3: -29.3394032, 24.3703098, -27.7487297, 23.1794968, -52.5188980, 52.1190376
4: -21.6460209, 27.3883381, -20.4769688, 25.9988651, -47.6448860, 47.8653069

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B1_A1_B1_A1

### Relational analysis result of NS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1086498, upper bound: 42.1490774
time: 0.61 seconds

## Relational analysis of NS_B1_B1_A1_B1_A2

### Relational analysis result of NS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1086498, upper bound: 42.1490774
time: 0.64 seconds

## BFS NS instance: NS_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.4692030, 7.1270952, -8.3126698, 7.0029864, -15.4721889, 15.4397640
1: -32.2723083, 26.1194496, -31.6496372, 25.6961651, -57.9684715, 57.7690849
2: -17.3193321, 26.8134594, -17.0221462, 26.3306942, -43.6500244, 43.8355942
3: -29.3394032, 24.3703098, -28.7570572, 23.9612827, -53.3006859, 53.1273651
4: -21.6460209, 27.3883381, -21.2366543, 26.8970623, -48.5430832, 48.6249924

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A1_B2_B1

### Relational analysis result of NS_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1083943, upper bound: 42.1511306
time: 0.62 seconds

## Relational analysis of NS_B1_B1_A1_B2_B2

### Relational analysis result of NS_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1086416, upper bound: 42.1602850
time: 0.66 seconds

## BFS NS instance: NS_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8602457, 7.3549442, -8.0307550, 6.7466135, -15.6068573, 15.3856964
1: -33.8532410, 26.8415222, -30.5918827, 24.7256145, -58.5788574, 57.4334030
2: -17.8718166, 27.6687317, -16.3989315, 25.4140530, -43.2858620, 44.0676537
3: -30.7879772, 25.1082878, -27.8172894, 23.0657597, -53.8537292, 52.9255753
4: -22.6179657, 28.1568451, -20.5222931, 25.9333649, -48.5513306, 48.6791382

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1234648, upper bound: 42.1649438
time: 0.55 seconds

## Relational analysis of NS_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1288485, upper bound: 42.1676556
time: 0.65 seconds

## BFS NS instance: NS_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.8602457, 7.3549442, -8.2552805, 6.9380341, -15.7982798, 15.6102247
1: -33.8532410, 26.8415222, -31.4587631, 25.4531250, -59.3063583, 58.3002853
2: -17.8718166, 27.6687317, -16.8641701, 26.0559254, -43.9277306, 44.5328941
3: -30.7879772, 25.1082878, -28.6033974, 23.7263508, -54.5143204, 53.7116852
4: -22.6179657, 28.1568451, -21.1234169, 26.6204662, -49.2384300, 49.2802620

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1092300, upper bound: 42.1703679
time: 0.82 seconds

## Relational analysis of NS_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1235639, upper bound: 42.1684657
time: 0.55 seconds

## Relational analysis of NS_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1289431, upper bound: 42.1699608
time: 0.58 seconds

## BFS NS instance: NS_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -7.8663201, 6.6567278, -8.0391760, 6.7095132, -14.5758333, 14.6959028
1: -29.9449348, 24.4546108, -30.6760406, 24.5716076, -54.5165405, 55.1306534
2: -16.1596031, 25.0472775, -16.2474995, 25.2017174, -41.3613205, 41.2947693
3: -27.1923218, 22.8111057, -27.8779030, 22.9841671, -50.1764870, 50.6890106
4: -20.0860844, 25.6169701, -20.5025425, 25.7676125, -45.8536873, 46.1195068

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_B1_A1_A1

### Relational analysis result of NS_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424117
time: 0.70 seconds

## Relational analysis of NS_B1_B2_B1_A1_A2

### Relational analysis result of NS_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424117
time: 0.65 seconds

## BFS NS instance: NS_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9910927, 7.5069184, -8.0391760, 6.7095132, -15.7006044, 15.5460949
1: -34.2945976, 27.5289803, -30.6760406, 24.5716076, -58.8662033, 58.2050209
2: -18.2817383, 28.1817150, -16.2474995, 25.2017174, -43.4834557, 44.4292068
3: -31.1813965, 25.6959438, -27.8779030, 22.9841671, -54.1655655, 53.5738449
4: -23.0182095, 28.7222137, -20.5025425, 25.7676125, -48.7858124, 49.2247505

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_B1_A2_A1

### Relational analysis result of NS_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424904
time: 0.62 seconds

## Relational analysis of NS_B1_B2_B1_A2_A2

### Relational analysis result of NS_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1424904
time: 0.64 seconds

## BFS NS instance: NS_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -7.8663201, 6.6567278, -8.8497143, 7.3543906, -15.2207108, 15.5064402
1: -29.9449348, 24.4546108, -33.7861099, 26.9046745, -56.8496056, 58.2407227
2: -16.1596031, 25.0472775, -17.8904190, 27.5339260, -43.6935272, 42.9376984
3: -27.1923218, 22.8111057, -30.7347507, 25.1340694, -52.3263931, 53.5458527
4: -20.0860844, 25.6169701, -22.6292763, 27.9977207, -48.0838051, 48.2462387

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_B2_B2_A1_A1

### Relational analysis result of NS_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1218612, upper bound: 42.1434428
time: 0.58 seconds

## Relational analysis of NS_B1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_B2_A1_A1

### Relational analysis result of NS_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1493021
time: 0.74 seconds

## Relational analysis of NS_B1_B2_B2_A1_A2

### Relational analysis result of NS_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1493021
time: 0.53 seconds

## BFS NS instance: NS_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9910736, 7.5069032, -8.8497143, 7.3543906, -16.3454609, 16.3566170
1: -34.2945137, 27.5289383, -33.7861099, 26.9046745, -61.1991844, 61.3150482
2: -18.2817154, 28.1816578, -17.8904190, 27.5339260, -45.8156395, 46.0720749
3: -31.1813202, 25.6959095, -30.7347507, 25.1340694, -56.3153915, 56.4306602
4: -23.0181561, 28.7221622, -22.6292763, 27.9977207, -51.0158768, 51.3514404

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B1_B2_B2_A2_B1

### Relational analysis result of NS_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1278563, upper bound: 42.1468611
time: 0.84 seconds

## Relational analysis of NS_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_B2_B2_A2_A1

### Relational analysis result of NS_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1482659
time: 0.65 seconds

## Relational analysis of NS_B1_B2_B2_A2_A2

### Relational analysis result of NS_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -42.1382479, upper bound: 42.1482659
time: 0.75 seconds

## BFS NS instance: NS_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -8.2933187, 6.9742584, -8.9121161, 7.3865962, -15.6799145, 15.8863735
1: -31.6038723, 25.5721340, -34.1363716, 27.0254211, -58.6292953, 59.7085037
2: -16.9427586, 26.2260132, -17.9496460, 27.7107887, -44.6535492, 44.1756439
3: -28.7310162, 23.8492527, -31.0244980, 25.1958065, -53.9268227, 54.8737488
4: -21.1968632, 26.7868099, -22.7767200, 28.1984425, -49.3953056, 49.5635300

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 25

## BFS NS instance: NS_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -9.0269470, 7.5751252, -8.9121161, 7.3865962, -16.4135418, 16.4872417
1: -34.5322113, 27.7942753, -34.1363716, 27.0254211, -61.5576324, 61.9306488
2: -18.4265404, 28.3571224, -17.9496460, 27.7107887, -46.1373291, 46.3067627
3: -31.3686218, 25.9227657, -31.0244980, 25.1958065, -56.5644302, 56.9472580
4: -23.1083031, 28.9631805, -22.7767200, 28.1984425, -51.3067474, 51.7398987

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_B2_A1_A2_A1

### Relational analysis result of NS_B2_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -42.0970085, upper bound: 42.1280680
time: 0.59 seconds

## Relational analysis of NS_B2_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -8.7134962, 7.2355103, -8.9121161, 7.3865962, -16.1000919, 16.1476250
1: -33.2927551, 26.4121399, -34.1363716, 27.0254211, -60.3181763, 60.5485077
2: -17.5778656, 27.2043705, -17.9496460, 27.7107887, -45.2886543, 45.1540146
3: -30.2742939, 24.7073841, -31.0244980, 25.1958065, -55.4701004, 55.7318802
4: -22.2384758, 27.6935310, -22.7767200, 28.1984425, -50.4369202, 50.4702530

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

## BFS NS instance: NS_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -8.5974884, 7.1295424, -8.9121161, 7.3865962, -15.9840851, 16.0416584
1: -32.8762245, 26.0751133, -34.1363716, 27.0254211, -59.9016457, 60.2114868
2: -17.3459415, 26.8145332, -17.9496460, 27.7107887, -45.0567284, 44.7641754
3: -29.8998280, 24.3132706, -31.0244980, 25.1958065, -55.0956345, 55.3377647
4: -21.9635506, 27.2787189, -22.7767200, 28.1984425, -50.1619949, 50.0554390

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.22 + 87.11 = 90.33 seconds
